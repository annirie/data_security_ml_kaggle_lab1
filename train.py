# python train.py --mode clean     baseline model
# python train.py --mode robust    FGSM + gaussian noise training
import argparse
import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import (CKPT_DIR, LOG_DIR, USE_AMP, Net, autocast, device, evaluate,
                   fgsm, fgsm_rs, gaussian_noise, get_loaders, load_checkpoint,
                   save_checkpoint, set_seed, write_csv)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["clean", "robust"], required=True)
    p.add_argument("--epochs", type=int, default=None, help="default: 10 for clean, 15 for robust")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=8, help="FGSM eps for training, in 1/255 units")
    p.add_argument("--warmup", type=int, default=2, help="epochs to ramp eps up from 0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=float, default=5, help="minutes between mid-epoch saves")
    p.add_argument("--fresh", action="store_true", help="ignore the old checkpoint and start over")
    return p.parse_args()


def train_step(model, opt, x, y, mode, eps):
    opt.zero_grad(set_to_none=True)

    if mode == "clean":
        with autocast():
            out = model(x)
            loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        return loss.item(), (out.argmax(1) == y).sum().item()

    # robust: half of the loss on noisy images, half on FGSM images
    x = gaussian_noise(x)
    x_adv = fgsm_rs(model, x, y, eps) if eps > 0 else x

    # two backward passes so only one graph is in memory at a time
    with autocast():
        out = model(x)
        loss_clean = F.cross_entropy(out, y)
    (0.5 * loss_clean).backward()
    with autocast():
        loss_adv = F.cross_entropy(model(x_adv), y)
    (0.5 * loss_adv).backward()
    opt.step()

    loss = 0.5 * (loss_clean.item() + loss_adv.item())
    return loss, (out.argmax(1) == y).sum().item()


def main():
    args = parse_args()
    set_seed()

    last_path = CKPT_DIR / f"{args.mode}_last.pt"
    best_path = CKPT_DIR / f"{args.mode}_best.pt"
    log_path = LOG_DIR / f"{args.mode}.csv"

    ckpt = None if args.fresh else load_checkpoint(last_path)
    if ckpt is not None:
        # keep settings from the first run, otherwise the lr schedule breaks
        for k in ("epochs", "batch_size", "lr", "eps", "warmup"):
            setattr(args, k, ckpt["args"][k])
    if args.epochs is None:
        args.epochs = 10 if args.mode == "clean" else 15

    train_loader, val_loader, _, _ = get_loaders(args.batch_size, args.workers)
    sampler = train_loader.sampler
    n_train = len(train_loader.dataset)
    n_batches = math.ceil(n_train / args.batch_size)

    model = Net(pretrained=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    state = dict(epoch=0, seen=0, loss_sum=0.0, correct=0, best=-1.0, history=[])

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        state = ckpt["state"]
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
        if device.type == "cuda" and ckpt["rng_cuda"] is not None:
            torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu())
        print(f"resuming: epoch {state['epoch'] + 1}, {state['seen']}/{n_train} images done")
    else:
        print("starting from scratch")

    print(f"device: {device}, bf16: {USE_AMP}, mode: {args.mode}, epochs: {args.epochs}, "
          f"batch: {args.batch_size}, eps: {args.eps}/255")

    eps_max = args.eps / 255

    def save_last():
        save_checkpoint(dict(
            model=model.state_dict(),
            opt=opt.state_dict(),
            sched=sched.state_dict(),
            state=state,
            args=vars(args),
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state() if device.type == "cuda" else None,
        ), last_path)

    try:
        while state["epoch"] < args.epochs:
            epoch = state["epoch"]
            sampler.set_epoch(epoch, state["seen"])
            model.train()
            t0 = last_save = time.time()

            bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}",
                       initial=state["seen"] // args.batch_size, total=n_batches)
            for x, y in bar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # ramp eps up during the first epochs, starting at full eps
                # from a clean pretrained model hurts accuracy a lot
                progress = epoch + state["seen"] / n_train
                eps = eps_max * min(1.0, progress / args.warmup) if args.warmup > 0 else eps_max

                loss, correct = train_step(model, opt, x, y, args.mode, eps)
                state["seen"] += y.numel()
                state["loss_sum"] += loss * y.numel()
                state["correct"] += correct
                bar.set_postfix(loss=f"{state['loss_sum'] / state['seen']:.3f}",
                                acc=f"{state['correct'] / state['seen']:.3f}")

                if time.time() - last_save > args.save_every * 60:
                    save_last()
                    last_save = time.time()
            bar.close()

            lr = opt.param_groups[0]["lr"]
            sched.step()

            val_clean = evaluate(model, val_loader)
            val_fgsm = evaluate(model, val_loader, lambda x, y: fgsm(model, x, y, eps_max))
            val_noise = evaluate(model, val_loader, lambda x, y: gaussian_noise(x, p=1.0))

            row = dict(
                epoch=epoch + 1,
                lr=round(lr, 6),
                train_loss=round(state["loss_sum"] / state["seen"], 4),
                train_acc=round(state["correct"] / state["seen"], 4),
                val_clean=round(val_clean, 4),
                val_fgsm=round(val_fgsm, 4),
                val_noise=round(val_noise, 4),
                minutes=round((time.time() - t0) / 60, 1),
            )
            state["history"].append(row)

            # clean model is picked by clean accuracy, robust one by the mean of all three.
            # skip warmup epochs for robust, eps is still small there
            score = val_clean if args.mode == "clean" else (val_clean + val_fgsm + val_noise) / 3
            is_best = score > state["best"] and (args.mode == "clean" or epoch + 1 >= args.warmup)
            if is_best:
                state["best"] = score
                save_checkpoint(dict(model=model.state_dict(), epoch=epoch + 1,
                                     mode=args.mode, eps=args.eps), best_path)

            state.update(epoch=epoch + 1, seen=0, loss_sum=0.0, correct=0)
            save_last()
            write_csv(log_path, state["history"])

            print(f"epoch {epoch + 1}: train acc {row['train_acc']:.3f}, val clean {val_clean:.3f}, "
                  f"fgsm {val_fgsm:.3f}, noise {val_noise:.3f}, {row['minutes']} min"
                  + ("  [best]" if is_best else ""))

    except KeyboardInterrupt:
        save_last()
        print("\nstopped, progress saved. run the same command to continue")
        return

    print(f"done, best model: {best_path}")


if __name__ == "__main__":
    main()
