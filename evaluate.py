# python evaluate.py
# tests checkpoints/clean_best.pt and checkpoints/robust_best.pt on the test split,
# saves tables and plots to results/
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import (CKPT_DIR, LOG_DIR, device, evaluate, fgsm, gaussian_noise,
                   get_loaders, load_model, pgd, set_seed)

OUT = Path("results")
EPS_LIST = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]   # in 1/255
PGD_EPS_LIST = [0, 1, 2, 4, 8, 16, 24]                          # in 1/255, PGD is ~10x slower
PGD_STEPS = 10
SIGMA_LIST = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
MODELS = ["clean", "robust"]


def fixed_noise(sigma):
    return lambda x, y: gaussian_noise(x, sigma=(sigma, sigma), p=1.0)


def find_limit(eps_list, accs, threshold):
    # first eps where accuracy drops to threshold, linear interpolation between points
    for i in range(1, len(accs)):
        if accs[i] <= threshold:
            e0, e1, a0, a1 = eps_list[i - 1], eps_list[i], accs[i - 1], accs[i]
            if a0 == a1:
                return e1
            return e0 + (a0 - threshold) * (e1 - e0) / (a0 - a1)
    return None


def plot_curves(x, curves, xlabel, title, path, threshold=None):
    plt.figure(figsize=(7, 4.5))
    for name, ys in curves.items():
        plt.plot(x, ys, marker="o", label=name)
    if threshold is not None:
        plt.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold {threshold:.0%}")
    plt.xlabel(xlabel)
    plt.ylabel("test accuracy")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fgsm_vs_pgd(fgsm_acc, pgd_acc, steps, threshold, path):
    # solid line = FGSM, dashed = PGD, same color for the same model
    plt.figure(figsize=(7, 4.5))
    for i, name in enumerate(fgsm_acc):
        color = f"C{i}"
        plt.plot(EPS_LIST, fgsm_acc[name], marker="o", color=color, label=f"{name}, FGSM")
        plt.plot(PGD_EPS_LIST, pgd_acc[name], marker="s", ls="--", color=color,
                 label=f"{name}, PGD-{steps}")
    plt.axhline(threshold, color="gray", ls=":", lw=1, label=f"threshold {threshold:.0%}")
    plt.axhline(0.10, color="lightgray", ls=":", lw=1, label="random guess 10%")
    plt.xlabel("eps (1/255)")
    plt.ylabel("test accuracy")
    plt.title("FGSM vs PGD")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    found = False
    for name in MODELS:
        path = LOG_DIR / f"{name}.csv"
        if not path.exists():
            continue
        found = True
        with open(path) as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        for ax, key in zip(axes, ["val_clean", "val_fgsm", "val_noise"]):
            ax.plot(ep, [float(r[key]) for r in rows], marker="o", label=name)
            ax.set_title(key)
    if not found:
        plt.close(fig)
        return
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_examples(models, loader, names, eps_show):
    # test split is sorted by class, take images spread over it to get different classes
    ds = loader.dataset
    step = len(ds) // 6
    items = [ds[i * step] for i in range(6)]
    x = torch.stack([it[0] for it in items]).to(device)
    y = torch.tensor([it[1] for it in items]).to(device)
    ref = models.get("robust") or next(iter(models.values()))

    set_seed()
    rows = [("clean", x), ("noise 0.08", fixed_noise(0.08)(x, y))]
    for e in eps_show:
        rows.append((f"fgsm {e}/255", fgsm(ref, x, y, e / 255)))

    fig, axes = plt.subplots(len(rows), len(x), figsize=(2.6 * len(x), 2.9 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        with torch.no_grad():
            preds = {m: model(imgs).argmax(1) for m, model in models.items()}
        for c in range(len(x)):
            ax = axes[r, c]
            ax.imshow(imgs[c].permute(1, 2, 0).cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            text = "\n".join(f"{m}: {names[preds[m][c]]}" for m in preds)
            ok = all(preds[m][c] == y[c] for m in preds)
            ax.set_title(text, fontsize=7, color="green" if ok else "red")
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle("true: " + ", ".join(names[i] for i in y.tolist()) +
                 "\n(fgsm images are made against the robust model)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    # 0.5: the model is wrong more often than right. 0.10 (random guess) is never
    # reached by single-step FGSM, so it gives no limit at all
    p.add_argument("--threshold", type=float, default=0.50,
                   help="accuracy below which the model counts as broken")
    p.add_argument("--pgd-steps", type=int, default=PGD_STEPS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)

    models = {}
    for name in MODELS:
        path = CKPT_DIR / f"{name}_best.pt"
        if path.exists():
            models[name] = load_model(path)
        else:
            print(f"{path} not found, skipping {name}")
    if not models:
        print("no models to test, train them first")
        return

    _, _, test_loader, names = get_loaders(args.batch_size, args.workers)
    print(f"test images: {len(test_loader.dataset)}, device: {device}")

    fgsm_acc, pgd_acc, noise_acc, summary = {}, {}, {}, []
    for name, model in models.items():
        print(f"\n{name}")

        accs = []
        for e in EPS_LIST:
            acc = evaluate(model, test_loader, lambda x, y: fgsm(model, x, y, e / 255))
            accs.append(acc)
            print(f"  fgsm eps {e:>2}/255: {acc:.4f}")
        fgsm_acc[name] = accs

        accs = []
        for e in PGD_EPS_LIST:
            set_seed()   # PGD has a random start
            acc = evaluate(model, test_loader,
                           lambda x, y: pgd(model, x, y, e / 255, steps=args.pgd_steps))
            accs.append(acc)
            print(f"  pgd-{args.pgd_steps} eps {e:>2}/255: {acc:.4f}")
        pgd_acc[name] = accs

        accs = []
        for s in SIGMA_LIST:
            set_seed()
            accs.append(evaluate(model, test_loader, fixed_noise(s)))
            print(f"  noise sigma {s:.2f}: {accs[-1]:.4f}")
        noise_acc[name] = accs

        # noise with the same params as in training: sigma ~ U(0.01, 0.08), every image
        set_seed()
        noise_train = evaluate(model, test_loader, lambda x, y: gaussian_noise(x, p=1.0))

        def fmt_limit(eps_list, accs, threshold):
            lim = find_limit(eps_list, accs, threshold)
            return f"{lim:.1f}/255" if lim is not None else f">{eps_list[-1]}/255"

        summary.append(dict(
            model=name,
            clean=round(fgsm_acc[name][0], 4),
            fgsm_4=round(fgsm_acc[name][EPS_LIST.index(4)], 4),
            fgsm_8=round(fgsm_acc[name][EPS_LIST.index(8)], 4),
            fgsm_16=round(fgsm_acc[name][EPS_LIST.index(16)], 4),
            pgd_4=round(pgd_acc[name][PGD_EPS_LIST.index(4)], 4),
            pgd_8=round(pgd_acc[name][PGD_EPS_LIST.index(8)], 4),
            pgd_16=round(pgd_acc[name][PGD_EPS_LIST.index(16)], 4),
            noise=round(noise_train, 4),
            # main limit: FGSM, accuracy below the threshold
            eps_limit_fgsm=fmt_limit(EPS_LIST, fgsm_acc[name], args.threshold),
            # model keeps less than half of its own clean accuracy
            eps_limit_fgsm_rel=fmt_limit(EPS_LIST, fgsm_acc[name], fgsm_acc[name][0] / 2),
            # same threshold under the stronger attack
            eps_limit_pgd=fmt_limit(PGD_EPS_LIST, pgd_acc[name], args.threshold),
            # model is as bad as random guessing (10 classes) under PGD
            eps_limit_pgd_10=fmt_limit(PGD_EPS_LIST, pgd_acc[name], 0.10),
        ))

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    with open(OUT / "fgsm_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps_255"] + list(fgsm_acc))
        for i, e in enumerate(EPS_LIST):
            w.writerow([e] + [round(fgsm_acc[m][i], 4) for m in fgsm_acc])

    with open(OUT / "pgd_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps_255"] + list(pgd_acc))
        for i, e in enumerate(PGD_EPS_LIST):
            w.writerow([e] + [round(pgd_acc[m][i], 4) for m in pgd_acc])

    with open(OUT / "noise_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma"] + list(noise_acc))
        for i, s in enumerate(SIGMA_LIST):
            w.writerow([s] + [round(noise_acc[m][i], 4) for m in noise_acc])

    plot_curves(EPS_LIST, fgsm_acc, "eps (1/255)", "FGSM attack", OUT / "fgsm_curve.png", args.threshold)
    plot_fgsm_vs_pgd(fgsm_acc, pgd_acc, args.pgd_steps, args.threshold, OUT / "fgsm_vs_pgd.png")
    plot_curves(SIGMA_LIST, noise_acc, "sigma", "Gaussian noise", OUT / "noise_curve.png")
    plot_training()
    plot_examples(models, test_loader, names, eps_show=[4, 8, 16])

    print(f"\nthreshold: {args.threshold:.0%}")
    for r in summary:
        print(r)
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()