# python main.py
# data -> clean model -> robust model -> evaluation.
# finished steps are skipped, unfinished training continues from its checkpoint,

import argparse
import subprocess
import sys
from pathlib import Path

from utils import CKPT_DIR, DATA_DIR, load_checkpoint

HERE = Path(__file__).resolve().parent
RESULTS = Path("results") / "summary.csv"


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *args]
    print("\n>>", " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd).returncode == 0


def data_ready():
    return all((DATA_DIR / s).is_dir() and len(list((DATA_DIR / s).iterdir())) == 10
               for s in ("train", "val"))


def training_done(mode):
    ckpt = load_checkpoint(CKPT_DIR / f"{mode}_last.pt")
    return ckpt is not None and ckpt["state"]["epoch"] >= ckpt["args"]["epochs"]


def eval_up_to_date():
    # results are newer than both best models
    if not RESULTS.exists():
        return False
    t = RESULTS.stat().st_mtime
    return all((CKPT_DIR / f"{m}_best.pt").stat().st_mtime < t for m in ("clean", "robust"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=None, help="only used when training starts")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--eval-again", action="store_true", help="redo evaluation even if results exist")
    args = p.parse_args()

    extra = []
    if args.batch_size is not None:
        extra += ["--batch-size", str(args.batch_size)]
    if args.workers is not None:
        extra += ["--workers", str(args.workers)]

    if data_ready():
        print("data: ok")
    elif not run("download_data.py"):
        sys.exit("download failed")

    for mode in ("clean", "robust"):
        if training_done(mode):
            print(f"{mode}: already trained, skipping")
            continue
        ok = run("train.py", "--mode", mode, *extra)
        if not ok or not training_done(mode):
            sys.exit(f"{mode} training not finished, run main.py again to continue")

    if eval_up_to_date() and not args.eval_again:
        print(f"evaluation: up to date, see {RESULTS.parent}/ (use --eval-again to redo)")
        return

    eval_args = ["--workers", str(args.workers)] if args.workers is not None else []
    if not run("evaluate.py", *eval_args):
        sys.exit("evaluation failed")
    print("\nall done")


if __name__ == "__main__":
    main()
