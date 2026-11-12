# pip install kagglehub
import os
import random
import shutil
from pathlib import Path

import kagglehub

DATASET = "liusha249/imagenet10"
OUT_DIR = Path(os.environ.get("DATA_DIR", "data"))
VAL_RATIO = 0.2
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_images(d):
    return any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file())


# find the folder whose subfolders are classes
def find_class_root(root):
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        subdirs = [s for s in d.iterdir() if s.is_dir()]
        if len(subdirs) >= 2 and all(has_images(s) for s in subdirs):
            return d
    raise RuntimeError(f"no class folders found in {root}")


def copy_tree(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


path = Path(kagglehub.dataset_download(DATASET))
print("downloaded to", path)

dirs = {p.name.lower(): p for p in path.rglob("*") if p.is_dir()}
train_dir = dirs.get("train")
val_dir = dirs.get("val") or dirs.get("valid") or dirs.get("validation") or dirs.get("test")

OUT_DIR.mkdir(exist_ok=True)

if train_dir and val_dir:
    copy_tree(find_class_root(train_dir), OUT_DIR / "train")
    copy_tree(find_class_root(val_dir), OUT_DIR / "val")
else:
    class_root = find_class_root(path)
    rng = random.Random(SEED)
    for split in ("train", "val"):
        if (OUT_DIR / split).exists():
            shutil.rmtree(OUT_DIR / split)

    for cls in sorted(d for d in class_root.iterdir() if d.is_dir()):
        files = sorted(f for f in cls.iterdir() if f.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        n_val = int(len(files) * VAL_RATIO)
        for split, chunk in (("val", files[:n_val]), ("train", files[n_val:])):
            dst = OUT_DIR / split / cls.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in chunk:
                shutil.copy2(f, dst / f.name)

for split in ("train", "val"):
    print(split)
    for cls in sorted((OUT_DIR / split).iterdir()):
        print(" ", cls.name, len(list(cls.iterdir())))
