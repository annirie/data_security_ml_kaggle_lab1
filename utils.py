import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, models, transforms

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CKPT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
SEED = 42
IMG_SIZE = 224
NUM_CLASSES = 10
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CLASS_NAMES = {
    "n02056570": "king penguin",
    "n02085936": "maltese dog",
    "n02128757": "snow leopard",
    "n02690373": "airliner",
    "n02692877": "airship",
    "n03095699": "container ship",
    "n04254680": "soccer ball",
    "n04285008": "sports car",
    "n04467665": "trailer truck",
    "n07747607": "orange",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


USE_AMP = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def autocast():
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_AMP)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- data ----------

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])


class ResumableSampler(Sampler):
    # same shuffle order for a given epoch, so after a restart
    # we can skip the samples that were already trained on
    def __init__(self, n, seed=SEED):
        self.n = n
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch, start=0):
        self.epoch = epoch
        self.start = start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.start:])

    def __len__(self):
        return self.n - self.start


def split_half(targets):
    # split data/val into val and test, half of each class goes to each
    rng = random.Random(SEED)
    val_idx, test_idx = [], []
    for c in sorted(set(targets)):
        idx = [i for i, t in enumerate(targets) if t == c]
        rng.shuffle(idx)
        half = len(idx) // 2
        val_idx += idx[:half]
        test_idx += idx[half:]
    return val_idx, test_idx


def get_loaders(batch_size=64, num_workers=4):
    train_set = datasets.ImageFolder(DATA_DIR / "train", train_tf)
    holdout = datasets.ImageFolder(DATA_DIR / "val", eval_tf)
    assert train_set.classes == holdout.classes

    val_idx, test_idx = split_half(holdout.targets)
    val_set = Subset(holdout, val_idx)
    test_set = Subset(holdout, test_idx)

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_set, sampler=ResumableSampler(len(train_set)), **kw)
    val_loader = DataLoader(val_set, shuffle=False, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)

    names = [CLASS_NAMES.get(c, c) for c in train_set.classes]
    return train_loader, val_loader, test_loader, names


# ---------- model ----------

class Net(nn.Module):
    # normalization is done inside, so inputs and eps are in [0, 1] pixel space
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, NUM_CLASSES)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def load_model(path):
    model = Net(pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------- noise and attacks ----------

def gaussian_noise(x, sigma=(0.01, 0.08), p=0.5):
    # for each image: with prob p add N(0, s^2), s ~ U(sigma)
    n = x.size(0)
    s = torch.empty(n, 1, 1, 1, device=x.device).uniform_(*sigma)
    mask = (torch.rand(n, 1, 1, 1, device=x.device) < p).float()
    return (x + torch.randn_like(x) * s * mask).clamp(0, 1)


def fgsm(model, x, y, eps):
    if eps == 0:
        return x
    x = x.clone().detach().requires_grad_(True)
    with autocast():
        loss = F.cross_entropy(model(x), y)
    grad, = torch.autograd.grad(loss, x)
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def fgsm_rs(model, x, y, eps, alpha=None):
    # FGSM with random start (Wong et al., 2020), used for training.
    # plain FGSM training often breaks suddenly (catastrophic overfitting)
    if alpha is None:
        alpha = 1.25 * eps
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).requires_grad_(True)
    with autocast():
        loss = F.cross_entropy(model(x + delta), y)
    grad, = torch.autograd.grad(loss, delta)
    delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
    return (x + delta).clamp(0, 1).detach()


def pgd(model, x, y, eps, steps=10, alpha=None):
    # PGD with random start (Madry et al., 2018), used only for evaluation:
    # a stronger multi-step attack to check that FGSM results are not
    # caused by gradient masking or catastrophic overfitting
    if eps == 0:
        return x
    if alpha is None:
        alpha = eps / 4
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        delta.requires_grad_(True)
        with autocast():
            loss = F.cross_entropy(model(x + delta), y)
        grad, = torch.autograd.grad(loss, delta)
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
        delta = ((x + delta).clamp(0, 1) - x).detach()
    return (x + delta).detach()


def evaluate(model, loader, attack=None):
    # attack: function (x, y) -> perturbed x, None means clean accuracy
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if attack is not None:
            x = attack(x, y)
        with torch.no_grad(), autocast():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


# ---------- checkpoints and logs ----------

def save_checkpoint(state, path):
    # write to a temp file and then rename,
    # so a power cut while saving doesn't break the previous checkpoint
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=True)


def write_csv(path, rows):
    # rewrites the whole file from the history stored in the checkpoint,
    # so the log never has duplicate or missing epochs after a restart
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)