"""
PyTorch ground-truth replica of PCX's VGG-7 backprop baseline on Tiny
ImageNet (arXiv 2407.01163, "Benchmarking Predictive Coding Networks --
Made Simple"). Pure PyTorch, no Deepity/pydeepity involved at all -- the
point is an independent, trustworthy number for THIS exact data pipeline
and THIS exact hardware, rather than trusting a number copied from a paper
that may have used different preprocessing.

Architecture, loss, optimizer, schedule and hyperparameters below are taken
directly from the paper's own released code
(github.com/liukidar/pcax, release v0.6.1,
examples/s4_1_discriminative_mode/VGG7/tinyimagenet/BP_SE.py +
VGG_BP_SE_tinyimagenet.yaml), not approximated from the paper's text/tables:

  - Architecture (6 conv layers + 1 linear classifier; padding and max-pool
    placement below are exact, verified by tracing the spatial dims all the
    way to the 512*4*4 the real code's Linear layer expects):
      Conv(3->128,   k3,s1,p1) -> LeakyReLU -> MaxPool(2,2)   # 56->28
      Conv(128->128, k3,s1,p1) -> LeakyReLU                   # 28
      Conv(128->256, k3,s1,p1) -> LeakyReLU -> MaxPool(2,2)   # 28->14
      Conv(256->256, k3,s1,p0) -> LeakyReLU                   # 14->12
      Conv(256->512, k3,s1,p1) -> LeakyReLU -> MaxPool(2,2)   # 12->6
      Conv(512->512, k3,s1,p0) -> LeakyReLU                   # 6->4
      flatten -> Linear(512*4*4, 200)
  - Loss: sum of squared error against a PLAIN one-hot target (no label
    smoothing) -- this is the "_SE" variant specifically, matching the
    reference accuracy number this script compares against.
  - Input: RandomCrop(56) + RandomHorizontalFlip(p=0.5) for train,
    CenterCrop(56) for val -- NOT the full 64x64 image. Normalized with
    the tinyimagenet PyPI package's own constants, which are plain
    ImageNet mean/std ([0.485,0.456,0.406]/[0.229,0.224,0.225]), not
    Tiny-ImageNet-specific statistics.
  - Optimizer: AdamW, warmup-cosine-decay schedule (init=lr, peak=1.1*lr
    over the first 10% of all steps, cosine decay to end=0.1*lr over the
    rest), batch_size=128, epochs=50. lr/wd below are the exact values
    from VGG_BP_SE_tinyimagenet.yaml (the result of the paper's own
    hyperparameter search for this architecture).
  - Reported result is BEST validation accuracy across all epochs, not
    the final epoch's, matching how the paper's own code reports it.

Reference number this is meant to reproduce: PCX's Table 1 reports
VGG-7 Backprop-SE on Tiny ImageNet at 46.08+-0.15% top-1.

Usage:
    python imagenet_pytorch_baseline.py [EPOCHS] [BATCH_SIZE]
(defaults: 50, 128, matching the paper exactly)
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from imagenet import DATA_DIR, IMG_SIZE, N_CHANNELS, load_wnids, load_train_set_cached, load_val_set_cached

CROP_SIZE = 56
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TinyImageNetDataset(Dataset):
    """Wraps the uint8 CHW arrays imagenet.py's own cached loaders already
    produce. Train: random 56x56 crop + random horizontal flip. Val: center
    56x56 crop. Both normalized with plain ImageNet mean/std, matching
    BP_SE.py's own transforms.Compose pipeline exactly."""

    def __init__(self, X_u8, y_idx, train: bool):
        self.X = X_u8.reshape(-1, N_CHANNELS, IMG_SIZE, IMG_SIZE)
        self.y = y_idx
        self.train = train
        self.margin = IMG_SIZE - CROP_SIZE

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (3, 64, 64) uint8

        if self.train:
            oh = np.random.randint(0, self.margin + 1)
            ow = np.random.randint(0, self.margin + 1)
            crop = img[:, oh:oh + CROP_SIZE, ow:ow + CROP_SIZE]
            if np.random.random() < 0.5:
                crop = crop[:, :, ::-1]
        else:
            oh = self.margin // 2
            ow = self.margin // 2
            crop = img[:, oh:oh + CROP_SIZE, ow:ow + CROP_SIZE]

        x = torch.from_numpy(np.ascontiguousarray(crop)).float() / 255.0
        return x, int(self.y[idx])


class VGG7(nn.Module):
    """Exact layer-for-layer port of BP_SE.py's ConvNet, see module
    docstring for the verified spatial trace and source."""

    def __init__(self, n_classes=200):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.fc = nn.Linear(512 * 4 * 4, n_classes)

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.act(self.conv2(x))
        x = self.pool3(self.act(self.conv3(x)))
        x = self.act(self.conv4(x))
        x = self.pool5(self.act(self.conv5(x)))
        x = self.act(self.conv6(x))
        x = x.flatten(1)
        return self.fc(x)


def se_loss(output, one_hot_target):
    """Sum of squared error against a plain one-hot target, matching
    BP_SE.py's se_loss() exactly -- NOT cross-entropy, and no eps
    smoothing (unlike imagenet.py's own to_one_hot())."""
    return (output - one_hot_target).pow(2).sum(dim=1).mean()


def main():
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    LR = 1.585e-4
    WD = 2.098e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device found -- this will be extremely slow.")

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    train_ds = TinyImageNetDataset(X_train_u8, y_train_idx, train=True)
    val_ds = TinyImageNetDataset(X_val_u8, y_val_idx, train=False)

    def worker_init_fn(worker_id):
        # DataLoader workers inherit numpy's RNG state via fork(); without
        # this, every worker produces the IDENTICAL "random" crop/flip
        # sequence, silently reducing augmentation diversity.
        np.random.seed(torch.initial_seed() % 2**32 + worker_id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=True,
                            worker_init_fn=worker_init_fn)

    model = VGG7(N_CLASSES).to(device)
    mean = MEAN.to(device)
    std = STD.to(device)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))
    peak_lr = 1.1 * LR
    end_lr = 0.1 * LR

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    def lr_at_step(step):
        # optax.warmup_cosine_decay_schedule: linear warmup LR->peak over
        # warmup_steps, then cosine decay peak->end over the remaining
        # (total_steps - warmup_steps) steps. Matches BP_SE.py's schedule
        # exactly (exponent=1.0, i.e. standard cosine).
        if step < warmup_steps:
            return LR + (peak_lr - LR) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cosine

    print(f"\n*** PyTorch VGG-7 ground truth, Tiny ImageNet, Backprop-SE ***")
    print(f"Replicating PCX's own BP_SE.py exactly: {EPOCHS} epochs, "
          f"batch_size={BATCH_SIZE}, lr={LR:.6g} (warmup-cosine to peak "
          f"{peak_lr:.6g}, end {end_lr:.6g}), wd={WD:.6g}, AdamW, "
          f"squared-error loss, 56x56 crops, LeakyReLU.")
    print(f"Reference: PCX reports 46.08+-0.15% top-1 for this exact setup.\n")

    global_step = 0
    best_acc = 0.0
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = (x - mean) / std

            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step)

            one_hot = F.one_hot(y, N_CLASSES).float()
            out = model(x)
            loss = se_loss(out, one_hot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        model.eval()
        correct1, correct5, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = (x - mean) / std
                out = model(x)
                top5 = out.topk(5, dim=1).indices
                correct1 += (top5[:, 0] == y).sum().item()
                correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
                total += y.size(0)

        acc1 = 100.0 * correct1 / total
        acc5 = 100.0 * correct5 / total
        best_acc = max(best_acc, acc1)
        elapsed = time.perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | "
              f"Train loss: {epoch_loss/n_batches:.4f} | "
              f"Val top1: {acc1:.2f}% | Val top5: {acc5:.2f}% | "
              f"Best top1: {best_acc:.2f}% | lr={lr_at_step(global_step):.2e}")

    print(f"\n=== Result ===")
    print(f"PyTorch VGG-7 ground truth, Tiny ImageNet, Backprop-SE: "
          f"best top-1 = {best_acc:.2f}%")
    print(f"Reference (PCX paper, same exact architecture/hyperparameters): "
          f"46.08+-0.15%")


if __name__ == "__main__":
    main()
