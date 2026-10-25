"""
Training loop for LGG tumor segmentation.
Run: python src/train.py
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import LGGDataset, split_patients
from model import UNet


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=config.BCE_WEIGHT, dice_weight=config.DICE_WEIGHT):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + 1e-6
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6
        dice = (num / den).mean()
        return self.bce_weight * bce + self.dice_weight * (1 - dice)


@torch.no_grad()
def dice_score(logits, targets, eps=1e-6):
    probs = (torch.sigmoid(logits) > 0.5).float()
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (num / den).mean().item()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_p, val_p, _ = split_patients()
    print(f"Patients -> train: {len(train_p)} | val: {len(val_p)}")

    train_ds = LGGDataset(train_p, augment=True)
    val_ds = LGGDataset(val_p, augment=False)
    print(f"Slices   -> train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE, min_lr=config.MIN_LR,
    )

    best_dice = -1.0
    ckpt_path = config.OUTPUT_DIR / config.CHECKPOINT_NAME
    csv_path = config.OUTPUT_DIR / config.METRICS_CSV
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dice", "lr"])

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item() * imgs.size(0)
                val_dice += dice_score(logits, masks) * imgs.size(0)
        val_loss /= len(val_ds)
        val_dice /= len(val_ds)

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{config.EPOCHS} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | lr {current_lr:.2e}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                    f"{val_dice:.6f}", f"{current_lr:.2e}"])

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best model (val_dice={best_dice:.4f})")

    print(f"Training complete. Best val Dice: {best_dice:.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()