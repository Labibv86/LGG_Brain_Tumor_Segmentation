"""
Test-set evaluation and result plotting.
Run: python src/evaluate.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import LGGDataset, split_patients
from model import UNet


@torch.no_grad()
def batch_metrics(logits, targets, eps=1e-6):
    """Return dict of per-batch mean Dice / IoU / Precision / Recall."""
    probs = (torch.sigmoid(logits) > 0.5).float()
    tp = (probs * targets).sum(dim=(2, 3))
    fp = (probs * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - probs) * targets).sum(dim=(2, 3))

    dice = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    prec = ((tp + eps) / (tp + fp + eps)).mean().item()
    rec = ((tp + eps) / (tp + fn + eps)).mean().item()
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec}


def plot_curves():
    csv_path = config.OUTPUT_DIR / config.METRICS_CSV
    epochs, train_loss, val_loss, val_dice = [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_dice.append(float(row["val_dice"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(epochs, val_dice, color="tab:green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Dice"); axes[1].set_title("Val Dice")
    plt.tight_layout()
    out = config.OUTPUT_DIR / config.CURVES_NAME
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved curves -> {out}")


def plot_predictions(model, loader, device, n=6):
    model.eval()
    shown = 0
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs_d, masks_d = imgs.to(device), masks.to(device)
            logits = model(imgs_d)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            masks_np = masks.numpy()
            imgs_np = imgs.numpy()

            for i in range(imgs.size(0)):
                if shown >= n:
                    break

                # --- NEW: only plot slices that actually contain tumor ---
                if masks_np[i, 0].sum() == 0:
                    continue
                # ---------------------------------------------------------
                img = np.transpose(imgs_np[i], (1, 2, 0))
                gt = masks_np[i, 0]
                pr = preds[i, 0]
                overlay = img.copy()
                overlay[gt > 0.5] = overlay[gt > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
                overlay[pr > 0.5] = overlay[pr > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5

                axes[shown, 0].imshow(img); axes[shown, 0].set_title("Image"); axes[shown, 0].axis("off")
                axes[shown, 1].imshow(gt, cmap="gray"); axes[shown, 1].set_title("GT"); axes[shown, 1].axis("off")
                axes[shown, 2].imshow(pr, cmap="gray"); axes[shown, 2].set_title("Pred"); axes[shown, 2].axis("off")
                axes[shown, 3].imshow(overlay); axes[shown, 3].set_title("GT(green)/Pred(red)"); axes[shown, 3].axis("off")
                shown += 1
            if shown >= n:
                break


    plt.tight_layout()
    out = config.OUTPUT_DIR / config.PRED_GRID_NAME
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved prediction grid -> {out}")


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    _, _, test_p = split_patients()
    test_ds = LGGDataset(test_p, augment=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    print(f"Test patients: {len(test_p)} | test slices: {len(test_ds)}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(config.OUTPUT_DIR / config.CHECKPOINT_NAME, map_location=device))
    model.eval()

    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    n_batches = 0
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            m = batch_metrics(logits, masks)
            for k in totals:
                totals[k] += m[k]
            n_batches += 1

    print("\n=== Test results ===")
    for k, v in totals.items():
        print(f"{k.capitalize():10s}: {v / n_batches:.4f}")

    plot_curves()
    plot_predictions(model, test_loader, device, n=6)


if __name__ == "__main__":
    main()