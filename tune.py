#!/usr/bin/env python3
"""
tune.py — Lightweight grid search over Learning Rate and Projection Dimension.

Grid:
    Learning Rate:    [1e-4, 5e-5]
    Projection Dim:   [256, 512]

Each combination trains for up to 5 epochs with early stopping (patience=2
on validation loss).  Results are printed to the console and the best
parameter set is saved to best_params.txt.
"""

import argparse
import itertools
import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ImageDataset, TextDataset
from model import UnpairedMultimodalLearner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search grid
# ---------------------------------------------------------------------------
LEARNING_RATES = [1e-4, 5e-5]
PROJ_DIMS = [256, 512]
TUNE_EPOCHS = 5
PATIENCE = 2


# ---------------------------------------------------------------------------
# Helpers (mirrored from train.py so tune.py is self-contained)
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cycle(loader):
    """Infinitely cycle through a DataLoader."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (top-1 accuracy, average CE loss) over *loader* (image-only)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(image=images)
        running_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total if total else 0.0, running_loss / total if total else 0.0


# ---------------------------------------------------------------------------
# Single tuning run
# ---------------------------------------------------------------------------

def run_trial(lr, proj_dim, num_classes, train_img_loader, text_loader,
              val_img_loader, device, seed, batch_size):
    """Train one (lr, proj_dim) combination and return (val_acc, val_loss, epochs_run)."""
    torch.manual_seed(seed)

    model = UnpairedMultimodalLearner(
        num_classes=num_classes,
        proj_dim=proj_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(len(train_img_loader), len(text_loader))
    text_iter = cycle(text_loader)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, TUNE_EPOCHS + 1):
        model.train()
        epoch_img_loss = 0.0
        epoch_txt_loss = 0.0
        t0 = time.time()
        img_iter = cycle(train_img_loader)

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()

            # --- Image forward ---
            img_batch = next(img_iter)
            images = img_batch["image"].to(device)
            img_labels = img_batch["label"].to(device)

            logits_img = model(image=images)
            loss_img = criterion(logits_img, img_labels)

            # --- Text forward ---
            txt_batch = next(text_iter)
            input_ids = txt_batch["input_ids"].to(device)
            attn_mask = txt_batch["attention_mask"].to(device)
            txt_labels = txt_batch["label"].to(device)

            logits_txt = model(input_ids=input_ids, attention_mask=attn_mask)
            loss_txt = criterion(logits_txt, txt_labels)

            # --- Combined backward + step ---
            total_loss = loss_img + loss_txt
            total_loss.backward()
            optimizer.step()

            epoch_img_loss += loss_img.item()
            epoch_txt_loss += loss_txt.item()

        elapsed = time.time() - t0
        avg_img = epoch_img_loss / steps_per_epoch
        avg_txt = epoch_txt_loss / steps_per_epoch

        val_acc, val_loss = evaluate(model, val_img_loader, device)
        logger.info(
            "  epoch %d/%d (%.0fs) — img_loss=%.4f  txt_loss=%.4f  "
            "val_acc=%.4f  val_loss=%.4f",
            epoch, TUNE_EPOCHS, elapsed, avg_img, avg_txt, val_acc, val_loss,
        )

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info(
                    "  Early stop at epoch %d (no val-loss improvement for %d epochs)",
                    epoch, PATIENCE,
                )
                break

    # Free GPU memory before next trial
    del model, optimizer
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return best_val_acc, best_val_loss, epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./best_params.txt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(args.seed)

    device = get_device()
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Data (loaded once, shared across all trials)
    # ------------------------------------------------------------------
    images_dir = os.path.join(args.data_dir, "images")
    text_dir = os.path.join(args.data_dir, "text")

    full_image_ds = ImageDataset(images_dir)
    text_ds = TextDataset(text_dir, max_length=args.max_length)
    num_classes = len(full_image_ds.classes)

    n = len(full_image_ds)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_img_ds, val_img_ds, _ = random_split(
        full_image_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_img_loader = DataLoader(
        train_img_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_img_loader = DataLoader(
        val_img_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    text_loader = DataLoader(
        text_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    logger.info("Train images: %d | Val images: %d | Text: %d | Classes: %d",
                len(train_img_ds), len(val_img_ds), len(text_ds), num_classes)

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    grid = list(itertools.product(LEARNING_RATES, PROJ_DIMS))
    results = []

    header = f"{'LR':>10} {'Proj Dim':>10} {'Val Acc':>10} {'Val Loss':>10} {'Epochs':>8}"
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(f"  Grid Search — {len(grid)} combinations x up to {TUNE_EPOCHS} epochs")
    print(f"{separator}")
    print(header)
    print(separator)

    for trial_idx, (lr, proj_dim) in enumerate(grid, 1):
        logger.info("Trial %d/%d — lr=%g  proj_dim=%d", trial_idx, len(grid), lr, proj_dim)

        val_acc, val_loss, epochs_run = run_trial(
            lr=lr,
            proj_dim=proj_dim,
            num_classes=num_classes,
            train_img_loader=train_img_loader,
            text_loader=text_loader,
            val_img_loader=val_img_loader,
            device=device,
            seed=args.seed,
            batch_size=args.batch_size,
        )

        results.append({
            "lr": lr,
            "proj_dim": proj_dim,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "epochs": epochs_run,
        })

        print(f"{lr:>10g} {proj_dim:>10d} {val_acc:>10.4f} {val_loss:>10.4f} {epochs_run:>8d}")

    print(separator)

    # ------------------------------------------------------------------
    # Best combination
    # ------------------------------------------------------------------
    best = max(results, key=lambda r: r["val_acc"])

    print(f"\nBest combination:")
    print(f"  Learning Rate:    {best['lr']}")
    print(f"  Projection Dim:   {best['proj_dim']}")
    print(f"  Val Accuracy:     {best['val_acc']:.4f}")
    print(f"  Val Loss:         {best['val_loss']:.4f}")
    print(f"  Epochs Run:       {best['epochs']}")

    # ------------------------------------------------------------------
    # Save to file
    # ------------------------------------------------------------------
    with open(args.output, "w") as f:
        f.write("Unpaired Multimodal Learner — Best Hyperparameters\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Learning Rate:      {best['lr']}\n")
        f.write(f"Projection Dim:     {best['proj_dim']}\n")
        f.write(f"Validation Acc:     {best['val_acc']:.4f}\n")
        f.write(f"Validation Loss:    {best['val_loss']:.4f}\n")
        f.write(f"Epochs Run:         {best['epochs']}\n")
        f.write(f"\n{'=' * 50}\n")
        f.write("Full Grid Results\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"{'LR':>12} {'Proj Dim':>10} {'Val Acc':>10} {'Val Loss':>10} {'Epochs':>8}\n")
        f.write("-" * 52 + "\n")
        for r in results:
            f.write(f"{r['lr']:>12g} {r['proj_dim']:>10d} {r['val_acc']:>10.4f} "
                    f"{r['val_loss']:>10.4f} {r['epochs']:>8d}\n")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
