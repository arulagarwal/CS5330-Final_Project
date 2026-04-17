#!/usr/bin/env python3
"""
train.py — Training loop for the Unpaired Multimodal Learner.

The classifier is initialized with zero-shot text anchors (text_anchors.pt),
so the cross-modal synergy is baked into the weights.  Training optimizes
only on image batches with Cross-Entropy loss.
Saves the best checkpoint (by validation image accuracy) to checkpoints/.
Runs a final top-1 accuracy evaluation on the held-out image test split.
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import ImageDataset, get_train_transform, get_eval_transform
from model import UnpairedMultimodalLearner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    """Return top-1 accuracy and average CE loss over *loader* (image-only)."""
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
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data (image-only — text synergy is via zero-shot classifier init)
    # ------------------------------------------------------------------
    images_dir = os.path.join(args.data_dir, "images")

    # Two dataset instances so train gets augmentations, eval gets deterministic crops
    full_train_ds = ImageDataset(images_dir, transform=get_train_transform())
    full_eval_ds = ImageDataset(images_dir, transform=get_eval_transform())

    # Shuffled index split (70 / 15 / 15)
    n = len(full_train_ds)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_img_ds = Subset(full_train_ds, indices[:n_train])
    val_img_ds = Subset(full_eval_ds, indices[n_train:n_train + n_val])
    test_img_ds = Subset(full_eval_ds, indices[n_train + n_val:])

    train_img_loader = DataLoader(
        train_img_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_img_loader = DataLoader(
        val_img_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    test_img_loader = DataLoader(
        test_img_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    logger.info(
        "Splits — train: %d | val: %d | test: %d",
        len(train_img_ds), len(val_img_ds), len(test_img_ds),
    )

    # ------------------------------------------------------------------
    # Model / zero-shot init / optimizer / loss
    # ------------------------------------------------------------------
    model = UnpairedMultimodalLearner(num_classes=len(full_train_ds.classes)).to(device)
    model.zero_shot_init(args.anchors)
    model.to(device)

    if args.freeze_anchors:
        model.classifier.weight.requires_grad = False
        logger.info("Text anchors (classifier weights) are FROZEN.")
    else:
        logger.info("Text anchors (classifier weights) are UNFROZEN (Latent Drift possible).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    # ------------------------------------------------------------------
    # Training loop (image-only)
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_img_loader)

    logger.info("Training for %d epochs (%d steps/epoch)", args.epochs, steps_per_epoch)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_img_loader, 1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(image=images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % args.log_interval == 0:
                avg = epoch_loss / step
                logger.info(
                    "Epoch %d  step %d/%d  loss=%.4f",
                    epoch, step, steps_per_epoch, avg,
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / steps_per_epoch
        logger.info("Epoch %d done in %.1fs — avg loss=%.4f", epoch, elapsed, avg_loss)

        # --- Validation ---
        val_acc, val_loss = evaluate(model, val_img_loader, device)
        logger.info("Epoch %d val — acc=%.4f  loss=%.4f", epoch, val_acc, val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.info("Saved new best model (val_acc=%.4f) → %s", val_acc, best_path)

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    logger.info("Loading best checkpoint for test evaluation: %s", best_path)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_acc, test_loss = evaluate(model, test_img_loader, device)
    print(f"\n{'='*50}")
    print(f"  Test top-1 accuracy: {test_acc:.4f}")
    print(f"  Test loss:           {test_loss:.4f}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the Unpaired Multimodal Learner.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--anchors", default="./text_anchors.pt",
                        help="Path to zero-shot text anchors (default: ./text_anchors.pt)")
    parser.add_argument("--freeze-anchors", action="store_true",
                        help="Freeze the classifier weights (text anchors) during training.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save the best model.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
