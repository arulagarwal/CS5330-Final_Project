#!/usr/bin/env python3
"""
train.py — Training loop for the Unpaired Multimodal Learner.

Alternates between image and text batches, each with Cross-Entropy loss.
Saves the best checkpoint (by validation image accuracy) to checkpoints/.
Runs a final top-1 accuracy evaluation on the held-out image test split.
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
# Helpers
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
    # Data
    # ------------------------------------------------------------------
    images_dir = os.path.join(args.data_dir, "images")
    text_dir = os.path.join(args.data_dir, "text")

    full_image_ds = ImageDataset(images_dir)
    text_ds = TextDataset(text_dir, max_length=args.max_length)

    # Train / val / test split for images (70 / 15 / 15)
    n = len(full_image_ds)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_img_ds, val_img_ds, test_img_ds = random_split(
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
    test_img_loader = DataLoader(
        test_img_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    text_loader = DataLoader(
        text_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    logger.info(
        "Splits — train images: %d | val images: %d | test images: %d | text: %d",
        len(train_img_ds), len(val_img_ds), len(test_img_ds), len(text_ds),
    )

    # ------------------------------------------------------------------
    # Model / optimizer / loss
    # ------------------------------------------------------------------
    model = UnpairedMultimodalLearner(num_classes=len(full_image_ds.classes)).to(device)
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
    # Training loop
    # ------------------------------------------------------------------
    # Cycle the shorter loader so both modalities run for the same number
    # of steps.  Each "step" = one image batch + one text batch.
    steps_per_epoch = max(len(train_img_loader), len(text_loader))

    text_iter = cycle(text_loader)

    logger.info("Training for %d epochs (%d steps/epoch)", args.epochs, steps_per_epoch)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_img_loss = 0.0
        epoch_txt_loss = 0.0
        img_steps = 0
        txt_steps = 0
        t0 = time.time()

        img_iter = cycle(train_img_loader)

        for step in range(1, steps_per_epoch + 1):
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
            img_steps += 1
            epoch_txt_loss += loss_txt.item()
            txt_steps += 1

            if step % args.log_interval == 0:
                avg_i = epoch_img_loss / img_steps
                avg_t = epoch_txt_loss / txt_steps
                logger.info(
                    "Epoch %d  step %d/%d  img_loss=%.4f  txt_loss=%.4f",
                    epoch, step, steps_per_epoch, avg_i, avg_t,
                )

        elapsed = time.time() - t0
        avg_img = epoch_img_loss / img_steps
        avg_txt = epoch_txt_loss / txt_steps
        logger.info(
            "Epoch %d done in %.1fs — avg img_loss=%.4f  avg txt_loss=%.4f",
            epoch, elapsed, avg_img, avg_txt,
        )

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
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--max-length", type=int, default=64)
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
