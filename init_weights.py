#!/usr/bin/env python3
"""
init_weights.py — Compute per-class text anchors for classifier initialization.

Passes all text descriptions through the frozen DistilBERT TextEncoder,
averages the 512-d [CLS] embeddings per class, L2-normalizes along dim=1,
and saves the [196, 512] tensor to disk.

Authors: Arul Agarwal, Anirudh Bakare, and Utkarsh Milind Khursale
"""

import argparse
import logging
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextEncoder

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def compute_text_anchors(text_dir, proj_dim=512, max_length=64, batch_size=64):
    """Compute L2-normalized per-class text embeddings.

    Returns
    -------
    anchors : Tensor
        Shape (num_classes, proj_dim), L2-normalized along dim=1.
    class_names : list[str]
        Ordered class names matching the row indices of *anchors*.
    """
    device = get_device()
    logger.info("Device: %s", device)

    # Frozen text encoder
    encoder = TextEncoder(proj_dim=proj_dim).to(device)
    encoder.eval()
    logger.info("Loaded frozen TextEncoder (proj_dim=%d)", proj_dim)

    # Text dataset
    text_ds = TextDataset(text_dir, max_length=max_length)
    num_classes = len(text_ds.classes)
    loader = DataLoader(text_ds, batch_size=batch_size, shuffle=False)

    logger.info("Text samples: %d | Classes: %d", len(text_ds), num_classes)

    # Accumulate embeddings per class
    class_embeds = defaultdict(list)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        # TextEncoder returns (B, proj_dim) — the [CLS] embedding
        cls_embeds = encoder(input_ids, attn_mask)         # (B, 512)

        for emb, label in zip(cls_embeds.cpu(), labels.tolist()):
            class_embeds[label].append(emb)

    # Average per class and stack into (num_classes, proj_dim)
    anchors = torch.zeros(num_classes, proj_dim)
    for label in range(num_classes):
        stacked = torch.stack(class_embeds[label])         # (N, 512)
        anchors[label] = stacked.mean(dim=0)

    # L2-normalize along the feature dimension
    anchors = F.normalize(anchors, dim=1)

    return anchors, text_ds.classes


def main():
    parser = argparse.ArgumentParser(
        description="Compute zero-shot text anchors for classifier initialization.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="./text_anchors.pt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    text_dir = os.path.join(args.data_dir, "text")
    anchors, class_names = compute_text_anchors(
        text_dir,
        proj_dim=args.proj_dim,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    logger.info("Anchor tensor shape: %s", anchors.shape)
    logger.info("Norm range: [%.4f, %.4f]",
                anchors.norm(dim=1).min().item(),
                anchors.norm(dim=1).max().item())

    torch.save({"anchors": anchors, "class_names": class_names}, args.output)
    print(f"Saved text anchors ({anchors.shape[0]} classes, {anchors.shape[1]}-d) "
          f"to {args.output}")


if __name__ == "__main__":
    main()
