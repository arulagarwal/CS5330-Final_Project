#!/usr/bin/env python3
"""
test.py — Latent-space analysis for the Unpaired Multimodal Learner.

Loads frozen checkpoint weights, extracts 512-d shared-backbone embeddings
for both image and text samples across 5 selected car classes, reduces to
2-D with t-SNE, and saves a scatter plot to latent_space.png.
"""

import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, random_split, Subset

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


@torch.no_grad()
def extract_embeddings(model, image=None, input_ids=None,
                       attention_mask=None):
    """Run the modality encoder and return 512-d embeddings before the classifier."""
    if image is not None:
        return model.image_encoder(image)                   # (B, 512)
    if attention_mask is None and input_ids is not None:
        attention_mask = torch.ones_like(input_ids)
    return model.text_encoder(input_ids, attention_mask)    # (B, 512)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Latent-space analysis of the Unpaired Multimodal Learner.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pt")
    parser.add_argument("--output", default="./latent_space_normalized.png")
    parser.add_argument("--n-classes", type=int, default=5,
                        help="Number of car classes to visualize")
    parser.add_argument("--max-images-per-class", type=int, default=20,
                        help="Cap on image samples per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    logger.info("Using device: %s", device)

    images_dir = os.path.join(args.data_dir, "images")
    text_dir = os.path.join(args.data_dir, "text")

    # ------------------------------------------------------------------
    # Datasets – replicate the same train/val/test split used in train.py
    # ------------------------------------------------------------------
    full_image_ds = ImageDataset(images_dir)
    text_ds = TextDataset(text_dir)

    n = len(full_image_ds)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    _, _, test_img_ds = random_split(
        full_image_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # ------------------------------------------------------------------
    # Pick 5 classes that appear in the test split
    # ------------------------------------------------------------------
    test_labels = [full_image_ds.samples[i][1] for i in test_img_ds.indices]
    unique_test_labels = sorted(set(test_labels))
    chosen_labels = random.sample(unique_test_labels,
                                  min(args.n_classes, len(unique_test_labels)))
    chosen_set = set(chosen_labels)
    label_to_class = {idx: name for name, idx in full_image_ds.class_to_idx.items()}
    chosen_names = [label_to_class[l] for l in chosen_labels]
    logger.info("Selected classes: %s", chosen_names)

    # ------------------------------------------------------------------
    # Gather image indices from the test split for the chosen classes
    # ------------------------------------------------------------------
    class_to_test_indices = {l: [] for l in chosen_labels}
    for sub_idx, global_idx in enumerate(test_img_ds.indices):
        label = full_image_ds.samples[global_idx][1]
        if label in chosen_set:
            class_to_test_indices[label].append(sub_idx)

    selected_img_indices = []
    for l in chosen_labels:
        pool = class_to_test_indices[l]
        cap = min(len(pool), args.max_images_per_class)
        selected_img_indices.extend(random.sample(pool, cap))

    img_subset = Subset(test_img_ds, selected_img_indices)
    img_loader = DataLoader(img_subset, batch_size=64, shuffle=False)

    # ------------------------------------------------------------------
    # Gather text samples for the same classes
    # ------------------------------------------------------------------
    text_label_map = {name: idx for name, idx in text_ds.class_to_idx.items()}
    selected_txt_indices = []
    for class_name in chosen_names:
        if class_name in text_label_map:
            txt_label = text_label_map[class_name]
            for i, (_, lbl) in enumerate(text_ds.samples):
                if lbl == txt_label:
                    selected_txt_indices.append(i)

    txt_subset = Subset(text_ds, selected_txt_indices)
    txt_loader = DataLoader(txt_subset, batch_size=64, shuffle=False)

    logger.info("Samples — images: %d, text: %d", len(img_subset), len(txt_subset))

    # ------------------------------------------------------------------
    # Load model (frozen)
    # ------------------------------------------------------------------
    num_classes = len(full_image_ds.classes)
    model = UnpairedMultimodalLearner(num_classes=num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ------------------------------------------------------------------
    # Extract 512-d backbone embeddings
    # ------------------------------------------------------------------
    img_embeds, img_labels = [], []
    for batch in img_loader:
        images = batch["image"].to(device)
        emb = extract_embeddings(model, image=images)
        img_embeds.append(emb.cpu())
        img_labels.extend(batch["label"].tolist())

    txt_embeds, txt_labels = [], []
    for batch in txt_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        emb = extract_embeddings(model, input_ids=ids, attention_mask=mask)
        txt_embeds.append(emb.cpu())
        txt_labels.extend(batch["label"].tolist())

    img_embeds = torch.cat(img_embeds)
    txt_embeds = torch.cat(txt_embeds)

    # L2-normalize both modalities to close the modality gap
    img_embeds = F.normalize(img_embeds, p=2, dim=1)
    txt_embeds = F.normalize(txt_embeds, p=2, dim=1)

    img_embeds = img_embeds.numpy()
    txt_embeds = txt_embeds.numpy()
    img_labels = np.array(img_labels)
    txt_labels = np.array(txt_labels)

    logger.info("Embedding shapes (L2-normalized) — images: %s, text: %s",
                img_embeds.shape, txt_embeds.shape)

    # ------------------------------------------------------------------
    # Map text labels (text_ds indexing) → image label space
    # ------------------------------------------------------------------
    txt_class_to_img_label = {}
    for name in chosen_names:
        txt_class_to_img_label[text_label_map[name]] = full_image_ds.class_to_idx[name]
    txt_labels_mapped = np.array([txt_class_to_img_label[l] for l in txt_labels])

    # ------------------------------------------------------------------
    # t-SNE reduction to 2-D
    # ------------------------------------------------------------------
    all_embeds = np.concatenate([img_embeds, txt_embeds], axis=0)
    all_labels = np.concatenate([img_labels, txt_labels_mapped], axis=0)
    modality = np.array(
        ["Image"] * len(img_embeds) + ["Text"] * len(txt_embeds)
    )

    effective_perplexity = min(args.perplexity, max(5, len(all_embeds) // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(all_embeds)
    logger.info("t-SNE complete — output shape: %s", coords.shape)

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    MARKERS = {"Image": "o", "Text": "^"}
    PALETTE = plt.cm.tab10.colors

    # Build a short display name for each chosen label
    label_order = sorted(chosen_labels)
    label_to_color = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(label_order)}

    # Shorten class names for the legend (e.g. "BMW M3 Coupe 2012" stays readable)
    def short_name(name):
        return name if len(name) <= 32 else name[:29] + "..."

    fig, ax = plt.subplots(figsize=(10, 8))

    for mod_name, marker in MARKERS.items():
        mask_mod = modality == mod_name
        for label_idx in label_order:
            mask_cls = all_labels == label_idx
            mask = mask_mod & mask_cls
            if not mask.any():
                continue
            cname = label_to_class[label_idx]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[label_to_color[label_idx]],
                marker=marker,
                s=70 if mod_name == "Text" else 40,
                alpha=0.85 if mod_name == "Text" else 0.65,
                edgecolors="k" if mod_name == "Text" else "none",
                linewidths=0.6,
                label=f"{short_name(cname)} ({mod_name})",
            )

    ax.set_title("Latent Space — L2-Normalized (t-SNE)", fontsize=14, weight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    logger.info("Plot saved to %s", args.output)
    print(f"\nLatent-space plot saved to {args.output}")


if __name__ == "__main__":
    main()
