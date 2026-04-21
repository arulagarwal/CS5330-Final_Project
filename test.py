#!/usr/bin/env python3
"""
test.py — Latent-space analysis for the Unpaired Multimodal Learner.

Loads frozen checkpoint weights, extracts 512-d image embeddings for a
handful of selected car classes, pairs them with the per-class classifier
weight rows (the text anchors the image encoder was trained against),
reduces the combined set to 2-D with t-SNE, and saves a scatter plot.
A companion isolated "halo" plot is written for a single class.

Authors: Arul Agarwal, Anirudh Bakare, and Utkarsh Milind Khursale
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
from torch.utils.data import DataLoader, Subset

from dataset import ImageDataset, get_eval_transform
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
def extract_image_embeddings(model, image):
    """Run the image encoder and return 512-d embeddings before the classifier."""
    return model.image_encoder(image)                       # (B, 512)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Latent-space analysis of the Unpaired Multimodal Learner.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt",
                        help="Path to the model weights to load.")
    parser.add_argument("--output", type=str, default="./latent_space.png",
                        help="Path to save the generated t-SNE plot.")
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

    # ------------------------------------------------------------------
    # Datasets – identical shuffled 70/15/15 split using the shared seed
    # ------------------------------------------------------------------
    full_eval_ds = ImageDataset(images_dir, transform=get_eval_transform())

    # Shuffled index split (70/15/15) — seeded for reproducibility
    n = len(full_eval_ds)
    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_indices = indices[n_train + n_val:]

    test_img_ds = Subset(full_eval_ds, test_indices)

    # ------------------------------------------------------------------
    # Pick 5 classes that appear in the test split
    # ------------------------------------------------------------------
    test_labels = [full_eval_ds.samples[i][1] for i in test_indices]
    unique_test_labels = sorted(set(test_labels))
    chosen_labels = random.sample(unique_test_labels,
                                  min(args.n_classes, len(unique_test_labels)))
    chosen_set = set(chosen_labels)
    label_to_class = {idx: name for name,
                      idx in full_eval_ds.class_to_idx.items()}
    chosen_names = [label_to_class[l] for l in chosen_labels]
    label_order = sorted(chosen_labels)
    logger.info("Selected classes: %s", chosen_names)

    # ------------------------------------------------------------------
    # Gather image indices from the test split for the chosen classes
    # ------------------------------------------------------------------
    class_to_test_indices = {l: [] for l in chosen_labels}
    for sub_idx, global_idx in enumerate(test_indices):
        label = full_eval_ds.samples[global_idx][1]
        if label in chosen_set:
            class_to_test_indices[label].append(sub_idx)

    selected_img_indices = []
    for l in chosen_labels:
        pool = class_to_test_indices[l]
        cap = min(len(pool), args.max_images_per_class)
        selected_img_indices.extend(random.sample(pool, cap))

    img_subset = Subset(test_img_ds, selected_img_indices)
    img_loader = DataLoader(img_subset, batch_size=64, shuffle=False)

    logger.info("Image samples: %d", len(img_subset))

    # ------------------------------------------------------------------
    # Load model (frozen)
    # ------------------------------------------------------------------
    num_classes = len(full_eval_ds.classes)
    model = UnpairedMultimodalLearner(num_classes=num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ------------------------------------------------------------------
    # Extract 512-d image embeddings
    # ------------------------------------------------------------------
    img_embeds, img_labels = [], []
    for batch in img_loader:
        images = batch["image"].to(device)
        emb = extract_image_embeddings(model, images)
        img_embeds.append(emb.cpu())
        img_labels.extend(batch["label"].tolist())

    img_embeds = torch.cat(img_embeds)
    img_labels = np.array(img_labels)
    img_embeds = F.normalize(img_embeds, dim=1)

    # ------------------------------------------------------------------
    # Use classifier weight rows as the actual text anchors the image
    # encoder was trained against (one row per class).
    # ------------------------------------------------------------------
    anchors_all = F.normalize(
        model.classifier.weight.detach().cpu(), dim=1
    )                                                      # (num_classes, 512)
    anchor_rows = anchors_all[label_order]                 # (n_classes, 512)
    anchor_labels = np.array(label_order)

    # Sanity metric: cosine similarity between each image and its own-class
    # anchor vs. the mean similarity to all other selected-class anchors.
    own_sims, other_sims = [], []
    label_to_row = {l: i for i, l in enumerate(label_order)}
    for emb, lbl in zip(img_embeds, img_labels):
        own_sims.append(
            float((emb @ anchor_rows[label_to_row[int(lbl)]]).item()))
        other_idx = [i for l, i in label_to_row.items() if l != int(lbl)]
        if other_idx:
            other_sims.append(
                float((emb @ anchor_rows[other_idx].T).mean().item())
            )
    logger.info(
        "Cosine sim(image, own-class anchor):    mean=%.4f",
        float(np.mean(own_sims)),
    )
    logger.info(
        "Cosine sim(image, other-class anchors): mean=%.4f",
        float(np.mean(other_sims)) if other_sims else float("nan"),
    )
    logger.info("img_scale (classifier magnitude multiplier): %.4f",
                float(model.img_scale.detach().cpu().item()))

    # ------------------------------------------------------------------
    # Concatenate and t-SNE
    # ------------------------------------------------------------------
    all_embeddings = torch.cat([img_embeds, anchor_rows], dim=0)
    all_embeddings_np = all_embeddings.cpu().numpy()
    logger.info("Combined embedding shape: %s", all_embeddings_np.shape)

    all_labels = np.concatenate([img_labels, anchor_labels], axis=0)
    modality = np.array(
        ["Image"] * len(img_embeds) + ["Text"] * len(anchor_rows)
    )

    effective_perplexity = min(
        args.perplexity, max(5, len(all_embeddings_np) // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(all_embeddings_np)
    logger.info("t-SNE complete — output shape: %s", coords.shape)

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    MARKERS = {"Image": "o", "Text": "^"}
    PALETTE = plt.cm.tab10.colors

    # Build a short display name for each chosen label
    label_to_color = {l: PALETTE[i % len(PALETTE)]
                      for i, l in enumerate(label_order)}

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
            legend_mod = "Anchor" if mod_name == "Text" else mod_name
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[label_to_color[label_idx]],
                marker=marker,
                s=250 if mod_name == "Text" else 40,
                alpha=0.95 if mod_name == "Text" else 0.65,
                edgecolors="k" if mod_name == "Text" else "none",
                linewidths=1.2 if mod_name == "Text" else 0.6,
                label=f"{short_name(cname)} ({legend_mod})",
            )

    ax.set_title("Latent Space — L2-Normalized (t-SNE)",
                 fontsize=14, weight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    logger.info("Plot saved to %s", args.output)
    print(f"\nLatent-space plot saved to {args.output}")

    # ==========================================
    # Isolated single-class halo plot
    # ==========================================
    logger.info("Generating isolated single-class halo plot...")

    # Pick the first class from our randomly selected test classes
    target_label_idx = label_order[0]
    class_name = label_to_class[target_label_idx]
    short_cname = short_name(class_name)

    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # 1. Plot ONLY the images for the target class
    target_img_mask = (all_labels == target_label_idx) & (modality == "Image")
    ax2.scatter(
        coords[target_img_mask, 0], coords[target_img_mask, 1],
        c='#1f77b4', marker='o', s=120, alpha=0.6,
        edgecolors='white', linewidths=1.5, label='Images'
    )

    # 2. Plot ONLY the text anchor for the target class
    target_txt_mask = (all_labels == target_label_idx) & (modality == "Text")
    ax2.scatter(
        coords[target_txt_mask, 0], coords[target_txt_mask, 1],
        c='#d62728', marker='^', s=500, edgecolors='black',
        linewidths=2, label='Text Anchor'
    )

    ax2.set_title(
        f'The Unpaired Halo Effect:\n{short_cname}', fontsize=20, weight='bold', pad=20)
    ax2.legend(fontsize=14, loc='best', frameon=True, shadow=True)
    ax2.axis('off')

    fig2.tight_layout()
    halo_output = "halo_effect.png"
    fig2.savefig(halo_output, dpi=300, transparent=True)

    logger.info("Halo plot saved to %s", halo_output)
    print(f"Isolated halo plot saved to {halo_output}")


if __name__ == "__main__":
    main()
