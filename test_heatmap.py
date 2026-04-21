#!/usr/bin/env python3
"""
test_heatmap.py — Cosine Similarity Heatmap for Modality Alignment.

Calculates the centroids of image embeddings for 5 test classes and computes
their raw cosine similarity against the corresponding text anchors. 
A strong diagonal line approaching 1.0 indicates a closed modality gap.
"""

import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
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
    return model.image_encoder(image)

def short_name(class_name):
    return class_name.split(' ')[0] if class_name else "Unknown"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cosine Similarity Heatmap")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--output", default='cosine_heatmap.png')
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    logger.info(f"Using device: {device}")

    # ==========================================
    # 1. LOAD DATA & SELECT CLASSES
    # ==========================================
    images_dir = os.path.join(args.data_dir, "images")
    transform = get_eval_transform()
    full_dataset = ImageDataset(images_dir, transform=transform)    
    
    all_classes = list(full_dataset.class_to_idx.keys())
    selected_classes = random.sample(all_classes, 10)
    selected_idx = [full_dataset.class_to_idx[c] for c in selected_classes]
    
    logger.info(f"Selected classes: {selected_classes}")

    indices_to_keep = []
    class_counts = {idx: 0 for idx in selected_idx}
    for i, (_, label) in enumerate(full_dataset.samples):
        if label in selected_idx and class_counts[label] < args.samples_per_class:
            indices_to_keep.append(i)
            class_counts[label] += 1

    subset = Subset(full_dataset, indices_to_keep)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    # ==========================================
    # 2. LOAD MODEL
    # ==========================================
    model = UnpairedMultimodalLearner(num_classes=len(full_dataset.class_to_idx)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ==========================================
    # 3. EXTRACT EMBEDDINGS & CALCULATE CENTROIDS
    # ==========================================
    img_embeds_list = []
    img_labels_list = []
    
    logger.info("Extracting image embeddings...")
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        embeds = extract_image_embeddings(model, images)
        img_embeds_list.append(embeds.cpu())
        img_labels_list.append(labels)

    img_embeds = torch.cat(img_embeds_list, dim=0)
    img_labels = torch.cat(img_labels_list, dim=0)

    # Calculate centroids for the selected image classes
    centroids = []
    for idx in selected_idx:
        class_mask = (img_labels == idx)
        class_embeds = img_embeds[class_mask]
        # Mean across the class samples
        centroid = class_embeds.mean(dim=0)
        centroids.append(centroid)
        
    centroids = torch.stack(centroids)  # (5, 512)

    # Get the corresponding text anchors
    text_anchors = model.classifier.weight.data.clone().cpu()
    selected_anchors = text_anchors[selected_idx]  # (5, 512)

    # ==========================================
    # 4. NORMALIZE & COMPUTE COSINE SIMILARITY
    # ==========================================
    # L2 normalize both centroids and anchors to ensure values are [-1, 1]
    centroids_norm = F.normalize(centroids, p=2, dim=-1)
    anchors_norm = F.normalize(selected_anchors, p=2, dim=-1)

    # Matrix multiplication of normalized vectors yields cosine similarity
    # Shape: (5_images, 512) @ (512, 5_texts) -> (5, 5)
    similarity_matrix = torch.matmul(centroids_norm, anchors_norm.T).numpy()

    # ==========================================
    # 5. PLOT HEATMAP
    # ==========================================
    logger.info("Generating Cosine Similarity Heatmap...")
    
    label_to_class = {idx: name for name, idx in full_dataset.class_to_idx.items()}
    short_names = [short_name(label_to_class[idx]) for idx in selected_idx]

    plt.figure(figsize=(10, 8))
    
    # Using a divergent colormap centered at 0
    sns.heatmap(
        similarity_matrix, 
        annot=True,               # Show values in cells
        fmt=".3f",                # 3 decimal places
        cmap="mako",              # Blue (negative) to Red (positive) divergent map
        center=0,
        xticklabels=short_names, 
        yticklabels=short_names,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title("Image Centroid vs Text Anchor Cosine Similarity", pad=20, fontsize=16)
    plt.xlabel("Text Anchors", fontsize=12, labelpad=10)
    plt.ylabel("Image Centroids", fontsize=12, labelpad=10)
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    logger.info("Complete. Saved to " + args.output)

if __name__ == "__main__":
    main()