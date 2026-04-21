#!/usr/bin/env python3
"""
download_data.py — Download Stanford Cars dataset and generate synthetic unpaired text descriptions.

Usage:
    python download_data.py [--data-dir ./data] [--skip-images] [--skip-text]
                            [--num-descriptions 5] [--seed 42]

Authors: Arul Agarwal, Anirudh Bakare, and Utkarsh Milind Khursale
"""

import argparse
import csv
import logging
import os
import random
import shutil
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIMARY_KAGGLE_DATASET = "jutrera/stanford-car-dataset-by-classes-folder"
FALLBACK_KAGGLE_DATASET = "eduardo4jesus/stanford-cars-dataset"

VEHICLE_TYPES = [
    "Crew Cab", "Club Coupe", "Regular Cab", "Extended Cab", "Super Cab",
    "SuperCab", "CrewMax", "Quad Cab", "Access Cab",
    "Convertible", "Hatchback", "Coupe", "Sedan", "SUV", "Wagon",
    "Van", "Minivan", "Pickup", "Cab", "Type-S", "IPL", "SRT-8", "SRT8",
    "GS", "SS", "GT", "Z06", "ZR1", "SRT", "Abarth",
]

COLORS = [
    "silver", "black", "white", "red", "blue", "dark gray", "pearl white",
    "midnight blue", "metallic green", "charcoal", "burgundy", "champagne",
]

FEATURES = [
    "chrome accents", "alloy wheels", "tinted windows", "roof rails",
    "dual exhaust tips", "LED headlights", "fog lamps", "sport suspension",
    "panoramic sunroof", "body-colored mirrors", "rear spoiler",
    "sculpted hood", "aggressive front fascia", "integrated running boards",
]

TEMPLATES = [
    "A {color} {year} {make_model} {body_type} with {feature1} and a sleek exterior profile.",
    "The {make_model} features a {body_type} body style typical of {year} designs, finished in {color}.",
    "This {body_type} from {make_model} has clean lines, {feature1}, and a modern {year}-era silhouette.",
    "A well-proportioned {year} {make_model} showcasing the classic {body_type} form factor with {feature2}.",
    "The {year} {make_model} {body_type} displays a bold front grille and {feature1}.",
    "Seen from the side, the {color} {make_model} {body_type} reveals smooth curves and {feature2}.",
    "A {year} {make_model} in {color}, combining the {body_type} layout with {feature1} and {feature2}.",
    "The rear of this {make_model} {body_type} features {feature1}, complementing its {color} paint.",
    "This {color} {year} {make_model} stands out with its {body_type} stance and {feature2}.",
    "An eye-catching {year} {make_model} {body_type} with {feature1}, {feature2}, and a {color} finish.",
    "The {make_model} {body_type} from {year} pairs a {color} exterior with {feature1}.",
    "Compact yet powerful, the {year} {make_model} {body_type} comes with {feature2} and a {color} body.",
    "A sporty {color} {year} {make_model} {body_type} equipped with {feature1} and {feature2}.",
    "The {year} {make_model} offers a refined {body_type} shape, {color} paint, and {feature1}.",
    "With its {color} exterior and {feature2}, the {year} {make_model} {body_type} is unmistakable.",
]

# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

def check_kaggle_credentials():
    """Return True if Kaggle credentials are available."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.isfile(kaggle_json):
        return True
    return False


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_organized_dataset():
    """Download the pre-organized Stanford Cars dataset from Kaggle. Returns the local path."""
    import kagglehub
    logger.info("Downloading pre-organized dataset: %s", PRIMARY_KAGGLE_DATASET)
    path = kagglehub.dataset_download(PRIMARY_KAGGLE_DATASET)
    logger.info("Downloaded to %s", path)
    return path


def download_raw_dataset():
    """Download the raw Stanford Cars dataset from Kaggle. Returns the local path."""
    import kagglehub
    logger.info("Downloading raw dataset: %s", FALLBACK_KAGGLE_DATASET)
    path = kagglehub.dataset_download(FALLBACK_KAGGLE_DATASET)
    logger.info("Downloaded to %s", path)
    return path


# ---------------------------------------------------------------------------
# Image organization
# ---------------------------------------------------------------------------

def _find_class_dirs(root):
    """Walk *root* and return a list of (class_name, dir_path) for directories that contain images."""
    class_dirs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        has_images = any(
            f.lower().endswith((".jpg", ".jpeg", ".png"))
            for f in filenames
        )
        if has_images:
            class_dirs.append((os.path.basename(dirpath), dirpath))
    return class_dirs


def organize_from_prebuilt(download_path, target_dir):
    """Copy class folders from the pre-organized Kaggle download into *target_dir*."""
    class_dirs = _find_class_dirs(download_path)
    if not class_dirs:
        raise RuntimeError(f"No image directories found under {download_path}")

    logger.info("Found %d class directories in pre-built dataset", len(class_dirs))
    for class_name, src_dir in class_dirs:
        dest = os.path.join(target_dir, class_name)
        if os.path.isdir(dest):
            # Merge: copy files that don't already exist
            for fname in os.listdir(src_dir):
                src_file = os.path.join(src_dir, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dest)
        else:
            shutil.copytree(src_dir, dest)


def organize_from_raw(download_path, target_dir):
    """Parse .mat annotations and organize images from the raw dataset into class folders."""
    import scipy.io
    import numpy as np

    # Locate annotation files
    meta_path = None
    annos_paths = []
    for dirpath, _dirs, files in os.walk(download_path):
        for f in files:
            full = os.path.join(dirpath, f)
            if f == "cars_meta.mat":
                meta_path = full
            elif f.endswith("_annos.mat"):
                annos_paths.append(full)

    if meta_path is None:
        raise RuntimeError(f"cars_meta.mat not found under {download_path}")
    if not annos_paths:
        raise RuntimeError(f"No annotation .mat files found under {download_path}")

    meta = scipy.io.loadmat(meta_path, squeeze_me=True)
    class_names = meta["class_names"]
    if isinstance(class_names, np.ndarray):
        class_names = [str(c) for c in class_names]

    logger.info("Loaded %d class names from %s", len(class_names), meta_path)

    # Find image directories (flat folders containing .jpg)
    image_dirs = set()
    for dirpath, _dirs, files in os.walk(download_path):
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
            image_dirs.add(dirpath)

    for annos_path in annos_paths:
        logger.info("Processing annotations from %s", annos_path)
        annos = scipy.io.loadmat(annos_path, squeeze_me=True)
        annotations = annos["annotations"]

        for ann in annotations:
            fname = str(ann["fname"])
            cls_idx = int(ann["class"]) - 1  # 1-indexed → 0-indexed
            class_name = class_names[cls_idx]

            # Find the image file
            src_file = None
            for img_dir in image_dirs:
                candidate = os.path.join(img_dir, fname)
                if os.path.isfile(candidate):
                    src_file = candidate
                    break

            if src_file is None:
                logger.warning("Image %s not found, skipping", fname)
                continue

            dest_dir = os.path.join(target_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_file, os.path.join(dest_dir, os.path.basename(fname)))


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def parse_class_name(class_name):
    """Extract make_model, body_type, and year from a Stanford Cars class name."""
    parts = class_name.rsplit(" ", 1)
    year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "unknown"
    remainder = parts[0] if year != "unknown" else class_name

    body_type = "car"
    for vtype in VEHICLE_TYPES:
        if remainder.lower().endswith(f" {vtype.lower()}"):
            body_type = vtype
            remainder = remainder[: -len(vtype)].rstrip()
            break

    return {"make_model": remainder, "body_type": body_type, "year": year}


def generate_descriptions(class_name, n=5):
    """Generate *n* synthetic visual-feature descriptions for a car class."""
    info = parse_class_name(class_name)
    descriptions = []
    chosen_templates = random.sample(TEMPLATES, min(n, len(TEMPLATES)))
    for tmpl in chosen_templates:
        desc = tmpl.format(
            make_model=info["make_model"],
            body_type=info["body_type"],
            year=info["year"],
            color=random.choice(COLORS),
            feature1=random.choice(FEATURES),
            feature2=random.choice(FEATURES),
        )
        descriptions.append(desc)
    return descriptions


def generate_text_dataset(class_names, output_dir, num_descriptions=5):
    """Generate CSV text-description files for all classes."""
    os.makedirs(output_dir, exist_ok=True)
    for class_name in sorted(class_names):
        descriptions = generate_descriptions(class_name, n=num_descriptions)
        csv_path = os.path.join(output_dir, f"{class_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["description"])
            writer.writeheader()
            for desc in descriptions:
                writer.writerow({"description": desc})
    logger.info("Generated %d text CSV files in %s", len(class_names), output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Stanford Cars dataset and generate synthetic text descriptions."
    )
    parser.add_argument("--data-dir", default="./data", help="Base data directory (default: ./data)")
    parser.add_argument("--skip-images", action="store_true", help="Skip image download")
    parser.add_argument("--skip-text", action="store_true", help="Skip text generation")
    parser.add_argument("--num-descriptions", type=int, default=5, help="Descriptions per class (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    random.seed(args.seed)

    images_dir = os.path.join(args.data_dir, "images")
    text_dir = os.path.join(args.data_dir, "text")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    logger.info("Data directories ready: %s, %s", images_dir, text_dir)

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------
    if not args.skip_images:
        # Check idempotency
        existing = [
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ]
        if len(existing) >= 100:
            logger.info("Found %d class folders in %s — skipping download.", len(existing), images_dir)
        else:
            if not check_kaggle_credentials():
                logger.error(
                    "Kaggle credentials not found. Please either:\n"
                    "  1. Place your kaggle.json at ~/.kaggle/kaggle.json, or\n"
                    "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
                    "Get your API token from https://www.kaggle.com/settings"
                )
                sys.exit(1)

            downloaded = False
            # Primary: pre-organized dataset
            try:
                path = download_organized_dataset()
                organize_from_prebuilt(path, images_dir)
                downloaded = True
            except Exception as e:
                logger.warning("Primary download failed: %s", e)

            # Fallback: raw dataset
            if not downloaded:
                try:
                    path = download_raw_dataset()
                    organize_from_raw(path, images_dir)
                    downloaded = True
                except Exception as e:
                    logger.warning("Fallback download failed: %s", e)

            if not downloaded:
                logger.error(
                    "Could not download the Stanford Cars dataset.\n"
                    "Please download it manually from Kaggle and place class folders in %s",
                    images_dir,
                )
                sys.exit(1)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    if not args.skip_text:
        class_names = sorted(
            d for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        )
        if not class_names:
            logger.error("No class folders found in %s — cannot generate text.", images_dir)
            sys.exit(1)

        generate_text_dataset(class_names, text_dir, num_descriptions=args.num_descriptions)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_classes = len([
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ])
    n_images = sum(
        len([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for _, _, files in os.walk(images_dir)
    )
    n_text = len([f for f in os.listdir(text_dir) if f.endswith(".csv")])
    print(f"\nDone! Summary:")
    print(f"  Classes:      {n_classes}")
    print(f"  Images:       {n_images}")
    print(f"  Text files:   {n_text}")


if __name__ == "__main__":
    main()
