#!/usr/bin/env python3
"""
dataset.py — Unpaired image and text PyTorch datasets for multimodal learning.
Provides:
    ImageDataset  — loads images from data/images/<ClassName>/*.jpg
    TextDataset   — loads descriptions from data/text/<ClassName>.csv, tokenized with DistilBERT
    get_image_dataloader / get_text_dataloader — independent DataLoaders for unpaired training

Authors: Arul Agarwal, Anirudh Bakare, and Utkarsh Milind Khursale

"""

import csv
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import DistilBertTokenizer


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform():
    """Training augmentations: random crop + flip (standard ViT pipeline)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform():
    """Evaluation transform: deterministic resize + center crop."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Image Dataset
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Loads images organized in class subfolders under *image_dir*.

    Each subfolder name is treated as a class label. Returns a dict with
    ``"image"`` (tensor) and ``"label"`` (int).
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform or self.default_transform()

        # Build class → index mapping from sorted subfolder names
        class_names = sorted(
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.classes = class_names

        # Collect (path, label) pairs
        self.samples = []
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(image_dir, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if os.path.splitext(fname)[1].lower() in self.IMAGE_EXTENSIONS:
                    self.samples.append((os.path.join(class_dir, fname), idx))

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "label": label}


# ---------------------------------------------------------------------------
# Text Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Loads text descriptions from CSV files under *text_dir* and tokenizes
    them with DistilBERT.

    Each CSV corresponds to one class and has a ``description`` column.
    Returns a dict with ``"input_ids"``, ``"attention_mask"``, and ``"label"``.
    """

    def __init__(self, text_dir, tokenizer_name="distilbert-base-uncased", max_length=64):
        self.text_dir = text_dir
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

        # Build class → index mapping from sorted CSV basenames
        csv_files = sorted(glob(os.path.join(text_dir, "*.csv")))
        class_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.classes = class_names

        # Collect (description, label) pairs
        self.samples = []
        for csv_path, class_name in zip(csv_files, class_names):
            label = self.class_to_idx[class_name]
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append((row["description"], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
        }


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_image_dataloader(image_dir, batch_size=32, shuffle=True,
                         num_workers=2, transform=None):
    """Create an image DataLoader for unpaired training."""
    dataset = ImageDataset(image_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_text_dataloader(text_dir, batch_size=32, shuffle=True,
                        num_workers=2, tokenizer_name="distilbert-base-uncased",
                        max_length=64):
    """Create a text DataLoader for unpaired training."""
    dataset = TextDataset(text_dir, tokenizer_name=tokenizer_name, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the unpaired datasets.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    text_dir = os.path.join(args.data_dir, "text")

    print("Loading ImageDataset...")
    img_loader = get_image_dataloader(images_dir, batch_size=args.batch_size)
    print(f"  {len(img_loader.dataset)} images, {len(img_loader.dataset.classes)} classes")

    print("Loading TextDataset...")
    txt_loader = get_text_dataloader(text_dir, batch_size=args.batch_size)
    print(f"  {len(txt_loader.dataset)} descriptions, {len(txt_loader.dataset.classes)} classes")

    print("\nSample image batch:")
    batch = next(iter(img_loader))
    print(f"  image shape: {batch['image'].shape}")
    print(f"  labels:      {batch['label']}")

    print("\nSample text batch:")
    batch = next(iter(txt_loader))
    print(f"  input_ids shape:     {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels:              {batch['label']}")

    print("\nIndependent iteration (first 3 batches each):")
    img_iter = iter(img_loader)
    txt_iter = iter(txt_loader)
    for i in range(3):
        ib = next(img_iter, None)
        tb = next(txt_iter, None)
        if ib and tb:
            print(f"  step {i}: img batch {ib['image'].shape[0]}, txt batch {tb['input_ids'].shape[0]}")
