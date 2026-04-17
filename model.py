#!/usr/bin/env python3
"""
model.py — Unpaired Multimodal Learner for Stanford Cars classification.

Architecture (following the reference UML paper):
    1. Image encoder  — ViT-Small/16 (timm) → Linear → 512-d pooled features
    2. Text encoder   — DistilBERT (HuggingFace) → Linear → 512-d [CLS] features
    3. Classification  — Linear → 196 classes, scaled by a learnable img_scale

The classifier can be initialized with zero-shot text anchors via zero_shot_init().
"""

import torch
import torch.nn as nn
import timm
from transformers import DistilBertModel


# ---------------------------------------------------------------------------
# Modality-specific encoders
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Pre-trained ViT-Small/16 with a linear projection to *proj_dim*."""

    def __init__(self, proj_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0,
        )
        backbone_dim = self.backbone.num_features  # 384 for vit_small
        self.proj = nn.Linear(backbone_dim, proj_dim)

    def forward(self, pixel_values):
        """pixel_values: (B, 3, 224, 224) → (B, num_patches+1, proj_dim)"""
        features = self.backbone.forward_features(pixel_values)  # (B, 197, 384)
        return self.proj(features)                                # (B, 197, 512)


class TextEncoder(nn.Module):
    """Pre-trained DistilBERT with a linear projection to *proj_dim*.

    Returns the [CLS] token embedding as the sentence representation.
    """

    def __init__(self, proj_dim=512, model_name="distilbert-base-uncased"):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.dim  # 768 for distilbert-base
        self.proj = nn.Linear(backbone_dim, proj_dim)

    def forward(self, input_ids, attention_mask):
        """input_ids, attention_mask: (B, seq_len) → (B, seq_len, proj_dim)"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(outputs.last_hidden_state)        # (B, seq_len, 512)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UnpairedMultimodalLearner(nn.Module):
    """Unpaired multimodal classifier for Stanford Cars.

    Follows the reference UML architecture: modality-specific encoders feed
    directly into a shared linear classifier.  A learnable ``img_scale``
    parameter scales the image logits.  The classifier can be warm-started
    with zero-shot text anchors via :meth:`zero_shot_init`.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 196 for Stanford Cars).
    proj_dim : int
        Shared embedding dimensionality produced by both encoders.
    """

    def __init__(self, num_classes=196, proj_dim=512):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # Modality encoders
        self.image_encoder = ImageEncoder(proj_dim=proj_dim)
        self.text_encoder = TextEncoder(proj_dim=proj_dim)

        # Shared classification head
        self.classifier = nn.Linear(proj_dim, num_classes, bias=False)

        # Learnable logit scale for images
        self.img_scale = nn.Parameter(torch.tensor(1.0))

    def zero_shot_init(self, weights_path):
        """Initialize the classifier with pre-computed text anchors.

        Parameters
        ----------
        weights_path : str
            Path to the ``text_anchors.pt`` file produced by ``init_weights.py``.
            Expected to contain an ``"anchors"`` key with shape (num_classes, proj_dim).
        """
        data = torch.load(weights_path, map_location="cpu", weights_only=True)
        anchors = data["anchors"]                           # (num_classes, proj_dim)
        assert anchors.shape == (self.num_classes, self.proj_dim), (
            f"Anchor shape {anchors.shape} does not match "
            f"classifier ({self.num_classes}, {self.proj_dim})"
        )
        self.classifier.weight.data = anchors
        print(f"=> Initialized classifier with zero-shot weights from {weights_path}")

    def forward(self, image=None, input_ids=None, attention_mask=None):
        """Accept an image batch **or** a text batch and return class logits.

        Parameters
        ----------
        image : Tensor | None
            (B, 3, 224, 224) image tensor.
        input_ids : Tensor | None
            (B, seq_len) token ids from DistilBERT tokenizer.
        attention_mask : Tensor | None
            (B, seq_len) attention mask (optional, defaults to all-ones).

        Returns
        -------
        logits : Tensor
            (B, num_classes) raw class scores.
        """
        if image is not None:
            img_features = self.image_encoder(image)            # (B, proj_dim)
            return self.classifier(img_features) * self.img_scale

        if input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            txt_features = self.text_encoder(input_ids, attention_mask)  # (B, proj_dim)
            return self.classifier(txt_features)

        raise ValueError("Provide either `image` or `input_ids`.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnpairedMultimodalLearner().to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total:,}  trainable: {trainable:,}")

    # Image forward pass
    dummy_img = torch.randn(2, 3, 224, 224, device=device)
    logits_img = model(image=dummy_img)
    print(f"Image  logits shape: {logits_img.shape}")  # (2, 196)

    # Text forward pass
    dummy_ids = torch.randint(0, 30522, (2, 32), device=device)
    dummy_mask = torch.ones_like(dummy_ids)
    logits_txt = model(input_ids=dummy_ids, attention_mask=dummy_mask)
    print(f"Text   logits shape: {logits_txt.shape}")  # (2, 196)
