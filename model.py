#!/usr/bin/env python3
"""
model.py — Unpaired Multimodal Learner for Stanford Cars classification.

Architecture:
    1. Image encoder  — ViT-Small/16 (timm) → Linear → 512-d
    2. Text encoder   — DistilBERT (HuggingFace) → Linear → 512-d
    3. Shared backbone — 4-layer TransformerEncoder
    4. Classification  — Linear → 196 classes

The forward pass accepts *either* an image batch or a text batch (not both),
routes through the matching encoder, then through the shared backbone and
classification head.
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
        """pixel_values: (B, 3, 224, 224) → (B, proj_dim)"""
        features = self.backbone(pixel_values)     # (B, 384)
        return self.proj(features)                  # (B, 512)


class TextEncoder(nn.Module):
    """Pre-trained DistilBERT with a linear projection to *proj_dim*.

    Uses the [CLS] token output as the sentence representation.
    """

    def __init__(self, proj_dim=512, model_name="distilbert-base-uncased"):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.dim  # 768 for distilbert-base
        self.proj = nn.Linear(backbone_dim, proj_dim)

    def forward(self, input_ids, attention_mask):
        """input_ids, attention_mask: (B, seq_len) → (B, proj_dim)"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]   # (B, 768)
        return self.proj(cls_embedding)                    # (B, 512)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UnpairedMultimodalLearner(nn.Module):
    """Unpaired multimodal classifier for Stanford Cars.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 196 for Stanford Cars).
    proj_dim : int
        Shared embedding dimensionality produced by both encoders.
    num_backbone_layers : int
        Number of layers in the shared TransformerEncoder backbone.
    nhead : int
        Number of attention heads in each backbone layer.
    dim_feedforward : int
        Feed-forward width inside each backbone layer.
    dropout : float
        Dropout applied inside the backbone.
    """

    def __init__(
        self,
        num_classes=196,
        proj_dim=512,
        num_backbone_layers=4,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # Modality encoders
        self.image_encoder = ImageEncoder(proj_dim=proj_dim)
        self.text_encoder = TextEncoder(proj_dim=proj_dim)

        # Shared backbone (operates on a length-1 sequence per sample)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.shared_backbone = nn.TransformerEncoder(
            encoder_layer, num_layers=num_backbone_layers,
        )

        # Classification head
        self.classifier = nn.Linear(proj_dim, num_classes)

    def _encode(self, image=None, input_ids=None, attention_mask=None):
        """Route through the correct modality encoder. Returns (B, proj_dim)."""
        if image is not None:
            return self.image_encoder(image)
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return self.text_encoder(input_ids, attention_mask)
        raise ValueError("Provide either `image` or `input_ids`.")

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
        emb = self._encode(image, input_ids, attention_mask)  # (B, proj_dim)

        # Shared backbone expects (B, S, D); treat each sample as a length-1 sequence
        emb = emb.unsqueeze(1)                                # (B, 1, proj_dim)
        emb = self.shared_backbone(emb)                       # (B, 1, proj_dim)
        emb = emb.squeeze(1)                                  # (B, proj_dim)

        return self.classifier(emb)                           # (B, num_classes)


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
