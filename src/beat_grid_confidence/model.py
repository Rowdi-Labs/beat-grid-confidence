"""Backbone wrapper: loads beat_this and exposes hidden states for head training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .heads import ConfidenceHead, TempoDistributionHead


class BeatGridConfidenceModel(nn.Module):
    """Wraps a frozen beat_this backbone with trainable confidence/tempo heads.

    Architecture:
        mel [B, 128, T] → frontend → transformer_blocks → hidden [B, T, 512]
                                                              │
                                          ┌───────────────────┼───────────────────┐
                                          │                   │                   │
                                    task_heads          ConfidenceHead      TempoDistHead
                                    (frozen)            (trainable)         (trainable)
                                    ↓                   ↓                   ↓
                              beat/downbeat         confidence [B,T]    tempo_dist [B,T,B']
                              logits [B,T]

    The key insight: beat_this has a clean `frontend → transformer_blocks → task_heads`
    structure. We split the forward pass after transformer_blocks to extract hidden
    states, then feed those to both the original task_heads and our new heads.
    No hooks needed.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 512,
        n_tempo_bins: int = 141,
        freeze_backbone: bool = True,
        enable_confidence: bool = True,
        enable_tempo: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim

        self.confidence_head = ConfidenceHead(hidden_dim) if enable_confidence else None
        self.tempo_head = TempoDistributionHead(hidden_dim, n_tempo_bins) if enable_tempo else None

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, spectrogram: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through backbone + heads.

        Args:
            spectrogram: Mel spectrogram [B, 128, T] (beat_this input format)

        Returns:
            Dictionary with keys:
                - beat_logits: [B, T] from backbone task heads
                - downbeat_logits: [B, T] from backbone task heads
                - confidence: [B, T] per-frame confidence (if enabled)
                - tempo_distribution: [B, T, n_bins] per-frame tempo posterior (if enabled)
                - hidden_states: [B, T, D] raw hidden states for analysis
        """
        # Split the backbone forward pass to extract hidden states
        # beat_this structure: frontend → transformer_blocks → task_heads
        x = self.backbone.frontend(spectrogram)  # [B, T, 512]
        hidden = self.backbone.transformer_blocks(x)  # [B, T, 512]

        # Original beat/downbeat predictions from frozen task heads
        logits = self.backbone.task_heads(hidden)  # {"beat": [B, T], "downbeat": [B, T]}

        result: dict[str, torch.Tensor] = {
            "beat_logits": logits["beat"],
            "downbeat_logits": logits["downbeat"],
            "hidden_states": hidden,
        }

        # Run trainable heads on hidden states
        if self.confidence_head is not None:
            result["confidence"] = self.confidence_head(hidden)

        if self.tempo_head is not None:
            result["tempo_distribution"] = self.tempo_head(hidden)

        return result


def load_backbone(
    checkpoint_path: str = "final0",
    device: str = "cpu",
) -> tuple[nn.Module, int]:
    """Load a beat_this checkpoint and return the model + hidden dim.

    Args:
        checkpoint_path: Path to checkpoint file, or one of the named
            checkpoints: "final0", "small0" (downloaded automatically
            by beat_this).
        device: Target device ("cpu", "cuda")

    Returns:
        Tuple of (backbone_module, hidden_dimension)
    """
    from beat_this.inference import load_model

    model = load_model(checkpoint_path=checkpoint_path, device=device)

    # Extract hidden dim from model config
    # beat_this stores hyperparameters in the checkpoint
    hidden_dim = model.transformer_blocks.norm.weight.shape[0]

    return model, hidden_dim


def create_model(
    checkpoint_path: str = "final0",
    device: str = "cpu",
    freeze_backbone: bool = True,
    enable_confidence: bool = True,
    enable_tempo: bool = True,
    n_tempo_bins: int = 141,
) -> BeatGridConfidenceModel:
    """Convenience function to create a full model from a beat_this checkpoint.

    Args:
        checkpoint_path: beat_this checkpoint path or name
        device: Target device
        freeze_backbone: Whether to freeze backbone weights
        enable_confidence: Whether to add confidence head
        enable_tempo: Whether to add tempo distribution head
        n_tempo_bins: Number of BPM bins for tempo head

    Returns:
        BeatGridConfidenceModel ready for training or inference
    """
    backbone, hidden_dim = load_backbone(checkpoint_path, device)

    model = BeatGridConfidenceModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        n_tempo_bins=n_tempo_bins,
        freeze_backbone=freeze_backbone,
        enable_confidence=enable_confidence,
        enable_tempo=enable_tempo,
    )

    return model.to(device)
