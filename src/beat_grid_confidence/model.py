"""Backbone wrapper: loads beat_this and exposes hidden states for head training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .heads import ConfidenceHead, TempoDistributionHead


class BeatGridConfidenceModel(nn.Module):
    """Wraps a frozen beat_this backbone with trainable confidence/tempo heads.

    The backbone's final hidden states are extracted via a forward hook and fed
    to lightweight linear heads. Only the heads are trained; the backbone stays frozen
    (or optionally LoRA-adapted).
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        n_tempo_bins: int = 141,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.confidence_head = ConfidenceHead(hidden_dim)
        self.tempo_head = TempoDistributionHead(hidden_dim, n_tempo_bins)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Hook storage for hidden states
        self._hidden_states: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register a forward hook on the backbone's final transformer layer
        to capture hidden states before the output projection."""
        # beat_this architecture: the last transformer block outputs hidden states
        # that are then projected to beat/downbeat logits. We tap in before projection.
        #
        # TODO: Verify the exact layer name from beat_this source.
        # This is a placeholder — the actual hook target depends on the beat_this
        # model architecture. Inspect with:
        #   for name, module in backbone.named_modules(): print(name)
        pass

    def forward(
        self, spectrogram: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through backbone + heads.

        Args:
            spectrogram: Mel spectrogram [B, T, n_mels]

        Returns:
            Dictionary with keys:
                - beat_logits: [B, T] from backbone
                - downbeat_logits: [B, T] from backbone
                - confidence: [B, T] per-frame confidence
                - tempo_distribution: [B, T, n_bins] per-frame tempo posterior
        """
        # Run backbone — this also triggers the hidden state hook
        backbone_out = self.backbone(spectrogram)

        # Extract hidden states captured by hook
        # TODO: Replace with actual hook-based extraction once beat_this
        # architecture is inspected. For now, use backbone output as placeholder.
        hidden = self._hidden_states if self._hidden_states is not None else backbone_out

        # Run heads on hidden states
        confidence = self.confidence_head(hidden)
        tempo_dist = self.tempo_head(hidden)

        return {
            "beat_logits": backbone_out,  # TODO: split beat/downbeat from backbone output
            "downbeat_logits": backbone_out,
            "confidence": confidence,
            "tempo_distribution": tempo_dist,
        }


def load_backbone(checkpoint_path: str, device: str = "cpu") -> tuple[nn.Module, int]:
    """Load a beat_this checkpoint and return the model + hidden dim.

    Args:
        checkpoint_path: Path to beat_this .ckpt or .pt file
        device: Target device

    Returns:
        Tuple of (backbone_module, hidden_dimension)
    """
    # TODO: Implement using beat_this package's loading utilities.
    # from beat_this.model import BeatThis
    # model = BeatThis.load_from_checkpoint(checkpoint_path)
    # hidden_dim = model.config.hidden_dim  # or however beat_this exposes this
    raise NotImplementedError(
        "Implement using beat_this package. "
        "See https://github.com/CPJKU/beat_this for model loading API."
    )
