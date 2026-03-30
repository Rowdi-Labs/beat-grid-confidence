"""Lightweight prediction heads trained on frozen backbone hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConfidenceHead(nn.Module):
    """Per-frame confidence prediction.

    Single linear projection from hidden dim to scalar, sigmoid output.
    Predicts whether the backbone's beat prediction at each frame is reliable.

    Trained against a binary correctness mask: 1 if the nearest ground-truth
    beat is within 50ms of the predicted beat, 0 otherwise.
    """

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] from backbone

        Returns:
            [B, T] confidence scores in [0, 1]
        """
        return torch.sigmoid(self.projection(hidden_states)).squeeze(-1)


class TempoDistributionHead(nn.Module):
    """Per-frame tempo posterior distribution.

    Linear projection from hidden dim to tempo bins, softmax output.
    Captures half-time / double-time ambiguity by predicting a distribution
    over plausible tempi rather than a single BPM estimate.

    Default bins: 141 bins spanning 60-200 BPM in 1-BPM steps.
    """

    def __init__(self, hidden_dim: int = 512, n_bins: int = 141, bpm_min: float = 60.0) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, n_bins)
        self.n_bins = n_bins
        self.bpm_min = bpm_min

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] from backbone

        Returns:
            [B, T, n_bins] tempo probability distribution per frame
        """
        return torch.softmax(self.projection(hidden_states), dim=-1)

    def bins_to_bpm(self) -> torch.Tensor:
        """Return a tensor mapping bin indices to BPM values."""
        return torch.arange(self.n_bins, dtype=torch.float32) + self.bpm_min

    def expected_tempo(self, distribution: torch.Tensor) -> torch.Tensor:
        """Compute expected BPM from a tempo distribution.

        Args:
            distribution: [B, T, n_bins] from forward()

        Returns:
            [B, T] expected BPM values
        """
        bpm_values = self.bins_to_bpm().to(distribution.device)
        return (distribution * bpm_values).sum(dim=-1)
