"""Basic tests for model components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from beat_grid_confidence.heads import ConfidenceHead, TempoDistributionHead
from beat_grid_confidence.losses import CombinedLoss, ConfidenceLoss, TempoDistributionLoss
from beat_grid_confidence.evaluation import (
    compute_confidence_brier,
    compute_continuity_span,
    compute_correction_effort,
)
from beat_grid_confidence.decode import decode_confidence_aware


class TestConfidenceHead:
    def test_output_shape(self) -> None:
        head = ConfidenceHead(hidden_dim=256)
        x = torch.randn(2, 100, 256)
        out = head(x)
        assert out.shape == (2, 100)

    def test_output_range(self) -> None:
        head = ConfidenceHead(hidden_dim=256)
        x = torch.randn(2, 100, 256)
        out = head(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestTempoDistributionHead:
    def test_output_shape(self) -> None:
        head = TempoDistributionHead(hidden_dim=256, n_bins=141)
        x = torch.randn(2, 100, 256)
        out = head(x)
        assert out.shape == (2, 100, 141)

    def test_output_sums_to_one(self) -> None:
        head = TempoDistributionHead(hidden_dim=256, n_bins=141)
        x = torch.randn(2, 100, 256)
        out = head(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_expected_tempo(self) -> None:
        head = TempoDistributionHead(hidden_dim=256, n_bins=141, bpm_min=60.0)
        x = torch.randn(1, 10, 256)
        dist = head(x)
        expected = head.expected_tempo(dist)
        assert expected.shape == (1, 10)
        assert expected.min() >= 60.0
        assert expected.max() <= 200.0


class TestConfidenceLoss:
    def test_perfect_prediction(self) -> None:
        loss_fn = ConfidenceLoss()
        confidence = torch.ones(2, 100)
        mask = torch.ones(2, 100)
        loss = loss_fn(confidence, mask)
        assert loss.item() < 0.01

    def test_worst_prediction(self) -> None:
        loss_fn = ConfidenceLoss()
        confidence = torch.zeros(2, 100)
        mask = torch.ones(2, 100)
        loss = loss_fn(confidence, mask)
        # BCE of (0, 1) should be very high
        assert loss.item() > 10.0


class TestProductMetrics:
    def test_perfect_continuity(self) -> None:
        beats = np.arange(0, 10, 0.5)  # regular beats
        span = compute_continuity_span(beats, beats)
        assert span == pytest.approx(9.5, abs=0.01)

    def test_zero_correction_effort(self) -> None:
        beats = np.arange(0, 10, 0.5)
        effort = compute_correction_effort(beats, beats)
        assert effort == 0

    def test_correction_effort_with_errors(self) -> None:
        predicted = np.arange(0, 10, 0.5)
        reference = np.arange(0, 10, 0.5)
        # Shift some beats to create errors
        predicted[5:10] += 0.2  # 5 consecutive wrong beats
        effort = compute_correction_effort(predicted, reference)
        assert effort >= 1

    def test_brier_perfect(self) -> None:
        confidence = np.ones(100)
        correctness = np.ones(100)
        brier = compute_confidence_brier(confidence, correctness)
        assert brier == pytest.approx(0.0, abs=1e-10)

    def test_brier_worst(self) -> None:
        confidence = np.zeros(100)
        correctness = np.ones(100)
        brier = compute_confidence_brier(confidence, correctness)
        assert brier == pytest.approx(1.0, abs=1e-10)


class TestDecoder:
    def test_decode_returns_grid(self) -> None:
        n_frames = 1000
        # Simulate periodic beat activations at 120 BPM, 50 fps
        frame_rate = 50.0
        period = frame_rate * 60.0 / 120.0  # 25 frames per beat

        beat_logits = np.zeros(n_frames)
        for i in range(0, n_frames, int(period)):
            beat_logits[i] = 3.0  # strong activation

        downbeat_logits = np.zeros(n_frames)
        for i in range(0, n_frames, int(period * 4)):
            downbeat_logits[i] = 3.0

        confidence = np.ones(n_frames) * 0.9

        grid = decode_confidence_aware(
            beat_logits, downbeat_logits, confidence, frame_rate=frame_rate
        )

        assert grid.primary.bpm > 0
        assert len(grid.primary.beats) > 0
        assert len(grid.confidence_curve) == n_frames

    def test_low_confidence_regions_detected(self) -> None:
        n_frames = 1000
        confidence = np.ones(n_frames) * 0.9
        # Create a low-confidence gap
        confidence[200:400] = 0.1

        beat_logits = np.zeros(n_frames)
        downbeat_logits = np.zeros(n_frames)
        # Add some beats
        for i in range(0, n_frames, 25):
            beat_logits[i] = 3.0

        grid = decode_confidence_aware(
            beat_logits, downbeat_logits, confidence, frame_rate=50.0
        )

        assert len(grid.low_confidence_regions) >= 1
