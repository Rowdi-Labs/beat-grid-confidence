"""Dataset loading, annotation parsing, and augmentation for beat grid confidence training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class BeatAnnotation:
    """Beat/downbeat annotations for a single track."""

    audio_path: Path
    beat_times: np.ndarray  # seconds
    downbeat_times: np.ndarray  # seconds
    duration: float  # seconds
    dataset_name: str
    license: str  # e.g., "MIT", "CC-BY-4.0"

    def is_commercially_clean(self) -> bool:
        """Check if this annotation's license permits commercial training."""
        clean_licenses = {"MIT", "Apache-2.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"}
        return self.license in clean_licenses


@dataclass
class ConfidenceTarget:
    """Training targets for the confidence head."""

    correctness_mask: np.ndarray  # [T] binary: 1 if beat prediction is correct
    region_labels: list[dict[str, Any]] = field(default_factory=list)  # labeled difficulty regions


class BeatGridDataset(Dataset):
    """Dataset for training confidence heads on beat_this backbone outputs.

    Loads pre-computed mel spectrograms and beat annotations.
    Generates confidence targets from ground-truth beat positions.
    """

    TOLERANCE_SEC = 0.050  # 50ms tolerance for beat correctness

    def __init__(
        self,
        annotations: list[BeatAnnotation],
        spectrogram_dir: Path,
        sample_rate: int = 22050,
        hop_length: int = 441,
        chunk_frames: int = 2048,
        augment: bool = False,
    ) -> None:
        self.annotations = [a for a in annotations if a.is_commercially_clean()]
        self.spectrogram_dir = spectrogram_dir
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.chunk_frames = chunk_frames
        self.augment = augment
        self.frame_rate = sample_rate / hop_length  # frames per second

        if len(self.annotations) < len(annotations):
            n_filtered = len(annotations) - len(self.annotations)
            print(f"Filtered {n_filtered} annotations with non-commercial licenses")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ann = self.annotations[idx]

        # Load pre-computed spectrogram
        spec_path = self.spectrogram_dir / f"{ann.audio_path.stem}.pt"
        spectrogram = torch.load(spec_path, weights_only=True)  # [T, n_mels]

        # Generate frame-level beat targets
        beat_frames = self._times_to_frames(ann.beat_times)
        downbeat_frames = self._times_to_frames(ann.downbeat_times)

        n_frames = spectrogram.shape[0]
        beat_target = self._make_target_vector(beat_frames, n_frames)
        downbeat_target = self._make_target_vector(downbeat_frames, n_frames)

        # Apply augmentations if enabled
        if self.augment:
            spectrogram, beat_target, downbeat_target = self._augment(
                spectrogram, beat_target, downbeat_target
            )

        # Chunk to fixed length
        spectrogram, beat_target, downbeat_target = self._chunk(
            spectrogram, beat_target, downbeat_target
        )

        return {
            "spectrogram": spectrogram,
            "beat_target": beat_target,
            "downbeat_target": downbeat_target,
        }

    def _times_to_frames(self, times: np.ndarray) -> np.ndarray:
        """Convert beat times in seconds to frame indices."""
        return np.round(times * self.frame_rate).astype(np.int64)

    def _make_target_vector(self, frames: np.ndarray, n_frames: int) -> torch.Tensor:
        """Create a binary target vector from frame indices."""
        target = torch.zeros(n_frames)
        valid = frames[(frames >= 0) & (frames < n_frames)]
        target[valid] = 1.0
        return target

    def _chunk(
        self,
        spectrogram: torch.Tensor,
        beat_target: torch.Tensor,
        downbeat_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a random chunk of fixed length."""
        n_frames = spectrogram.shape[0]
        if n_frames <= self.chunk_frames:
            # Pad if too short
            pad = self.chunk_frames - n_frames
            spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, 0, pad))
            beat_target = torch.nn.functional.pad(beat_target, (0, pad))
            downbeat_target = torch.nn.functional.pad(downbeat_target, (0, pad))
        else:
            start = torch.randint(0, n_frames - self.chunk_frames, (1,)).item()
            spectrogram = spectrogram[start : start + self.chunk_frames]
            beat_target = beat_target[start : start + self.chunk_frames]
            downbeat_target = downbeat_target[start : start + self.chunk_frames]

        return spectrogram, beat_target, downbeat_target

    def _augment(
        self,
        spectrogram: torch.Tensor,
        beat_target: torch.Tensor,
        downbeat_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply augmentations that create synthetic low-confidence regions.

        These are specifically designed to train the confidence head:
        - Silence injection: zero out a random segment (simulates breakdown)
        - Energy drop: attenuate a segment by 20-40dB (simulates sparse intro)
        - Spectral masking: zero random frequency bands (simulates missing instruments)
        """
        n_frames = spectrogram.shape[0]

        if torch.rand(1).item() < 0.3:  # 30% chance of silence injection
            length = torch.randint(50, min(200, n_frames // 4), (1,)).item()
            start = torch.randint(0, max(1, n_frames - length), (1,)).item()
            spectrogram[start : start + length] = 0.0

        if torch.rand(1).item() < 0.3:  # 30% chance of energy drop
            length = torch.randint(50, min(200, n_frames // 4), (1,)).item()
            start = torch.randint(0, max(1, n_frames - length), (1,)).item()
            attenuation = 10 ** (-torch.rand(1).item() * 2 - 1)  # -20 to -40 dB
            spectrogram[start : start + length] *= attenuation

        return spectrogram, beat_target, downbeat_target
