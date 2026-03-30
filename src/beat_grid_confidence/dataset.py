"""Dataset loading for beat_this annotations format.

beat_this annotations use .beats files (tab-separated: time, beat_position)
where beat_position=1 indicates a downbeat. Spectrograms are pre-extracted
as .npy/.npz files at 22050 Hz, 128 mel bins, hop_length=441 (50 FPS).

See: https://github.com/CPJKU/beat_this_annotations
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# beat_this spectrogram parameters (must match pre-extraction)
SAMPLE_RATE = 22050
HOP_LENGTH = 441
N_MELS = 128
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # ~50 FPS


@dataclass
class TrackAnnotation:
    """Parsed annotation for a single track."""

    stem: str  # e.g. "ballroom_Albums-AnaBelen_Veneo-01"
    dataset: str  # e.g. "ballroom"
    beat_times: np.ndarray  # seconds
    downbeat_times: np.ndarray  # seconds (empty if dataset has no downbeats)
    has_downbeats: bool
    fold: int  # 0-7 for cross-validation


def load_beats_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a .beats annotation file.

    Args:
        path: Path to .beats file

    Returns:
        Tuple of (beat_times, downbeat_times). downbeat_times is empty
        if the file has no beat position column.
    """
    data = np.loadtxt(path)

    if data.ndim == 1:
        # 1D format: beat times only, no downbeat info
        return data, np.array([])

    # 2D format: [time, beat_position] where position=1 is downbeat
    beat_times = data[:, 0]
    beat_positions = data[:, 1]
    downbeat_times = beat_times[beat_positions == 1]

    return beat_times, downbeat_times


def load_dataset_annotations(
    annotations_dir: Path,
    dataset_name: str,
) -> list[TrackAnnotation]:
    """Load all annotations for a dataset.

    Args:
        annotations_dir: Root of beat_this_annotations clone
        dataset_name: e.g. "ballroom", "hainsworth"

    Returns:
        List of TrackAnnotation for each track
    """
    dataset_dir = annotations_dir / dataset_name

    # Check if dataset has downbeats
    info_path = dataset_dir / "info.json"
    has_downbeats = False
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
            has_downbeats = info.get("has_downbeats", False)

    # Load fold assignments
    folds: dict[str, int] = {}
    folds_path = dataset_dir / "8-folds.split"
    if folds_path.exists():
        with open(folds_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) == 2:
                        folds[parts[0]] = int(parts[1])

    # Load each .beats file
    beats_dir = dataset_dir / "annotations" / "beats"
    if not beats_dir.exists():
        return []

    annotations = []
    for beats_path in sorted(beats_dir.glob("*.beats")):
        stem = beats_path.stem
        beat_times, downbeat_times = load_beats_file(beats_path)

        if not has_downbeats:
            downbeat_times = np.array([])

        annotations.append(TrackAnnotation(
            stem=stem,
            dataset=dataset_name,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            has_downbeats=has_downbeats,
            fold=folds.get(stem, -1),
        ))

    return annotations


def load_all_annotations(
    annotations_dir: Path,
    datasets: list[str] | None = None,
) -> list[TrackAnnotation]:
    """Load annotations from all (or specified) datasets.

    Args:
        annotations_dir: Root of beat_this_annotations clone
        datasets: List of dataset names, or None for all

    Returns:
        Combined list of TrackAnnotation across all datasets
    """
    if datasets is None:
        # Discover all datasets
        datasets = [
            d.name for d in sorted(annotations_dir.iterdir())
            if d.is_dir() and (d / "annotations" / "beats").exists()
        ]

    all_annotations = []
    for name in datasets:
        anns = load_dataset_annotations(annotations_dir, name)
        all_annotations.extend(anns)
        print(f"  {name}: {len(anns)} tracks (downbeats: {anns[0].has_downbeats if anns else 'N/A'})")

    print(f"Total: {len(all_annotations)} tracks from {len(datasets)} datasets")
    return all_annotations


def make_splits(
    annotations: list[TrackAnnotation],
    val_fold: int = 0,
    test_fold: int = 1,
) -> tuple[list[TrackAnnotation], list[TrackAnnotation], list[TrackAnnotation]]:
    """Split annotations into train/val/test using 8-fold assignments.

    Standard beat_this protocol: fold N = val, fold (N+1)%8 = test, rest = train.

    Args:
        annotations: All annotations
        val_fold: Fold number for validation
        test_fold: Fold number for test

    Returns:
        Tuple of (train, val, test) annotation lists
    """
    train, val, test = [], [], []
    unassigned = []

    for ann in annotations:
        if ann.fold == val_fold:
            val.append(ann)
        elif ann.fold == test_fold:
            test.append(ann)
        elif ann.fold >= 0:
            train.append(ann)
        else:
            unassigned.append(ann)

    # Unassigned tracks (no fold info) go to training
    train.extend(unassigned)

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


class BeatGridConfidenceDataset(Dataset):
    """PyTorch dataset for training confidence heads.

    Loads pre-extracted spectrograms and beat annotations.
    Generates confidence correctness targets from ground-truth beat positions.
    """

    TOLERANCE_SEC = 0.050  # 50ms tolerance for beat correctness

    def __init__(
        self,
        annotations: list[TrackAnnotation],
        spectrogram_dir: Path,
        chunk_frames: int = 1500,
        augment: bool = False,
    ) -> None:
        self.annotations = annotations
        self.spectrogram_dir = spectrogram_dir
        self.chunk_frames = chunk_frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ann = self.annotations[idx]

        # Load pre-extracted spectrogram
        # beat_this stores these in {dataset}/ subdirs as .npy or in .npz bundles
        spec = self._load_spectrogram(ann)
        n_frames = spec.shape[0]

        # Create frame-level beat/downbeat targets
        beat_target = self._times_to_target(ann.beat_times, n_frames)
        downbeat_target = self._times_to_target(ann.downbeat_times, n_frames)

        # Apply augmentations
        if self.augment:
            spec, beat_target, downbeat_target = self._augment(
                spec, beat_target, downbeat_target
            )

        # Chunk to fixed length
        spec, beat_target, downbeat_target = self._chunk(
            spec, beat_target, downbeat_target
        )

        # beat_this expects [T, 128] (time-first) — our spectrograms
        # are already stored as [T, n_mels], so no transpose needed

        return {
            "spectrogram": spec,
            "beat_target": beat_target,
            "downbeat_target": downbeat_target,
            "stem": ann.stem,
        }

    def _load_spectrogram(self, ann: TrackAnnotation) -> torch.Tensor:
        """Load pre-extracted spectrogram for a track.

        Tries .npy file first, then .npz bundle.
        """
        # Try individual .npy file
        npy_path = self.spectrogram_dir / ann.dataset / f"{ann.stem}.npy"
        if npy_path.exists():
            data = np.load(npy_path, mmap_mode="r")
            return torch.from_numpy(np.array(data)).float()

        # Try .npz bundle
        npz_path = self.spectrogram_dir / ann.dataset / f"{ann.dataset}.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as bundle:
                if ann.stem in bundle:
                    return torch.from_numpy(bundle[ann.stem]).float()

        raise FileNotFoundError(
            f"No spectrogram found for {ann.stem}. "
            f"Checked: {npy_path} and {npz_path}. "
            f"Run beat_this spectrogram extraction first."
        )

    def _times_to_target(self, times: np.ndarray, n_frames: int) -> torch.Tensor:
        """Convert beat/downbeat times to frame-level binary target."""
        if len(times) == 0:
            return torch.zeros(n_frames)

        target = torch.zeros(n_frames)
        frames = np.round(times * FRAME_RATE).astype(np.int64)
        valid = frames[(frames >= 0) & (frames < n_frames)]
        target[valid] = 1.0
        return target

    def _chunk(
        self,
        spec: torch.Tensor,
        beat_target: torch.Tensor,
        downbeat_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a random chunk of fixed length."""
        n_frames = spec.shape[0]

        if n_frames <= self.chunk_frames:
            pad = self.chunk_frames - n_frames
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad))
            beat_target = torch.nn.functional.pad(beat_target, (0, pad))
            downbeat_target = torch.nn.functional.pad(downbeat_target, (0, pad))
        else:
            start = torch.randint(0, n_frames - self.chunk_frames, (1,)).item()
            end = start + self.chunk_frames
            spec = spec[start:end]
            beat_target = beat_target[start:end]
            downbeat_target = downbeat_target[start:end]

        return spec, beat_target, downbeat_target

    def _augment(
        self,
        spec: torch.Tensor,
        beat_target: torch.Tensor,
        downbeat_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthetic augmentations for confidence head training.

        Creates artificial low-confidence regions by corrupting the spectrogram.
        These teach the confidence head that corrupted regions are unreliable.
        """
        n_frames = spec.shape[0]

        # Silence injection (simulates breakdown)
        if torch.rand(1).item() < 0.3:
            length = torch.randint(50, min(200, max(51, n_frames // 4)), (1,)).item()
            start = torch.randint(0, max(1, n_frames - length), (1,)).item()
            spec[start : start + length] = 0.0

        # Energy drop (simulates sparse intro)
        if torch.rand(1).item() < 0.3:
            length = torch.randint(50, min(200, max(51, n_frames // 4)), (1,)).item()
            start = torch.randint(0, max(1, n_frames - length), (1,)).item()
            attenuation = 10 ** (-torch.rand(1).item() * 2 - 1)  # -20 to -40 dB
            spec[start : start + length] *= attenuation

        return spec, beat_target, downbeat_target
