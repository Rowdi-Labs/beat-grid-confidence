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
    try:
        data = np.loadtxt(path)
    except ValueError:
        return np.array([]), np.array([])

    if data.size == 0:
        return np.array([]), np.array([])

    if data.ndim == 1 and data.shape[0] <= 2:
        # Single beat entry — treat as 1D
        return np.atleast_1d(data[0] if data.ndim == 1 and data.shape[0] == 2 else data), np.array([])

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
        """Extract a random chunk of fixed length. If chunk_frames <= 0, return full track."""
        if self.chunk_frames <= 0:
            return spec, beat_target, downbeat_target

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


class HiddenStatesDataset(Dataset):
    """Dataset that loads pre-extracted hidden states + logits.

    Uses backbone logits to compute the CORRECT confidence target:
    "regional backbone accuracy" — how well does the backbone predict beats
    in a local window around each frame?

    This is fundamentally different from "is there a beat near this frame"
    (which just learns the base rate). The correct signal is:
    - Find frames where backbone predicts beats (logit > 0)
    - Check which predicted beats match ground truth (within 50ms)
    - Smooth into a regional accuracy score

    Memory budget per batch (batch=32, T=1500, D=512):
        hidden states: 32 * 1500 * 512 * 4B = 94 MB
        logits:        32 * 1500 * 2 * 4B   = 0.37 MB
        Total: ~95 MB
    """

    def __init__(
        self,
        annotations: list[TrackAnnotation],
        hidden_states_dir: Path,
        chunk_frames: int = 1500,
    ) -> None:
        self.annotations = annotations
        self.hidden_states_dir = hidden_states_dir
        self.chunk_frames = chunk_frames

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ann = self.annotations[idx]
        ds_dir = self.hidden_states_dir / ann.dataset

        # Load pre-extracted hidden states [T, 512]
        hidden_path = ds_dir / f"{ann.stem}.hidden.npy"
        if not hidden_path.exists():
            # Fall back to old format
            old_path = ds_dir / f"{ann.stem}.npy"
            if old_path.exists():
                hidden_path = old_path
            else:
                raise FileNotFoundError(
                    f"No hidden states for {ann.stem}. "
                    f"Run: python scripts/extract_hidden_states.py --force --datasets {ann.dataset}"
                )
        hidden = torch.from_numpy(np.load(hidden_path).astype(np.float32))
        n_frames = hidden.shape[0]

        # Load pre-extracted logits [T] each
        logits_path = ds_dir / f"{ann.stem}.logits.npz"
        if logits_path.exists():
            with np.load(logits_path) as lf:
                beat_logits = torch.from_numpy(lf["beat_logits"][:n_frames])
        else:
            beat_logits = None

        # Create frame-level beat target from ground truth
        beat_target = self._times_to_target(ann.beat_times, n_frames)

        # Compute the CORRECT confidence target: regional backbone accuracy
        if beat_logits is not None:
            correctness = self._compute_regional_accuracy(beat_logits, beat_target, n_frames)
        else:
            # Fallback if no logits available
            correctness = torch.ones(n_frames) * 0.5

        # Chunk to fixed length
        if self.chunk_frames > 0 and n_frames > self.chunk_frames:
            start = torch.randint(0, n_frames - self.chunk_frames, (1,)).item()
            end = start + self.chunk_frames
            hidden = hidden[start:end]
            correctness = correctness[start:end]
        elif self.chunk_frames > 0 and n_frames < self.chunk_frames:
            pad = self.chunk_frames - n_frames
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
            correctness = torch.nn.functional.pad(correctness, (0, pad), value=0.5)

        return {
            "hidden_states": hidden,
            "correctness": correctness,
        }

    def _times_to_target(self, times: np.ndarray, n_frames: int) -> torch.Tensor:
        if len(times) == 0:
            return torch.zeros(n_frames)
        target = torch.zeros(n_frames)
        frames = np.round(times * FRAME_RATE).astype(np.int64)
        valid = frames[(frames >= 0) & (frames < n_frames)]
        target[valid] = 1.0
        return target

    @staticmethod
    def _compute_regional_accuracy(
        beat_logits: torch.Tensor,
        beat_target: torch.Tensor,
        n_frames: int,
        tolerance_frames: int = 3,
        window_frames: int = 50,  # ~1 second window at 50fps
    ) -> torch.Tensor:
        """Compute regional backbone accuracy as the confidence target.

        For each frame, looks at a local window and computes:
        - Which frames does the backbone predict beats? (logit > 0, local peaks)
        - Of those predicted beats, what fraction match ground truth?

        Returns a smooth [0, 1] target per frame. High = backbone is right here.

        This is the RIGHT target because it answers "is the backbone reliable
        in this region?" rather than "is there a beat near this frame?"
        """
        # Step 1: Find backbone's predicted beat positions (peaks in logits)
        sigmoid = torch.sigmoid(beat_logits)
        pred_peaks = torch.zeros(n_frames, dtype=torch.bool)
        for i in range(1, n_frames - 1):
            if sigmoid[i] > 0.3 and sigmoid[i] > sigmoid[i - 1] and sigmoid[i] > sigmoid[i + 1]:
                pred_peaks[i] = True
        pred_frames = torch.where(pred_peaks)[0].float()

        # Step 2: Find ground truth beat positions
        gt_frames = torch.where(beat_target > 0.5)[0].float()

        if len(pred_frames) == 0 or len(gt_frames) == 0:
            return torch.ones(n_frames) * 0.5  # uncertain

        # Step 3: For each predicted beat, is it correct? (within tolerance of GT)
        # [P] vs [G] -> [P, G] distances
        pred_to_gt = torch.abs(pred_frames.unsqueeze(1) - gt_frames.unsqueeze(0))
        pred_correct = pred_to_gt.min(dim=1).values <= tolerance_frames  # [P] bool

        # Step 4: Compute regional accuracy with a sliding window
        # Place correct/incorrect scores at predicted beat positions, then smooth
        accuracy_signal = torch.full((n_frames,), float('nan'))
        for i, pf in enumerate(pred_frames.long()):
            accuracy_signal[pf] = 1.0 if pred_correct[i] else 0.0

        # Smooth: for each frame, average the accuracy of predicted beats in window
        regional = torch.zeros(n_frames)
        half_w = window_frames // 2
        for f in range(n_frames):
            lo = max(0, f - half_w)
            hi = min(n_frames, f + half_w + 1)
            window = accuracy_signal[lo:hi]
            valid = window[~torch.isnan(window)]
            if len(valid) > 0:
                regional[f] = valid.mean()
            else:
                regional[f] = 0.5  # no predictions in window → uncertain

        return regional
