"""Tests for annotation loading and dataset pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from beat_grid_confidence.dataset import (
    FRAME_RATE,
    TrackAnnotation,
    load_beats_file,
    load_dataset_annotations,
    make_splits,
)


@pytest.fixture
def mock_annotations_dir(tmp_path: Path) -> Path:
    """Create a minimal beat_this_annotations directory structure."""
    dataset_dir = tmp_path / "test_dataset"
    beats_dir = dataset_dir / "annotations" / "beats"
    beats_dir.mkdir(parents=True)

    # info.json
    (dataset_dir / "info.json").write_text('{"has_downbeats": true}')

    # 8-folds.split
    splits = "test_dataset_track_01\t0\ntest_dataset_track_02\t1\ntest_dataset_track_03\t2\n"
    (dataset_dir / "8-folds.split").write_text(splits)

    # .beats files (2D format: time, beat_position)
    for i in range(1, 4):
        beats = []
        for beat_idx in range(40):
            time = beat_idx * 0.5
            position = (beat_idx % 4) + 1
            beats.append(f"{time:.3f}\t{position}")
        (beats_dir / f"test_dataset_track_{i:02d}.beats").write_text("\n".join(beats))

    return tmp_path


class TestLoadBeatsFile:
    def test_2d_format(self, tmp_path: Path) -> None:
        content = "0.0\t1\n0.5\t2\n1.0\t3\n1.5\t4\n2.0\t1\n"
        path = tmp_path / "test.beats"
        path.write_text(content)

        beats, downbeats = load_beats_file(path)
        assert len(beats) == 5
        assert beats[0] == pytest.approx(0.0)
        assert beats[-1] == pytest.approx(2.0)
        # Downbeats are where position == 1
        assert len(downbeats) == 2
        assert downbeats[0] == pytest.approx(0.0)
        assert downbeats[1] == pytest.approx(2.0)

    def test_1d_format(self, tmp_path: Path) -> None:
        content = "0.0\n0.5\n1.0\n1.5\n2.0\n"
        path = tmp_path / "test.beats"
        path.write_text(content)

        beats, downbeats = load_beats_file(path)
        assert len(beats) == 5
        assert len(downbeats) == 0  # no downbeat info in 1D format


class TestLoadDatasetAnnotations:
    def test_loads_all_tracks(self, mock_annotations_dir: Path) -> None:
        anns = load_dataset_annotations(mock_annotations_dir, "test_dataset")
        assert len(anns) == 3

    def test_parses_beats(self, mock_annotations_dir: Path) -> None:
        anns = load_dataset_annotations(mock_annotations_dir, "test_dataset")
        ann = anns[0]
        assert len(ann.beat_times) == 40
        assert ann.beat_times[0] == pytest.approx(0.0)
        assert ann.beat_times[1] == pytest.approx(0.5)

    def test_parses_downbeats(self, mock_annotations_dir: Path) -> None:
        anns = load_dataset_annotations(mock_annotations_dir, "test_dataset")
        ann = anns[0]
        assert ann.has_downbeats is True
        assert len(ann.downbeat_times) == 10  # every 4th beat

    def test_parses_folds(self, mock_annotations_dir: Path) -> None:
        anns = load_dataset_annotations(mock_annotations_dir, "test_dataset")
        folds = {a.stem: a.fold for a in anns}
        assert folds["test_dataset_track_01"] == 0
        assert folds["test_dataset_track_02"] == 1
        assert folds["test_dataset_track_03"] == 2


class TestMakeSplits:
    def test_split_by_fold(self, mock_annotations_dir: Path) -> None:
        anns = load_dataset_annotations(mock_annotations_dir, "test_dataset")
        train, val, test = make_splits(anns, val_fold=0, test_fold=1)
        assert len(val) == 1
        assert len(test) == 1
        assert len(train) == 1
        assert val[0].fold == 0
        assert test[0].fold == 1
