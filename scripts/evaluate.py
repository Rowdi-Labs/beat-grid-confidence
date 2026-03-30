"""Evaluation script: compute standard + product metrics on a test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate beat-grid-confidence model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/test"),
        help="Directory with test spectrograms and annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation.json"),
        help="Output JSON with metric results",
    )
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_dir}")

    # TODO: Implement evaluation
    # 1. Load model from checkpoint
    # 2. Run inference on test set
    # 3. Decode with confidence-aware decoder
    # 4. Compute standard metrics (mir_eval)
    # 5. Compute product metrics (evaluation.py)
    # 6. Save results to JSON
    raise NotImplementedError(
        "Evaluation not yet implemented. "
        "Phase 0 milestone: build evaluation harness with standard metrics first."
    )


if __name__ == "__main__":
    main()
