"""Training entry point for beat-grid-confidence heads."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train beat-grid-confidence heads")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/confidence_only.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to beat_this backbone checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for checkpoints and logs",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        config["model"]["checkpoint"] = str(args.checkpoint)

    print(f"Config: {args.config}")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Heads: {config['model']['heads']}")
    print(f"Freeze backbone: {config['model']['freeze_backbone']}")
    print(f"Output: {args.output_dir}")

    # TODO: Implement training loop
    # 1. Load backbone via beat_this package
    # 2. Attach heads (model.py)
    # 3. Load dataset (dataset.py)
    # 4. Train with PyTorch Lightning
    # 5. Log to wandb
    # 6. Save best checkpoint
    raise NotImplementedError(
        "Training loop not yet implemented. "
        "Phase 0 milestone: reproduce beat_this baseline first."
    )


if __name__ == "__main__":
    main()
