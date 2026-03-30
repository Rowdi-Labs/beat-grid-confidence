"""Training entry point for beat-grid-confidence heads.

Two modes:
1. Pre-extracted hidden states (recommended, memory-efficient):
    python scripts/extract_hidden_states.py --spectrogram-dir data/spectrograms --output-dir data/hidden_states --datasets ballroom
    python scripts/train.py --hidden-states-dir data/hidden_states --annotations-dir data/beat_this_annotations --datasets ballroom

2. End-to-end with backbone (requires GPU with 16GB+ VRAM):
    python scripts/train.py --spectrogram-dir data/spectrograms --annotations-dir data/beat_this_annotations --datasets ballroom

Memory budget (pre-extracted, batch=32, T=1500):
    hidden states: 32 * 1500 * 512 * 4B = 94 MB
    targets:       32 * 1500 * 4B       = 0.2 MB
    head params:   513 * 4B             = 0.002 MB
    Total: ~95 MB (vs ~16 GB with backbone)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from beat_grid_confidence.dataset import (
    HiddenStatesDataset,
    load_all_annotations,
    make_splits,
)
from beat_grid_confidence.heads import ConfidenceHead


class ConfidenceHeadTask(pl.LightningModule):
    """Lightweight training task: confidence head only, no backbone.

    Loads pre-extracted hidden states and trains a single Linear(512, 1) head.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        hidden_dim = config["model"]["hidden_dim"]
        self.confidence_head = ConfidenceHead(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.confidence_head(hidden_states)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        confidence = self.confidence_head(batch["hidden_states"])
        correctness = batch["correctness"]

        loss = torch.nn.functional.binary_cross_entropy(confidence, correctness)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        confidence = self.confidence_head(batch["hidden_states"])
        correctness = batch["correctness"]

        loss = torch.nn.functional.binary_cross_entropy(confidence, correctness)
        brier = torch.mean((confidence - correctness) ** 2)

        # Per-class accuracy
        pred_correct = confidence > 0.5
        actual_correct = correctness > 0.5
        accuracy = (pred_correct == actual_correct).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/confidence_brier", brier, prog_bar=True)
        self.log("val/accuracy", accuracy, prog_bar=True)

    def configure_optimizers(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["training"]["max_epochs"],
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train beat-grid-confidence heads")
    parser.add_argument("--config", type=Path, default=Path("configs/confidence_only.yaml"))
    parser.add_argument("--hidden-states-dir", type=Path, required=True,
                        help="Pre-extracted hidden states from extract_hidden_states.py")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--test-fold", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load annotations
    print("Loading annotations...")
    all_annotations = load_all_annotations(args.annotations_dir, datasets=args.datasets)
    train_anns, val_anns, _ = make_splits(
        all_annotations, val_fold=args.val_fold, test_fold=args.test_fold
    )

    # Create datasets (hidden states, not spectrograms)
    chunk_frames = config["data"]["chunk_frames"]
    train_dataset = HiddenStatesDataset(train_anns, args.hidden_states_dir, chunk_frames)
    val_dataset = HiddenStatesDataset(val_anns, args.hidden_states_dir, chunk_frames)

    # Estimate memory
    batch_size = config["training"]["batch_size"]
    hidden_dim = config["model"]["hidden_dim"]
    mem_mb = batch_size * chunk_frames * hidden_dim * 4 / 1e6
    print(f"\nMemory estimate: {mem_mb:.0f} MB per batch "
          f"(batch={batch_size}, T={chunk_frames}, D={hidden_dim})")

    # Data loaders
    use_mps = torch.backends.mps.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=2,
        pin_memory=not use_mps,
        persistent_workers=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Create task (just the head, no backbone)
    task = ConfidenceHeadTask(config)

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints",
            monitor="val/confidence_brier",
            mode="min",
            save_top_k=3,
            filename="bgc-{epoch:02d}-{val/confidence_brier:.4f}",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/confidence_brier",
            mode="min",
            patience=config["training"]["early_stopping_patience"],
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer — CPU is fine for 513 params
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="cpu",  # Tiny model, CPU is plenty
        devices=1,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        log_every_n_steps=config["logging"]["log_every_n_steps"],
    )

    print(f"\nConfig: {args.config}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Output: {args.output_dir}\n")

    trainer.fit(task, train_loader, val_loader)


if __name__ == "__main__":
    main()
