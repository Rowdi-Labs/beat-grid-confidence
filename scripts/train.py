"""Training entry point for beat-grid-confidence heads.

Usage:
    # Phase 1: confidence head only
    python scripts/train.py --config configs/confidence_only.yaml --annotations-dir /path/to/beat_this_annotations --spectrogram-dir /path/to/spectrograms

    # Full model
    python scripts/train.py --config configs/base.yaml --annotations-dir /path/to/beat_this_annotations --spectrogram-dir /path/to/spectrograms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from beat_grid_confidence.dataset import (
    BeatGridConfidenceDataset,
    load_all_annotations,
    make_splits,
)
from beat_grid_confidence.model import create_model


class BeatGridConfidenceTask(pl.LightningModule):
    """PyTorch Lightning training task for confidence heads."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Create model with frozen backbone + trainable heads
        self.model = create_model(
            checkpoint_path=config["model"]["checkpoint"] or "final0",
            device="cpu",  # Lightning handles device placement
            freeze_backbone=config["model"]["freeze_backbone"],
            enable_confidence=config["model"]["heads"]["confidence"],
            enable_tempo=config["model"]["heads"]["tempo_distribution"],
            n_tempo_bins=config["model"]["tempo_bins"],
        )

        # Loss functions
        self.beat_tolerance_frames = int(
            config["data"]["beat_tolerance_sec"] * (22050 / 441)
        )

    def forward(self, spectrogram: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(spectrogram)

    def _compute_correctness_mask(
        self,
        beat_logits: torch.Tensor,
        beat_target: torch.Tensor,
    ) -> torch.Tensor:
        """Generate binary correctness mask for confidence training.

        For each frame where the model predicts a beat (logit > 0), check
        if there's a ground-truth beat within tolerance. 1 = correct, 0 = wrong.
        """
        batch_size, n_frames = beat_logits.shape
        predicted_active = beat_logits > 0  # pre-sigmoid threshold

        mask = torch.zeros_like(beat_logits)
        for b in range(batch_size):
            gt_frames = torch.where(beat_target[b] > 0.5)[0]
            if len(gt_frames) == 0:
                continue

            for f in range(n_frames):
                if not predicted_active[b, f]:
                    # Not predicting a beat here — confidence is less relevant
                    # Use ground truth: if there's a beat nearby, confidence should be high
                    pass
                # Check if any ground truth beat is within tolerance
                min_dist = torch.min(torch.abs(gt_frames.float() - f))
                mask[b, f] = 1.0 if min_dist <= self.beat_tolerance_frames else 0.0

        return mask

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch["spectrogram"])

        total_loss = torch.tensor(0.0, device=self.device)

        # Confidence loss
        if "confidence" in outputs:
            correctness = self._compute_correctness_mask(
                outputs["beat_logits"], batch["beat_target"]
            )
            conf_loss = torch.nn.functional.binary_cross_entropy(
                outputs["confidence"], correctness
            )
            total_loss = total_loss + self.config["loss"]["lambda_confidence"] * conf_loss
            self.log("train/confidence_loss", conf_loss, prog_bar=True)

        # Tempo distribution loss (Phase 2)
        if "tempo_distribution" in outputs:
            # TODO: Implement tempo target generation
            pass

        self.log("train/total_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(batch["spectrogram"])

        if "confidence" in outputs:
            correctness = self._compute_correctness_mask(
                outputs["beat_logits"], batch["beat_target"]
            )
            conf_loss = torch.nn.functional.binary_cross_entropy(
                outputs["confidence"], correctness
            )
            # Brier score = MSE between confidence and correctness
            brier = torch.mean((outputs["confidence"] - correctness) ** 2)

            self.log("val/confidence_loss", conf_loss, prog_bar=True)
            self.log("val/confidence_brier", brier, prog_bar=True)

    def configure_optimizers(self) -> dict:
        # Only optimize head parameters (backbone is frozen)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["training"]["max_epochs"],
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train beat-grid-confidence heads")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/confidence_only.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, required=True,
        help="Path to beat_this_annotations clone",
    )
    parser.add_argument(
        "--spectrogram-dir", type=Path, required=True,
        help="Path to pre-extracted spectrograms",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--val-fold", type=int, default=0, help="Validation fold (0-7)",
    )
    parser.add_argument(
        "--test-fold", type=int, default=1, help="Test fold (0-7)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config["model"]["checkpoint"] is None:
        config["model"]["checkpoint"] = "final0"

    # Load annotations
    print("Loading annotations...")
    all_annotations = load_all_annotations(args.annotations_dir)
    train_anns, val_anns, test_anns = make_splits(
        all_annotations, val_fold=args.val_fold, test_fold=args.test_fold
    )

    # Create datasets
    train_dataset = BeatGridConfidenceDataset(
        annotations=train_anns,
        spectrogram_dir=args.spectrogram_dir,
        chunk_frames=config["data"]["chunk_frames"],
        augment=config["data"]["augment"],
    )
    val_dataset = BeatGridConfidenceDataset(
        annotations=val_anns,
        spectrogram_dir=args.spectrogram_dir,
        chunk_frames=config["data"]["chunk_frames"],
        augment=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create Lightning task
    task = BeatGridConfidenceTask(config)

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints",
            monitor=config["logging"]["monitor"],
            mode=config["logging"]["mode"],
            save_top_k=config["logging"]["save_top_k"],
            filename="bgc-{epoch:02d}-{val/confidence_brier:.4f}",
        ),
        pl.callbacks.EarlyStopping(
            monitor=config["logging"]["monitor"],
            mode=config["logging"]["mode"],
            patience=config["training"]["early_stopping_patience"],
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["hardware"]["accelerator"],
        devices=config["hardware"]["devices"],
        precision=config["hardware"]["precision"],
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        log_every_n_steps=config["logging"]["log_every_n_steps"],
    )

    # Train
    print(f"\nConfig: {args.config}")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Heads: {config['model']['heads']}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Output: {args.output_dir}\n")

    trainer.fit(task, train_loader, val_loader)


if __name__ == "__main__":
    main()
