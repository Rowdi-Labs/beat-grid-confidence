"""Evaluation script: compute standard + product metrics on a test set.

Usage:
    # Evaluate baseline beat_this (no confidence heads)
    python scripts/evaluate.py --baseline --annotations-dir /path/to/beat_this_annotations --spectrogram-dir /path/to/spectrograms

    # Evaluate trained model with confidence heads
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.ckpt --annotations-dir /path/to/beat_this_annotations --spectrogram-dir /path/to/spectrograms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from beat_grid_confidence.dataset import (
    FRAME_RATE,
    BeatGridConfidenceDataset,
    load_all_annotations,
    make_splits,
)
from beat_grid_confidence.decode import decode_confidence_aware
from beat_grid_confidence.evaluation import (
    compute_confidence_brier,
    compute_continuity_span,
    compute_correction_effort,
    compute_relock_latency,
)
from beat_grid_confidence.model import create_model

console = Console()


def evaluate_track(
    outputs: dict[str, np.ndarray],
    beat_times_gt: np.ndarray,
    downbeat_times_gt: np.ndarray,
) -> dict[str, float]:
    """Evaluate a single track with standard + product metrics."""

    # Decode grid from model outputs
    confidence = outputs.get("confidence")
    if confidence is None:
        confidence = np.ones(outputs["beat_logits"].shape[0])

    grid = decode_confidence_aware(
        beat_logits=outputs["beat_logits"],
        downbeat_logits=outputs["downbeat_logits"],
        confidence=confidence,
        tempo_distribution=outputs.get("tempo_distribution"),
        frame_rate=FRAME_RATE,
    )

    pred_beats = grid.primary.beats
    pred_downbeats = grid.primary.downbeats

    results: dict[str, float] = {}

    # Standard metrics (mir_eval)
    if len(pred_beats) > 0 and len(beat_times_gt) > 0:
        beat_scores = mir_eval.beat.evaluate(beat_times_gt, pred_beats)
        results["beat_f1"] = beat_scores["F-measure"]
        results["beat_cemgil"] = beat_scores["Cemgil"]
        results["beat_cml_t"] = beat_scores["CMLt"]
        results["beat_cml_c"] = beat_scores["CMLc"]
        results["beat_aml_t"] = beat_scores["AMLt"]
        results["beat_aml_c"] = beat_scores["AMLc"]

    if len(pred_downbeats) > 0 and len(downbeat_times_gt) > 0:
        db_scores = mir_eval.beat.evaluate(downbeat_times_gt, pred_downbeats)
        results["downbeat_f1"] = db_scores["F-measure"]

    # Product metrics
    results["continuity_span"] = compute_continuity_span(pred_beats, beat_times_gt)
    results["correction_effort"] = float(compute_correction_effort(pred_beats, beat_times_gt))

    # Confidence-specific metrics (only if confidence head is present)
    if "confidence" in outputs and outputs["confidence"] is not None:
        # Generate correctness mask for calibration
        tolerance_frames = int(0.050 * FRAME_RATE)
        n_frames = len(confidence)
        correctness = np.zeros(n_frames)
        gt_frames = np.round(beat_times_gt * FRAME_RATE).astype(np.int64)
        gt_frames = gt_frames[(gt_frames >= 0) & (gt_frames < n_frames)]

        for f in range(n_frames):
            if len(gt_frames) > 0:
                min_dist = np.min(np.abs(gt_frames - f))
                correctness[f] = 1.0 if min_dist <= tolerance_frames else 0.0

        results["confidence_brier"] = compute_confidence_brier(confidence, correctness)
        results["relock_latency"] = compute_relock_latency(
            confidence, pred_beats, beat_times_gt, frame_rate=FRAME_RATE
        )
        results["n_low_conf_regions"] = float(len(grid.low_confidence_regions))
        results["n_alternate_hypotheses"] = float(len(grid.alternates))

    return results


def run_evaluation(
    model: torch.nn.Module,
    annotations: list,
    spectrogram_dir: Path,
    device: str,
) -> dict[str, list[float]]:
    """Run evaluation across all test tracks."""

    dataset = BeatGridConfidenceDataset(
        annotations=annotations,
        spectrogram_dir=spectrogram_dir,
        chunk_frames=0,  # use full track length
        augment=False,
    )

    all_results: dict[str, list[float]] = {}

    for i, ann in enumerate(annotations):
        try:
            item = dataset[i]
        except FileNotFoundError:
            continue

        spectrogram = item["spectrogram"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(spectrogram)

        # Convert to numpy
        np_outputs = {
            k: v.squeeze(0).cpu().numpy()
            for k, v in outputs.items()
            if isinstance(v, torch.Tensor)
        }

        track_results = evaluate_track(
            np_outputs, ann.beat_times, ann.downbeat_times
        )

        for metric, value in track_results.items():
            all_results.setdefault(metric, []).append(value)

        if (i + 1) % 50 == 0:
            console.print(f"  Evaluated {i + 1}/{len(annotations)} tracks...")

    return all_results


def print_results(results: dict[str, list[float]], title: str = "Results") -> None:
    """Pretty-print evaluation results."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right", style="yellow")
    table.add_column("N", justify="right")

    # Group metrics
    standard = ["beat_f1", "beat_cemgil", "beat_cml_t", "beat_cml_c", "beat_aml_t", "beat_aml_c", "downbeat_f1"]
    product = ["continuity_span", "correction_effort", "relock_latency", "confidence_brier"]

    for metric in standard + product:
        if metric in results:
            vals = results[metric]
            table.add_row(
                metric,
                f"{np.mean(vals):.4f}",
                f"{np.std(vals):.4f}",
                str(len(vals)),
            )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate beat-grid-confidence model")
    parser.add_argument("--checkpoint", type=Path, help="Trained model checkpoint")
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline beat_this only")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--spectrogram-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load annotations
    console.print("[bold]Loading annotations...[/bold]")
    all_annotations = load_all_annotations(args.annotations_dir)
    _, _, test_anns = make_splits(all_annotations, val_fold=args.val_fold, test_fold=args.test_fold)

    # Load model
    if args.baseline:
        console.print("[bold]Loading baseline beat_this...[/bold]")
        model = create_model(
            checkpoint_path="final0",
            device=args.device,
            enable_confidence=False,
            enable_tempo=False,
        )
    elif args.checkpoint:
        console.print(f"[bold]Loading checkpoint: {args.checkpoint}[/bold]")
        # TODO: Load from Lightning checkpoint
        raise NotImplementedError("Lightning checkpoint loading — use --baseline for now")
    else:
        parser.error("Specify --checkpoint or --baseline")

    model.eval()

    # Run evaluation
    console.print(f"[bold]Evaluating on {len(test_anns)} test tracks...[/bold]")
    results = run_evaluation(model, test_anns, args.spectrogram_dir, args.device)

    # Print results
    print_results(results, title="Test Set Results")

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for metric, vals in results.items()
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
