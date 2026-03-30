"""Evaluation: standard + product metrics on test set.

Two modes:
  --baseline: evaluate backbone beat/downbeat logits only (no confidence)
  --checkpoint: evaluate backbone + trained confidence head

Both load pre-extracted logits + hidden states. No backbone inference at eval time.
Peak memory: ~50 MB (just loading numpy arrays + tiny head forward pass).

Usage:
    python scripts/evaluate.py --baseline \
        --hidden-states-dir data/hidden_states --annotations-dir data/beat_this_annotations --datasets ballroom

    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.ckpt \
        --hidden-states-dir data/hidden_states --annotations-dir data/beat_this_annotations --datasets ballroom
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
from beat_grid_confidence.heads import ConfidenceHead

console = Console()


def load_track_data(
    hidden_states_dir: Path,
    dataset: str,
    stem: str,
) -> dict[str, np.ndarray]:
    """Load pre-extracted hidden states + logits for one track."""
    ds_dir = hidden_states_dir / dataset

    # Load logits
    logits_path = ds_dir / f"{stem}.logits.npz"
    if not logits_path.exists():
        raise FileNotFoundError(f"No logits for {stem}. Run extract_hidden_states.py --force")

    with np.load(logits_path) as lf:
        beat_logits = lf["beat_logits"]
        downbeat_logits = lf["downbeat_logits"]

    # Load hidden states (for confidence head)
    hidden_path = ds_dir / f"{stem}.hidden.npy"
    hidden = None
    if hidden_path.exists():
        hidden = np.load(hidden_path).astype(np.float32)

    return {
        "beat_logits": beat_logits,
        "downbeat_logits": downbeat_logits,
        "hidden_states": hidden,
    }


def evaluate_track(
    outputs: dict[str, np.ndarray],
    beat_times_gt: np.ndarray,
    downbeat_times_gt: np.ndarray,
) -> dict[str, float]:
    """Evaluate a single track."""
    confidence = outputs.get("confidence")
    if confidence is None:
        confidence = np.ones(outputs["beat_logits"].shape[0])

    grid = decode_confidence_aware(
        beat_logits=outputs["beat_logits"],
        downbeat_logits=outputs["downbeat_logits"],
        confidence=confidence,
        frame_rate=FRAME_RATE,
    )

    pred_beats = grid.primary.beats
    pred_downbeats = grid.primary.downbeats
    results: dict[str, float] = {}

    # Standard metrics
    if len(pred_beats) > 0 and len(beat_times_gt) > 0:
        beat_scores = mir_eval.beat.evaluate(beat_times_gt, pred_beats)
        results["beat_f1"] = beat_scores["F-measure"]
        results["beat_cemgil"] = beat_scores["Cemgil"]
        results["beat_cml_t"] = beat_scores["Correct Metric Level Total"]
        results["beat_cml_c"] = beat_scores["Correct Metric Level Continuous"]
        results["beat_aml_t"] = beat_scores["Any Metric Level Total"]
        results["beat_aml_c"] = beat_scores["Any Metric Level Continuous"]

    if len(pred_downbeats) > 0 and len(downbeat_times_gt) > 0:
        db_scores = mir_eval.beat.evaluate(downbeat_times_gt, pred_downbeats)
        results["downbeat_f1"] = db_scores["F-measure"]

    # Product metrics
    results["continuity_span"] = compute_continuity_span(pred_beats, beat_times_gt)
    results["correction_effort"] = float(compute_correction_effort(pred_beats, beat_times_gt))

    # Confidence metrics
    if "confidence" in outputs and outputs["confidence"] is not None:
        # Regional accuracy correctness for calibration check
        tolerance_frames = int(0.050 * FRAME_RATE)
        n_frames = len(confidence)
        gt_frames = np.round(beat_times_gt * FRAME_RATE).astype(np.int64)
        gt_frames = gt_frames[(gt_frames >= 0) & (gt_frames < n_frames)]

        if len(gt_frames) > 0:
            all_frames = np.arange(n_frames)
            distances = np.abs(all_frames[:, None] - gt_frames[None, :])
            min_distances = distances.min(axis=1)
            correctness = (min_distances <= tolerance_frames).astype(np.float64)
        else:
            correctness = np.zeros(n_frames)

        results["confidence_brier"] = compute_confidence_brier(confidence, correctness)
        results["mean_confidence"] = float(np.mean(confidence))
        results["confidence_std"] = float(np.std(confidence))
        results["relock_latency"] = compute_relock_latency(
            confidence, pred_beats, beat_times_gt, frame_rate=FRAME_RATE
        )
        results["n_low_conf_regions"] = float(len(grid.low_confidence_regions))
        results["n_alternates"] = float(len(grid.alternates))

    return results


def print_results(results: dict[str, list[float]], title: str = "Results") -> None:
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right", style="yellow")
    table.add_column("N", justify="right")

    ordered = [
        "beat_f1", "beat_cemgil", "beat_cml_t", "beat_cml_c",
        "beat_aml_t", "beat_aml_c", "downbeat_f1",
        "continuity_span", "correction_effort",
        "confidence_brier", "mean_confidence", "confidence_std",
        "relock_latency", "n_low_conf_regions", "n_alternates",
    ]

    for metric in ordered:
        if metric in results:
            vals = results[metric]
            table.add_row(metric, f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}", str(len(vals)))

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate beat-grid-confidence model")
    parser.add_argument("--checkpoint", type=Path, help="Trained confidence head .ckpt")
    parser.add_argument("--baseline", action="store_true", help="No confidence head")
    parser.add_argument("--hidden-states-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--val-fold", type=int, default=0)
    args = parser.parse_args()

    # Load annotations
    console.print("[bold]Loading annotations...[/bold]")
    all_annotations = load_all_annotations(args.annotations_dir, datasets=args.datasets)
    _, _, test_anns = make_splits(all_annotations, val_fold=args.val_fold, test_fold=args.test_fold)

    # Load confidence head if not baseline
    confidence_head = None
    if args.checkpoint:
        console.print(f"[bold]Loading confidence head: {args.checkpoint}[/bold]")
        ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        head_state = {
            k.replace("confidence_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("confidence_head.")
        }
        hidden_dim = head_state["projection.weight"].shape[1]
        confidence_head = ConfidenceHead(hidden_dim)
        confidence_head.load_state_dict(head_state)
        confidence_head.eval()
        console.print(f"  Loaded {sum(p.numel() for p in confidence_head.parameters())} params (dim={hidden_dim})")
    elif not args.baseline:
        parser.error("Specify --checkpoint or --baseline")

    # Evaluate
    console.print(f"[bold]Evaluating {len(test_anns)} test tracks...[/bold]")
    all_results: dict[str, list[float]] = {}

    for i, ann in enumerate(test_anns):
        try:
            data = load_track_data(args.hidden_states_dir, ann.dataset, ann.stem)
        except FileNotFoundError as e:
            console.print(f"  [dim]Skipping {ann.stem}: {e}[/dim]")
            continue

        outputs: dict[str, np.ndarray] = {
            "beat_logits": data["beat_logits"],
            "downbeat_logits": data["downbeat_logits"],
        }

        # Run confidence head on hidden states
        if confidence_head is not None and data["hidden_states"] is not None:
            h = torch.from_numpy(data["hidden_states"]).unsqueeze(0)
            with torch.no_grad():
                conf = confidence_head(h)
            outputs["confidence"] = conf.squeeze(0).numpy()

        track_results = evaluate_track(outputs, ann.beat_times, ann.downbeat_times)
        for metric, value in track_results.items():
            all_results.setdefault(metric, []).append(value)

        if (i + 1) % 25 == 0:
            console.print(f"  {i + 1}/{len(test_anns)} tracks...")

    print_results(all_results, title="Test Set Results")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for metric, vals in all_results.items()
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
