"""Generate all paper figures from evaluation data.

Usage:
    python scripts/generate_figures.py \
        --hidden-states-dir data/hidden_states \
        --annotations-dir data/beat_this_annotations \
        --checkpoint outputs/v4/checkpoints/bgc-epoch=23-val/confidence_brier=0.0255.ckpt \
        --output-dir paper/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

# Consistent style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#059669",
    "muted": "#6B7280",
    "light": "#E5E7EB",
}


def load_per_track_data(
    hidden_states_dir: Path,
    annotations_dir: Path,
    checkpoint_path: Path | None,
) -> list[dict]:
    """Load per-track metrics for all available tracks."""
    import mir_eval
    from beat_grid_confidence.dataset import load_all_annotations, FRAME_RATE
    from beat_grid_confidence.decode import decode_confidence_aware
    from beat_grid_confidence.evaluation import compute_correction_effort, compute_continuity_span
    from beat_grid_confidence.heads import ConfidenceHead

    # Load confidence head if provided
    head = None
    if checkpoint_path:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = {k.replace("confidence_head.", ""): v
                 for k, v in ckpt["state_dict"].items() if "confidence_head" in k}
        if "net.0.weight" in state:
            hidden_dim = state["net.0.weight"].shape[1]
            bottleneck = state["net.0.weight"].shape[0]
            head = ConfidenceHead(hidden_dim, bottleneck=bottleneck)
        else:
            hidden_dim = state["projection.weight"].shape[1]
            head = ConfidenceHead(hidden_dim)
        head.load_state_dict(state)
        head.eval()

    anns = load_all_annotations(annotations_dir, datasets=["ballroom", "gtzan"])
    per_track = []

    for ann in anns:
        ds_dir = hidden_states_dir / ann.dataset
        hp = ds_dir / f"{ann.stem}.hidden.npy"
        lp = ds_dir / f"{ann.stem}.logits.npz"
        if not hp.exists() or not lp.exists():
            continue

        h_np = np.load(hp).astype(np.float32)
        with np.load(lp) as lf:
            bl = lf["beat_logits"]
            dl = lf["downbeat_logits"]

        # Confidence
        if head is not None:
            with torch.no_grad():
                conf = head(torch.from_numpy(h_np).unsqueeze(0)).squeeze(0).numpy()
        else:
            conf = np.ones(len(bl))

        # Activations
        beat_sigmoid = 1 / (1 + np.exp(-bl))
        peaks_mask = np.zeros(len(beat_sigmoid), dtype=bool)
        for i in range(1, len(beat_sigmoid) - 1):
            if beat_sigmoid[i] > 0.3 and beat_sigmoid[i] > beat_sigmoid[i-1] and beat_sigmoid[i] > beat_sigmoid[i+1]:
                peaks_mask[i] = True
        peak_mean = float(np.mean(beat_sigmoid[peaks_mask])) if peaks_mask.any() else 0.0

        # Decode — tempo-regularized
        grid = decode_confidence_aware(bl, dl, conf, frame_rate=FRAME_RATE)
        pred_beats = grid.primary.beats

        # Simple peak-pick
        simple_beats = np.where(peaks_mask)[0] / FRAME_RATE

        if len(pred_beats) == 0 or len(ann.beat_times) == 0:
            continue

        scores_reg = mir_eval.beat.evaluate(ann.beat_times, pred_beats)
        scores_simple = mir_eval.beat.evaluate(ann.beat_times, simple_beats) if len(simple_beats) > 0 else {"F-measure": 0.0}

        per_track.append({
            "stem": ann.stem,
            "dataset": ann.dataset,
            "f1_regularized": scores_reg["F-measure"],
            "f1_simple": scores_simple["F-measure"],
            "effort_regularized": compute_correction_effort(pred_beats, ann.beat_times),
            "effort_simple": compute_correction_effort(simple_beats, ann.beat_times) if len(simple_beats) > 0 else 999,
            "continuity_regularized": compute_continuity_span(pred_beats, ann.beat_times),
            "continuity_simple": compute_continuity_span(simple_beats, ann.beat_times) if len(simple_beats) > 0 else 0.0,
            "mean_conf": float(np.mean(conf)),
            "peak_mean": peak_mean,
            "beat_sigmoid": beat_sigmoid,
            "conf_curve": conf,
            "beat_times_gt": ann.beat_times,
            "pred_beats": pred_beats,
        })

    return per_track


def fig1_decoder_comparison(data: list[dict], out_dir: Path) -> None:
    """Figure 1: F1 vs correction effort for both decoders — scatter + marginals."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel A: F1 comparison
    ax = axes[0]
    f1_reg = [d["f1_regularized"] for d in data]
    f1_sim = [d["f1_simple"] for d in data]
    ax.scatter(f1_sim, f1_reg, s=8, alpha=0.4, c=COLORS["primary"], edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", c=COLORS["muted"], lw=0.8, label="Equal")
    ax.set_xlabel("Peak-picking F1")
    ax.set_ylabel("Tempo-regularized F1")
    ax.set_title("(a) Per-track Beat F1")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=False)

    # Panel B: Correction effort comparison
    ax = axes[1]
    eff_reg = [d["effort_regularized"] for d in data]
    eff_sim = [d["effort_simple"] for d in data]
    ax.scatter(eff_sim, eff_reg, s=8, alpha=0.4, c=COLORS["secondary"], edgecolors="none")
    max_eff = max(max(eff_reg), max(eff_sim))
    ax.plot([0, max_eff], [0, max_eff], "--", c=COLORS["muted"], lw=0.8, label="Equal")
    ax.set_xlabel("Peak-picking effort")
    ax.set_ylabel("Tempo-regularized effort")
    ax.set_title("(b) Per-track Correction Effort")
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_decoder_comparison.pdf")
    fig.savefig(out_dir / "fig1_decoder_comparison.png")
    plt.close(fig)
    print(f"  fig1_decoder_comparison")


def fig2_confidence_vs_effort(data: list[dict], out_dir: Path) -> None:
    """Figure 2: Per-track confidence vs correction effort — trained vs heuristic."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    efforts = np.array([d["effort_regularized"] for d in data])
    mean_confs = np.array([d["mean_conf"] for d in data])
    peak_means = np.array([d["peak_mean"] for d in data])

    # Color by dataset
    colors = [COLORS["primary"] if d["dataset"] == "ballroom" else COLORS["secondary"] for d in data]

    # Panel A: Trained head
    ax = axes[0]
    ax.scatter(mean_confs, efforts, s=8, alpha=0.4, c=colors, edgecolors="none")
    rho, _ = spearmanr(mean_confs, efforts)
    ax.set_xlabel("Mean confidence (trained head)")
    ax.set_ylabel("Correction effort")
    ax.set_title(f"(a) Trained MLP ($\\rho$ = {rho:.2f})")

    # Panel B: Heuristic
    ax = axes[1]
    ax.scatter(peak_means, efforts, s=8, alpha=0.4, c=colors, edgecolors="none")
    rho2, _ = spearmanr(peak_means, efforts)
    ax.set_xlabel("Peak activation mean (no training)")
    ax.set_ylabel("Correction effort")
    ax.set_title(f"(b) Activation heuristic ($\\rho$ = {rho2:.2f})")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["primary"], markersize=5, label="Ballroom"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["secondary"], markersize=5, label="GTZAN"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right", frameon=False)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_confidence_vs_effort.pdf")
    fig.savefig(out_dir / "fig2_confidence_vs_effort.png")
    plt.close(fig)
    print(f"  fig2_confidence_vs_effort")


def fig3_triage_recall(data: list[dict], out_dir: Path) -> None:
    """Figure 3: Triage recall curve — checking top-K% by confidence, what % of bad tracks caught?"""
    efforts = np.array([d["effort_regularized"] for d in data])
    mean_confs = np.array([d["mean_conf"] for d in data])
    peak_means = np.array([d["peak_mean"] for d in data])

    bad_threshold = np.percentile(efforts, 75)
    is_bad = efforts >= bad_threshold
    n_bad = is_bad.sum()

    percentages = np.arange(1, 101)
    methods = {
        "Trained head": np.argsort(mean_confs),         # low confidence first
        "Peak activation": np.argsort(peak_means),       # low activation first
        "Random": np.random.RandomState(42).permutation(len(data)),
    }
    method_colors = {
        "Trained head": COLORS["primary"],
        "Peak activation": COLORS["accent"],
        "Random": COLORS["muted"],
    }

    fig, ax = plt.subplots(figsize=(4, 3))

    for name, order in methods.items():
        recalls = []
        for pct in percentages:
            n_check = max(1, len(data) * pct // 100)
            checked = set(order[:n_check])
            caught = sum(1 for i in range(len(data)) if i in checked and is_bad[i])
            recalls.append(caught / n_bad if n_bad > 0 else 0)
        ax.plot(percentages, recalls, label=name, color=method_colors[name],
                lw=1.5 if name != "Random" else 1.0,
                ls="-" if name != "Random" else "--")

    ax.set_xlabel("% of library checked")
    ax.set_ylabel("% of bad tracks found")
    ax.set_title("Triage: prioritizing grid corrections")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_triage_recall.pdf")
    fig.savefig(out_dir / "fig3_triage_recall.png")
    plt.close(fig)
    print(f"  fig3_triage_recall")


def fig4_example_tracks(data: list[dict], out_dir: Path) -> None:
    """Figure 4: Example waveforms — one easy track, one hard track, with confidence overlay."""
    # Find easiest and hardest tracks
    sorted_data = sorted(data, key=lambda d: d["effort_regularized"])
    easy = [d for d in sorted_data if d["f1_regularized"] > 0.95 and d["effort_regularized"] == 0]
    hard = [d for d in sorted_data if d["f1_regularized"] < 0.3 and d["effort_regularized"] > 2]

    if not easy or not hard:
        print("  fig4_example_tracks: skipped (not enough extreme examples)")
        return

    easy_track = easy[0]
    hard_track = hard[0]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.5), sharex=False)

    for ax, track, label in [(axes[0], easy_track, "Easy"), (axes[1], hard_track, "Hard")]:
        t = np.arange(len(track["beat_sigmoid"])) / 50.0  # 50 FPS

        # Beat activation
        ax.fill_between(t, track["beat_sigmoid"], alpha=0.3, color=COLORS["primary"], label="Beat activation")

        # Confidence overlay
        ax.plot(t[:len(track["conf_curve"])], track["conf_curve"],
                color=COLORS["secondary"], lw=1.0, alpha=0.8, label="Confidence")

        # GT beats as ticks
        for bt in track["beat_times_gt"]:
            if bt <= t[-1]:
                ax.axvline(bt, color=COLORS["accent"], alpha=0.3, lw=0.5)

        # Predicted beats as ticks
        for pb in track["pred_beats"]:
            if pb <= t[-1]:
                ax.axvline(pb, color=COLORS["secondary"], alpha=0.2, lw=0.5, ls="--")

        dataset_label = track["dataset"].capitalize()
        ax.set_title(f"{label}: {track['stem'][:40]} ({dataset_label}, F1={track['f1_regularized']:.2f}, effort={track['effort_regularized']:.0f})",
                     fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Activation")
        if ax == axes[1]:
            ax.set_xlabel("Time (s)")
        if ax == axes[0]:
            ax.legend(loc="upper right", frameon=False, fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_example_tracks.pdf")
    fig.savefig(out_dir / "fig4_example_tracks.png")
    plt.close(fig)
    print(f"  fig4_example_tracks")


def fig5_effort_distribution(data: list[dict], out_dir: Path) -> None:
    """Figure 5: Histogram of correction effort for both decoders."""
    fig, ax = plt.subplots(figsize=(4, 2.8))

    eff_reg = [d["effort_regularized"] for d in data]
    eff_sim = [d["effort_simple"] for d in data]

    bins = np.arange(0, max(max(eff_reg), max(eff_sim)) + 2) - 0.5
    ax.hist(eff_sim, bins=bins, alpha=0.5, color=COLORS["muted"], label=f"Peak-pick (mean={np.mean(eff_sim):.1f})", density=True)
    ax.hist(eff_reg, bins=bins, alpha=0.5, color=COLORS["primary"], label=f"Tempo-reg (mean={np.mean(eff_reg):.1f})", density=True)

    ax.set_xlabel("Correction effort (anchors)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of correction effort")
    ax.legend(frameon=False)
    ax.set_xlim(-0.5, 15)

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_effort_distribution.pdf")
    fig.savefig(out_dir / "fig5_effort_distribution.png")
    plt.close(fig)
    print(f"  fig5_effort_distribution")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--hidden-states-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading per-track data...")
    data = load_per_track_data(args.hidden_states_dir, args.annotations_dir, args.checkpoint)
    print(f"  {len(data)} tracks loaded")
    print()
    print("Generating figures:")

    fig1_decoder_comparison(data, args.output_dir)
    fig2_confidence_vs_effort(data, args.output_dir)
    fig3_triage_recall(data, args.output_dir)
    fig4_example_tracks(data, args.output_dir)
    fig5_effort_distribution(data, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
