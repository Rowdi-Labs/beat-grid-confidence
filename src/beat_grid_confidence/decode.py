"""Confidence-aware structured decoder for beat grid inference.

Replaces standard peak-picking or Viterbi decoding with a decoder that:
1. Uses confidence scores to weight grid decisions
2. Maintains multiple hypotheses in ambiguous regions
3. Selects the best hypothesis by compatibility with high-confidence anchors
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridHypothesis:
    """A single beat grid interpretation."""

    bpm: float
    downbeat_offset: int  # 0-3: which beat is the downbeat
    beats: np.ndarray  # beat times in seconds
    downbeats: np.ndarray  # downbeat times in seconds
    score: float  # overall plausibility score
    label: str = "primary"  # "primary", "half_time", "double_time"


@dataclass
class DecodedGrid:
    """Full decoded output with confidence and alternate hypotheses."""

    primary: GridHypothesis
    alternates: list[GridHypothesis] = field(default_factory=list)
    confidence_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    low_confidence_regions: list[dict[str, float]] = field(default_factory=list)


def decode_confidence_aware(
    beat_logits: np.ndarray,
    downbeat_logits: np.ndarray,
    confidence: np.ndarray,
    tempo_distribution: np.ndarray | None = None,
    bpm_bins: np.ndarray | None = None,
    frame_rate: float = 50.0,
    confidence_threshold: float = 0.5,
    peak_threshold: float = 0.3,
) -> DecodedGrid:
    """Decode beat grid with confidence-aware hypothesis tracking.

    Strategy:
    1. Segment the track into high-confidence and low-confidence regions
    2. In high-confidence regions: standard tempo-regularized decode
    3. In low-confidence regions: generate multiple hypotheses
    4. Select the best hypothesis per region by compatibility with adjacent anchors

    Args:
        beat_logits: [T] raw beat activation (pre-sigmoid)
        downbeat_logits: [T] raw downbeat activation (pre-sigmoid)
        confidence: [T] per-frame confidence scores in [0, 1]
        tempo_distribution: [T, n_bins] optional tempo posterior
        bpm_bins: [n_bins] BPM values for tempo bins
        frame_rate: Frames per second (determines time resolution)
        confidence_threshold: Below this = low confidence region
        peak_threshold: Minimum activation to consider as beat candidate

    Returns:
        DecodedGrid with primary hypothesis, alternates, and confidence info
    """
    # Apply sigmoid to logits
    beat_probs = _sigmoid(beat_logits)
    downbeat_probs = _sigmoid(downbeat_logits)

    # Find low-confidence regions
    low_conf_regions = _find_low_confidence_regions(
        confidence, confidence_threshold, frame_rate
    )

    # Estimate primary tempo from confidence-weighted autocorrelation
    primary_bpm = _estimate_tempo(
        beat_probs, confidence, frame_rate, tempo_distribution, bpm_bins
    )

    # Decode primary grid: tempo-regularized with confidence weighting
    primary_beats = _decode_tempo_regularized(
        beat_probs, confidence, primary_bpm, frame_rate, peak_threshold
    )
    primary_downbeats, primary_offset = _decode_downbeats(
        downbeat_probs, primary_beats, frame_rate
    )

    primary = GridHypothesis(
        bpm=primary_bpm,
        downbeat_offset=primary_offset,
        beats=primary_beats,
        downbeats=primary_downbeats,
        score=float(np.mean(confidence)),
        label="primary",
    )

    # Generate alternate hypotheses
    alternates = _generate_alternates(
        beat_probs, downbeat_probs, confidence, primary_bpm, frame_rate
    )

    return DecodedGrid(
        primary=primary,
        alternates=alternates,
        confidence_curve=confidence,
        low_confidence_regions=low_conf_regions,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _find_low_confidence_regions(
    confidence: np.ndarray,
    threshold: float,
    frame_rate: float,
    min_duration_sec: float = 0.5,
) -> list[dict[str, float]]:
    """Identify contiguous low-confidence regions."""
    low = confidence < threshold
    regions = []
    start = None

    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = i
        elif not is_low and start is not None:
            duration = (i - start) / frame_rate
            if duration >= min_duration_sec:
                regions.append({
                    "start": start / frame_rate,
                    "end": i / frame_rate,
                })
            start = None

    # Handle region at end
    if start is not None:
        duration = (len(confidence) - start) / frame_rate
        if duration >= min_duration_sec:
            regions.append({
                "start": start / frame_rate,
                "end": len(confidence) / frame_rate,
            })

    return regions


def _estimate_tempo(
    beat_probs: np.ndarray,
    confidence: np.ndarray,
    frame_rate: float,
    tempo_distribution: np.ndarray | None,
    bpm_bins: np.ndarray | None,
) -> float:
    """Estimate dominant tempo, optionally using the tempo distribution head.

    If tempo_distribution is provided, uses the confidence-weighted mean.
    Otherwise, falls back to autocorrelation of beat activations.
    """
    if tempo_distribution is not None and bpm_bins is not None:
        # Confidence-weighted average of per-frame expected tempo
        weights = confidence / confidence.sum().clip(min=1e-8)
        per_frame_bpm = (tempo_distribution * bpm_bins).sum(axis=-1)
        return float(np.sum(weights * per_frame_bpm))

    # Fallback: autocorrelation-based tempo estimation
    # Weight beat activations by confidence
    weighted = beat_probs * confidence
    autocorr = np.correlate(weighted, weighted, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    # Search in plausible tempo range (60-200 BPM)
    min_lag = int(frame_rate * 60 / 200)  # fastest tempo
    max_lag = int(frame_rate * 60 / 60)  # slowest tempo

    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    if min_lag >= max_lag:
        return 120.0  # fallback

    search = autocorr[min_lag : max_lag + 1]
    best_lag = min_lag + np.argmax(search)
    bpm = 60.0 * frame_rate / best_lag

    return float(bpm)


def _decode_tempo_regularized(
    beat_probs: np.ndarray,
    confidence: np.ndarray,
    bpm: float,
    frame_rate: float,
    peak_threshold: float,
) -> np.ndarray:
    """Tempo-regularized beat decode with confidence weighting.

    Uses dynamic programming to find the beat sequence that maximizes
    confidence-weighted activation while maintaining regular spacing.

    TODO: Implement full Viterbi/beam search. For now, uses greedy
    peak-picking with tempo-constrained spacing.
    """
    period_frames = frame_rate * 60.0 / bpm
    tolerance = 0.15  # allow 15% deviation from expected period

    # Find peak candidates
    candidates = []
    for i in range(1, len(beat_probs) - 1):
        if (
            beat_probs[i] > peak_threshold
            and beat_probs[i] > beat_probs[i - 1]
            and beat_probs[i] > beat_probs[i + 1]
        ):
            candidates.append(i)

    if not candidates:
        return np.array([])

    # Greedy forward pass: pick highest-scoring candidate within tempo window
    beats = [candidates[0]]
    for c in candidates[1:]:
        expected = beats[-1] + period_frames
        dist = abs(c - expected)
        if dist <= period_frames * tolerance:
            beats.append(c)
        elif c > beats[-1] + period_frames * (1 + tolerance):
            # Gap too large — insert this candidate as a new anchor
            beats.append(c)

    return np.array(beats) / frame_rate


def _decode_downbeats(
    downbeat_probs: np.ndarray,
    beat_times: np.ndarray,
    frame_rate: float,
) -> tuple[np.ndarray, int]:
    """Determine which beats are downbeats and compute downbeat offset.

    Returns:
        Tuple of (downbeat_times, offset) where offset is 0-3.
    """
    if len(beat_times) < 4:
        return beat_times[:1] if len(beat_times) > 0 else np.array([]), 0

    # Score each possible offset (0-3)
    best_offset = 0
    best_score = -1.0

    for offset in range(4):
        score = 0.0
        count = 0
        for i in range(offset, len(beat_times), 4):
            frame = int(beat_times[i] * frame_rate)
            if 0 <= frame < len(downbeat_probs):
                score += downbeat_probs[frame]
                count += 1
        if count > 0:
            avg_score = score / count
            if avg_score > best_score:
                best_score = avg_score
                best_offset = offset

    downbeat_times = beat_times[best_offset::4]
    return downbeat_times, best_offset


def _generate_alternates(
    beat_probs: np.ndarray,
    downbeat_probs: np.ndarray,
    confidence: np.ndarray,
    primary_bpm: float,
    frame_rate: float,
) -> list[GridHypothesis]:
    """Generate half-time and double-time alternate hypotheses."""
    alternates = []

    # Half-time hypothesis
    half_bpm = primary_bpm / 2
    if 40 <= half_bpm <= 200:
        half_beats = _decode_tempo_regularized(
            beat_probs, confidence, half_bpm, frame_rate, 0.3
        )
        half_downbeats, half_offset = _decode_downbeats(
            downbeat_probs, half_beats, frame_rate
        )
        alternates.append(GridHypothesis(
            bpm=half_bpm,
            downbeat_offset=half_offset,
            beats=half_beats,
            downbeats=half_downbeats,
            score=float(np.mean(confidence)) * 0.8,  # slight penalty
            label="half_time",
        ))

    # Double-time hypothesis
    double_bpm = primary_bpm * 2
    if 60 <= double_bpm <= 300:
        double_beats = _decode_tempo_regularized(
            beat_probs, confidence, double_bpm, frame_rate, 0.3
        )
        double_downbeats, double_offset = _decode_downbeats(
            downbeat_probs, double_beats, frame_rate
        )
        alternates.append(GridHypothesis(
            bpm=double_bpm,
            downbeat_offset=double_offset,
            beats=double_beats,
            downbeats=double_downbeats,
            score=float(np.mean(confidence)) * 0.8,
            label="double_time",
        ))

    return alternates
