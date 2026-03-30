"""Evaluation metrics: standard MIR + product-relevant metrics."""

from __future__ import annotations

from dataclasses import dataclass

import mir_eval
import numpy as np


@dataclass
class StandardMetrics:
    """Standard beat tracking metrics from mir_eval."""

    beat_f1: float
    beat_cemgil: float
    beat_goto: float
    beat_cml_t: float  # Correct Metrical Level, continuity required
    beat_cml_c: float
    beat_aml_t: float  # Allowed Metrical Level
    beat_aml_c: float
    downbeat_f1: float


@dataclass
class ProductMetrics:
    """Product-relevant metrics for DJ/DAW grid quality."""

    relock_latency_frames: float  # Mean frames to recover after low-confidence region
    correction_effort: int  # Number of manual anchors needed to fix the track
    continuity_span: float  # Longest run of correct beats (seconds)
    confidence_brier: float  # Brier score of confidence predictions
    hypothesis_recall_at_2: float  # Fraction where correct grid is in top-2
    hypothesis_recall_at_3: float  # Fraction where correct grid is in top-3


def compute_standard_metrics(
    predicted_beats: np.ndarray,
    reference_beats: np.ndarray,
    predicted_downbeats: np.ndarray,
    reference_downbeats: np.ndarray,
) -> StandardMetrics:
    """Compute standard MIR beat tracking metrics.

    Args:
        predicted_beats: Predicted beat times in seconds
        reference_beats: Ground-truth beat times in seconds
        predicted_downbeats: Predicted downbeat times in seconds
        reference_downbeats: Ground-truth downbeat times in seconds

    Returns:
        StandardMetrics with all computed values
    """
    beat_scores = mir_eval.beat.evaluate(reference_beats, predicted_beats)
    downbeat_scores = mir_eval.beat.evaluate(reference_downbeats, predicted_downbeats)

    return StandardMetrics(
        beat_f1=beat_scores["F-measure"],
        beat_cemgil=beat_scores["Cemgil"],
        beat_goto=beat_scores["Goto"],
        beat_cml_t=beat_scores["CMLt"],
        beat_cml_c=beat_scores["CMLc"],
        beat_aml_t=beat_scores["AMLt"],
        beat_aml_c=beat_scores["AMLc"],
        downbeat_f1=downbeat_scores["F-measure"],
    )


def compute_relock_latency(
    confidence: np.ndarray,
    predicted_beats: np.ndarray,
    reference_beats: np.ndarray,
    confidence_threshold: float = 0.5,
    tolerance_sec: float = 0.050,
    frame_rate: float = 50.0,
) -> float:
    """Measure how quickly the grid re-establishes after a low-confidence region.

    Finds regions where confidence drops below threshold, then measures
    how many frames after confidence recovers until beats are correct again.

    Args:
        confidence: [T] per-frame confidence scores
        predicted_beats: Predicted beat times in seconds
        reference_beats: Ground-truth beat times in seconds
        confidence_threshold: Below this = low confidence
        tolerance_sec: Beat correctness tolerance
        frame_rate: Frames per second

    Returns:
        Mean relock latency in frames. Lower is better.
    """
    # Find low-confidence region boundaries
    low_conf = confidence < confidence_threshold
    transitions = np.diff(low_conf.astype(int))
    recovery_points = np.where(transitions == -1)[0]  # low -> high

    if len(recovery_points) == 0:
        return 0.0

    relock_latencies = []
    for recovery_frame in recovery_points:
        recovery_time = recovery_frame / frame_rate

        # Find the first correct beat after recovery
        post_beats = predicted_beats[predicted_beats >= recovery_time]
        for beat_time in post_beats:
            # Check if this beat is correct
            if len(reference_beats) > 0:
                min_dist = np.min(np.abs(reference_beats - beat_time))
                if min_dist <= tolerance_sec:
                    relock_frames = (beat_time - recovery_time) * frame_rate
                    relock_latencies.append(relock_frames)
                    break

    return float(np.mean(relock_latencies)) if relock_latencies else float("inf")


def compute_continuity_span(
    predicted_beats: np.ndarray,
    reference_beats: np.ndarray,
    tolerance_sec: float = 0.050,
) -> float:
    """Find the longest uninterrupted run of correct beats.

    Args:
        predicted_beats: Predicted beat times in seconds
        reference_beats: Ground-truth beat times in seconds
        tolerance_sec: Beat correctness tolerance

    Returns:
        Duration of longest correct span in seconds
    """
    if len(predicted_beats) == 0 or len(reference_beats) == 0:
        return 0.0

    # Check each predicted beat for correctness
    correct = np.array([
        np.min(np.abs(reference_beats - bt)) <= tolerance_sec for bt in predicted_beats
    ])

    # Find longest run of True values
    max_span = 0.0
    current_start = -1

    for i, is_correct in enumerate(correct):
        if is_correct and current_start == -1:
            current_start = i
        elif not is_correct and current_start != -1:
            span = predicted_beats[i - 1] - predicted_beats[current_start]
            max_span = max(max_span, span)
            current_start = -1

    # Check final run
    if current_start != -1:
        span = predicted_beats[-1] - predicted_beats[current_start]
        max_span = max(max_span, span)

    return max_span


def compute_confidence_brier(
    confidence: np.ndarray,
    correctness_mask: np.ndarray,
) -> float:
    """Brier score for confidence calibration.

    Measures how well the confidence score predicts actual correctness.
    Lower is better. 0.0 = perfect calibration.

    Args:
        confidence: [T] predicted confidence scores in [0, 1]
        correctness_mask: [T] binary actual correctness

    Returns:
        Brier score (mean squared error between confidence and correctness)
    """
    return float(np.mean((confidence - correctness_mask) ** 2))


def compute_correction_effort(
    predicted_beats: np.ndarray,
    reference_beats: np.ndarray,
    tolerance_sec: float = 0.050,
    min_gap_sec: float = 2.0,
) -> int:
    """Estimate the number of manual anchor corrections needed.

    Counts the number of distinct incorrect regions (gaps between correct spans
    longer than min_gap_sec). Each such region would require at least one manual
    anchor point to fix.

    Args:
        predicted_beats: Predicted beat times in seconds
        reference_beats: Ground-truth beat times in seconds
        tolerance_sec: Beat correctness tolerance
        min_gap_sec: Minimum gap to count as a separate correction region

    Returns:
        Number of anchor corrections needed. 0 = perfect grid.
    """
    if len(predicted_beats) == 0 or len(reference_beats) == 0:
        return len(reference_beats)  # Everything needs correction

    correct = np.array([
        np.min(np.abs(reference_beats - bt)) <= tolerance_sec for bt in predicted_beats
    ])

    # Find incorrect regions
    incorrect_regions = 0
    in_incorrect = False

    for i, is_correct in enumerate(correct):
        if not is_correct and not in_incorrect:
            in_incorrect = True
            incorrect_regions += 1
        elif is_correct:
            in_incorrect = False

    return incorrect_regions
