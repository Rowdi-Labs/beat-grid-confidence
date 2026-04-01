# Methodology & Experiment Log

This document tracks all experimental decisions, results, and rationale for the beat-grid-confidence project. It serves as the primary source for the paper's methodology section and ensures reproducibility.

**Update protocol:** After every training run or significant experiment, add an entry under [Experiment Log](#experiment-log) with: config, data, results, and interpretation. Keep the [Current Best](#current-best) section pointing to the latest promising result.

---

## Research Question

Can lightweight prediction heads trained on frozen beat_this hidden states produce useful per-region confidence estimates for beat grid editing?

"Useful" means:
1. **Calibrated:** When the model says 0.9 confidence, the backbone is correct ~90% of the time in that region.
2. **Discriminative:** Confidence varies meaningfully between easy and hard regions.
3. **Actionable:** Low-confidence regions correspond to passages where the backbone actually fails (breakdowns, fills, sparse intros, genre-specific difficulty).

## Model Architecture

### Backbone (frozen)
- **Model:** beat_this `final0` checkpoint (JKU Linz, ISMIR 2024)
- **License:** MIT
- **Architecture:** Conv frontend → 6-layer rotary-attention transformer → SumHead
- **Parameters:** 20.3M (all frozen)
- **Hidden dim:** 512
- **Frame rate:** 50 FPS (22050 Hz sample rate, 441 hop length)
- **Input:** Log-mel spectrogram [B, T, 128]

### Confidence Head (trainable)
| Version | Architecture | Parameters | Notes |
|---------|-------------|------------|-------|
| v1-v3 | Linear(512, 1) + sigmoid | 513 | Collapses to near-constant output |
| v4 | MLP: 512→64→1, ReLU+sigmoid | ~33K | Pending — hypothesis: needs nonlinearity |

### Pre-extraction Pipeline
Since the backbone is frozen, hidden states and beat/downbeat logits are deterministic per spectrogram. We pre-extract them once:

```
spectrogram [T, 128]  →  backbone.frontend()
                      →  backbone.transformer_blocks()
                      →  hidden_states [T, 512]  (saved as float16 .npy)
                      →  backbone.task_heads()
                      →  beat_logits [T], downbeat_logits [T]  (saved as float32 .npz)
```

This eliminates the 20M-param backbone from the training loop. Training memory drops from ~16GB (attention O(T²)) to ~95MB (loading pre-extracted vectors).

**Critical:** Use full forward pass (`--chunk-frames 0`), not chunked inference. Chunked extraction with overlap blending introduces ~6% Beat F1 degradation. Verified: full forward pass reproduces baseline exactly (0.899 F1).

## Confidence Target Design

### Attempt 1: Beat proximity (v1) — WRONG
- **Target:** Binary mask, 1 if GT beat is within 50ms (3 frames at 50 FPS), 0 otherwise
- **Problem:** At 120 BPM, only ~24% of frames are near a beat. The head learned to predict beat proximity (mean confidence 0.288), not backbone quality. This is just a redundant beat detector, not a confidence estimator.

### Attempt 2: Regional backbone accuracy (v2-v3) — CORRECT SIGNAL, NEEDS MORE CAPACITY
- **Target:** For each frame, look at a 1-second window (50 frames). Find all frames where the backbone predicts a beat (sigmoid(logit) > 0.3, local peak). Of those predicted beats, what fraction have a GT beat within 50ms?
- **Implementation:** `HiddenStatesDataset._compute_regional_accuracy()` in `dataset.py`
- **Properties:**
  - Continuous [0, 1] — not binary
  - 0.5 for uncertain regions (no predictions in window)
  - Directly answers "is the backbone reliable here?"
  - Computed from pre-extracted logits, so no backbone needed during training

### Evaluation Metric Alignment
**Known issue (as of v3):** The evaluation script computes Brier score against beat-proximity (the v1 target), not regional accuracy (the training target). This makes the reported Brier score misleading. TODO: fix eval to compute calibration against the matching target.

## Datasets

### Training Data
| Dataset | Source | License | Tracks | Notes |
|---------|--------|---------|--------|-------|
| Ballroom | beat_this_annotations | MIT (annotations) | 685 | Easy for beat_this — F1 0.899 |
| GTZAN mini | mirdata download | Research use | 100 | 10 genres, harder — F1 0.839 |

### Evaluation Data
Same as training for now (fold-based splits for Ballroom, all tracks for GTZAN). Eventual paper should use held-out datasets.

### Data Processing
1. **Spectrograms:** 128 mel bins, 50 FPS, from beat_this preprocessing
2. **Annotations:** beat_this_annotations repo (`.beats` files, tab-separated time+position)
3. **Splits:** 8-fold from beat_this_annotations, fold 0 = val, fold 1 = test, rest = train

## Decoder

The confidence-aware decoder in `decode.py` replaces standard peak-picking:
- High-confidence regions: tempo-regularized decode (greedy, tempo-constrained spacing)
- Low-confidence regions: maintain multiple hypotheses (half-time, double-time)
- Alternate hypothesis tracking: always generate primary + half/double-time variants

**Current limitation:** The decoder is not yet confidence-weighted. It uses the same logic regardless of confidence scores because the confidence head doesn't yet produce useful variance.

---

## Experiment Log

### Experiment 0: Baseline Reproduction
- **Date:** 2026-03-30
- **Goal:** Verify beat_this `final0` on Ballroom matches published numbers
- **Data:** Ballroom test fold (86 tracks)
- **Method:** Pre-extracted spectrograms → full model forward pass → decoder → mir_eval
- **Results:**

| Metric | Our Result | Published |
|--------|-----------|-----------|
| Beat F1 | 0.899 | ~0.90-0.92 |
| AMLt | 0.928 | ~0.93 |
| CMLt | 0.791 | ~0.80 |
| Downbeat F1 | 0.672 | ~0.70 |

- **Conclusion:** Baseline reproduced within expected range. Phase 0 complete.

### Experiment 1: Confidence Head v1 — Beat Proximity Target
- **Date:** 2026-03-30
- **Config:** `configs/confidence_only.yaml`, Linear(512, 1), LR 3e-3, 30 epochs
- **Data:** Ballroom train fold (513 tracks), hidden states pre-extracted with chunked inference
- **Target:** Binary correctness mask (GT beat within 50ms)
- **Results:** Val Brier 0.080, Val accuracy 92.4%
- **Interpretation:** Looks good on paper, but the head learned beat proximity (mean confidence 0.288 ≈ base rate of frames near beats). Not actually predicting backbone quality. Also, chunked extraction degraded logits by ~6%.
- **Decision:** Wrong target. Redesign.

### Experiment 2: Confidence Head v2 — Regional Accuracy Target
- **Date:** 2026-03-31
- **Config:** Same architecture, same hyperparams
- **Data:** Ballroom train fold (513 tracks), hidden states + logits pre-extracted with chunked inference
- **Target:** Regional backbone accuracy (1-second window)
- **Results:** Val Brier 0.034, early stopped epoch 17/30
- **Interpretation:** Better Brier, but Ballroom is too easy — mean confidence 0.889, barely any low-conf regions. The backbone genuinely is reliable on Ballroom. Need harder data.
- **Decision:** Add GTZAN for genre diversity.

### Experiment 3: Confidence Head v3 — Ballroom + GTZAN
- **Date:** 2026-04-01
- **Config:** Same architecture, same hyperparams
- **Data:** 613 tracks (513 ballroom + 100 GTZAN mini), hidden states + logits extracted with **full forward pass** (no chunking)
- **Target:** Regional backbone accuracy
- **Results:** Val Brier 0.026 (best yet), early stopped epoch 22/30
- **Cross-dataset evaluation:**

| Metric | Ballroom (86 test) | GTZAN (100 all) |
|--------|-------------------|-----------------|
| Beat F1 | 0.886 | 0.839 |
| Correction effort | 2.0 | 3.84 |
| Mean confidence | 0.907 | 0.895 |
| Confidence std | 0.072 | 0.065 |

- **Interpretation:** GTZAN is genuinely harder (correction effort 2x, F1 lower), but the linear head barely differentiates (0.907 vs 0.895, std ~0.07). The head collapses to near-constant output because a single linear projection can't learn nonlinear decision boundaries in 512-dim hidden-state space.
- **Decision:** Upgrade to MLP head (512→64→1 with ReLU) for v4.

### Experiment 4: Confidence Head v4 — MLP Head
- **Date:** 2026-04-01 (pending)
- **Config:** MLP(512→64→1), ReLU+sigmoid, ~33K params
- **Data:** 613 tracks (ballroom + GTZAN mini)
- **Target:** Regional backbone accuracy
- **Hypothesis:** A single hidden layer with ReLU can learn nonlinear patterns like "high energy in hidden state dimensions X AND low energy in Y → low confidence"
- **Results:** PENDING

---

## Current Best

**V3** (as of 2026-04-01): Val Brier 0.026, but linear head produces insufficient variance for practical use. V4 (MLP) is the next iteration.

---

## Reproduction Instructions

```bash
# 1. Clone and install
git clone https://github.com/Rowdi-Labs/beat-grid-confidence.git
cd beat-grid-confidence
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Get annotations
git clone https://github.com/CPJKU/beat_this_annotations.git data/beat_this_annotations

# 3. Get spectrograms (need audio files)
python scripts/prepare_audio.py --audio-dir /path/to/audio --dataset-name ballroom --output-dir data

# 4. Extract hidden states (full forward pass, ~500MB peak)
python scripts/extract_hidden_states.py --spectrogram-dir data/spectrograms --output-dir data/hidden_states --datasets ballroom --chunk-frames 0

# 5. Train
python scripts/train.py --hidden-states-dir data/hidden_states --annotations-dir data/beat_this_annotations --datasets ballroom --output-dir outputs/v3

# 6. Evaluate
python scripts/evaluate.py --checkpoint outputs/v3/checkpoints/best.ckpt --hidden-states-dir data/hidden_states --annotations-dir data/beat_this_annotations --datasets ballroom
```
