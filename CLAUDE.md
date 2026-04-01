# beat-grid-confidence — Agent Instructions

## What This Project Is

An open-source research project from Rowdi Labs extending [beat_this](https://github.com/CPJKU/beat_this) (JKU Linz, MIT license) with confidence estimation and tempo ambiguity tracking. This is Rowdi Labs' **first public repo** — it represents the org publicly and must be polished.


## beat_this Architecture (Verified)

These facts were verified by loading the actual `final0` checkpoint on an M1 Mac:

```
Input: [B, T, 128] (time-first, 128 mel bins)
       ↓
frontend (Sequential: Rearrange→BN1d→Conv2d→partial transformers→Linear(1024→512))
       ↓
transformer_blocks (6× [Attention, FeedForward] + RMSNorm, rotary embeddings)
       ↓
Hidden states: [B, T, 512]    ← OUR HOOK POINT (split forward pass here)
       ↓
task_heads (SumHead: Linear(512→2) → {"beat": [B,T], "downbeat": [B,T]})
```

**Key numbers:**
- Total params: 20,324,558
- Hidden dim: 512
- Frame rate: 50 FPS (22050 Hz / 441 hop)
- Named checkpoints: `"final0"` (77MB, server), `"small0"` (10.5MB, browser)

**Gotchas discovered:**
- Input is `[B, T, 128]` NOT `[B, 128, T]` — the frontend's first layer is `Rearrange('b t f -> b f t')`
- RMSNorm uses `.scale` and `.gamma`, NOT `.weight` — don't use norm attributes for dim detection
- Use `model.task_heads.beat_downbeat_lin.in_features` to get hidden dim reliably
- `load_model()` auto-downloads checkpoints to `~/.cache/torch/hub/checkpoints/`
- No forward hooks needed — just split: `frontend(x) → transformer_blocks(x) → task_heads(x)`

## Annotations Format (beat_this_annotations)

Repo: https://github.com/CPJKU/beat_this_annotations

- `.beats` files: tab-separated `time\tbeat_position` (position 1 = downbeat)
- `info.json` per dataset: `{"has_downbeats": true/false}`
- `8-folds.split`: tab-separated `track_id\tfold_number` (0-7)
- Standard protocol: fold N = val, fold (N+1)%8 = test, rest = train
- Spectrograms: pre-extracted as `.npy` or `.npz`, 50 FPS, 128 mel bins

## Development Environment

```bash
# Python 3.12 (3.14 is too new for PyTorch)
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# MPS (Apple Silicon) is available but NOT used for training
# The confidence head is 513-33K params — CPU is faster than MPS overhead
```

## Project Structure

```
src/beat_grid_confidence/
├── model.py        # BeatGridConfidenceModel: frozen backbone + trainable heads
├── heads.py        # ConfidenceHead, TempoDistributionHead
├── dataset.py      # BeatGridConfidenceDataset, HiddenStatesDataset, annotation loading
├── losses.py       # BCE for confidence, KL-div with Gaussian smoothing for tempo
├── evaluation.py   # Standard MIR metrics (mir_eval) + product metrics
└── decode.py       # Confidence-aware decoder with alternate hypothesis tracking

scripts/
├── train.py              # PyTorch Lightning training (head-only, no backbone)
├── evaluate.py           # Full eval pipeline with Rich output
├── extract_hidden_states.py  # Pre-extract hidden states + logits from backbone
├── prepare_audio.py      # Audio → spectrogram → hidden states → logits (one-stop)
└── export_onnx.py        # ONNX export for browser/server deployment (Phase 3)

configs/
├── base.yaml              # Full model config (all heads)
└── confidence_only.yaml   # Phase 1 config (confidence head only)

data/
├── README.md              # Dataset inventory and license audit
├── spectrograms/          # Pre-extracted mel spectrograms (gitignored)
├── hidden_states/         # Pre-extracted hidden states + logits (gitignored)
├── beat_this_annotations/ # Cloned annotations repo
└── mirdata_cache/         # Downloaded audio datasets (gitignored)
```

## Data Pipeline

The pipeline separates backbone inference from head training to avoid OOM:

```
1. Spectrograms (one-time):
   prepare_audio.py --audio-dir ... --dataset-name ...
   OR: use pre-extracted spectrograms from beat_this

2. Hidden states + logits (one-time, full forward pass):
   extract_hidden_states.py --spectrogram-dir data/spectrograms --output-dir data/hidden_states --chunk-frames 0

3. Head training (fast, CPU, ~95 MB):
   train.py --hidden-states-dir data/hidden_states --annotations-dir data/beat_this_annotations

4. Evaluation (no backbone needed):
   evaluate.py --checkpoint ... --hidden-states-dir data/hidden_states --annotations-dir ...
```

**Memory budget:**
| Step | Peak memory | Why |
|------|------------|-----|
| Extraction (full pass, 1 track) | ~500 MB | Attention O(T^2) for ~1500 frames |
| Extraction (chunked, 256 frames) | ~84 MB | Smaller attention, but stitching artifacts |
| Head training (batch=32) | ~95 MB | Just loading [B,T,512] hidden states |
| Evaluation | ~50 MB | Loading numpy + tiny head forward pass |

**IMPORTANT:** Always use `--chunk-frames 0` (full forward pass) for extraction. Chunked extraction introduces ~6% F1 drop from overlap blending artifacts.

## Data Licensing — STRICT RULES

**Commercially clean (OK for training):**
- beat_this annotations (MIT)
- Groove MIDI (CC BY 4.0)
- Synthetic augmentations we generate
- Internal product beat audit labels

**Evaluation only (NON-COMMERCIAL):**
- Ballroom (CC BY-NC-SA 4.0)
- Hainsworth (CC BY-NC-SA 4.0)
- MIREX organizer-provided datasets

**Never train on NC-licensed data.** This is a hard rule. See `data/README.md`.

## Confidence Head — What Works and What Doesn't

### V1: Binary correctness target (WRONG)
- Target: "is there a GT beat within 50ms of this frame?"
- Problem: just learns the base rate (~24% of frames near a beat at 120 BPM)
- Mean confidence: 0.288 — learned "predict beat proximity" not quality

### V2: Regional backbone accuracy target (CORRECT SIGNAL)
- Target: in a 1-second window, what fraction of backbone's predicted beats match GT?
- Brier: 0.034 (good), but Ballroom is too easy for differentiation

### V3: Ballroom + GTZAN combined (CURRENT)
- 613 training tracks (513 ballroom + 100 GTZAN mini)
- Brier: 0.026 (best yet)
- BUT: linear head (512→1) collapses to near-constant output
  - Ballroom mean confidence: 0.907, std: 0.072
  - GTZAN mean confidence: 0.895, std: 0.065
  - Not enough variance to flag hard regions

### V4 (NEXT): MLP head
- Architecture: 512 → 64 → 1 (with ReLU), ~33K params
- Hypothesis: linear head can't represent nonlinear decision boundaries
- Still CPU-trainable, still fast

## Product Metrics (Novel Contribution)

| Metric | Implementation | Why it matters |
|--------|---------------|----------------|
| Relock latency | `evaluation.py:compute_relock_latency` | DJ needs instant lock after breakdown |
| Correction effort | `evaluation.py:compute_correction_effort` | User cost proxy — fewer anchors = better |
| Continuity span | `evaluation.py:compute_continuity_span` | Longest uninterrupted correct-beat run |
| Confidence calibration | `evaluation.py:compute_confidence_brier` | Confidence must be trustworthy for UI |

## Baseline Results (Full Forward Pass)

| Metric | Ballroom (86 tracks) | GTZAN (100 tracks) |
|--------|---------------------|-------------------|
| Beat F1 | 0.899 | 0.839 |
| AMLt | 0.928 | 0.919 |
| CMLt | 0.791 | 0.770 |
| Downbeat F1 | 0.672 | 0.650 |
| Continuity span | 23.8s | 18.1s |
| Correction effort | 2.0 | 3.84 |

## Project Phases and Gates

**Phase 0 (COMPLETE):** Baseline reproduced on Ballroom (Beat F1: 0.899)
**Phase 1 (IN PROGRESS):** Confidence head training — v1-v3 done, v4 (MLP) next
**Gate 1:** Confidence std > 0.15 on GTZAN, Brier < 0.15, meaningful low-conf regions
**Phase 2:** Tempo distribution head, confidence-aware decoder, paper
**Gate 2:** Product metrics improve >10%, standard metrics don't regress
**Phase 3:** HuggingFace release, ONNX export, product integration, ISMIR submission

**Kill criteria:** Gate 1 fails + LoRA also fails, or $3K spent with no publishable result.

## Collaboration Context

- **JKU Linz (CPJKU):** Made beat_this. Potential co-authors. Reach out after baseline reproduction, not before.
- **Publication target:** ISMIR 2026 (full paper) or ISMIR LBD / DMRN (workshop)
- **Grant applications:** Google TPU Research Cloud, HuggingFace community grants. Drafts in `.grants/` (gitignored).

## Style and Conventions

- Python 3.11+ type hints throughout
- Ruff for linting (`ruff check`)
- pytest for testing
- PyTorch Lightning for training
- YAML configs for hyperparameters
- Rich for CLI output formatting
- All tensor shapes documented in docstrings as `[B, T, D]` notation
