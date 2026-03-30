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
- Trainable (our heads only): 72,846 (0.36%)
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

# MPS (Apple Silicon) is available for GPU acceleration
# Use device="mps" for faster inference on M1/M2/M3
```

## Project Structure

```
src/beat_grid_confidence/
├── model.py        # BeatGridConfidenceModel: frozen backbone + trainable heads
├── heads.py        # ConfidenceHead (512→1, sigmoid), TempoDistributionHead (512→141, softmax)
├── dataset.py      # Loads .beats annotations, pre-extracted spectrograms, 8-fold splits
├── losses.py       # BCE for confidence, KL-div with Gaussian smoothing for tempo
├── evaluation.py   # Standard MIR metrics (mir_eval) + product metrics
└── decode.py       # Confidence-aware decoder with alternate hypothesis tracking

scripts/
├── train.py        # PyTorch Lightning training with frozen backbone
├── evaluate.py     # Full evaluation pipeline with Rich output
└── export_onnx.py  # ONNX export for browser/server deployment (Phase 3)

configs/
├── base.yaml              # Full model config (all heads)
└── confidence_only.yaml   # Phase 1 config (confidence head only)

data/
└── README.md       # Dataset inventory and license audit
```

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

**Never train on NC-licensed data.** This is a hard rule. The whole point of this project's licensing posture is to produce commercially usable weights. See `data/README.md` for the full audit.

## Confidence Head Training

- **Supervision signal:** Binary correctness mask — 1 if nearest ground-truth beat is within 50ms (2.5 frames at 50 FPS), 0 otherwise
- **Synthetic augmentation:** Inject silence/energy drops into clean tracks, label corrupted regions as low-confidence
- **Loss:** Binary cross-entropy
- **Key metric:** Brier score (MSE between predicted confidence and actual correctness)

## Product Metrics (Novel Contribution)

These are what differentiate this project from standard beat tracking research:

| Metric | Implementation | Why it matters |
|--------|---------------|----------------|
| Relock latency | `evaluation.py:compute_relock_latency` | DJ needs instant lock after a breakdown |
| Correction effort | `evaluation.py:compute_correction_effort` | User cost proxy — fewer manual anchors = better |
| Continuity span | `evaluation.py:compute_continuity_span` | Longest uninterrupted correct-beat run |
| Confidence calibration | `evaluation.py:compute_confidence_brier` | Confidence must be trustworthy for UI flagging |

## Project Phases and Gates

**Phase 0 (current):** Reproduce baseline, build evaluation harness
**Phase 1:** Train confidence head, curate hard cases
**Gate 1:** Brier < 0.15, confidence correlates with error (r > 0.5)
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
