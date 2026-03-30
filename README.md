# beat-grid-confidence

Confidence-aware beat grid prediction for music production and DJ software.

Extends [beat_this](https://github.com/CPJKU/beat_this) (JKU Linz, ISMIR 2024) with learned confidence estimation, tempo ambiguity tracking, and a structured decoder designed for real-world grid editing workflows.

## Problem

Beat trackers optimize for event accuracy (F-measure), but products need more:

- **Where is the grid likely wrong?** Standard models output a single best path with no uncertainty signal.
- **Is it half-time or double-time?** Ambiguous sections force users to manually re-grid.
- **How fast does the grid recover after a breakdown?** DJ tracks have long sparse sections where models drift silently.

This project adds lightweight prediction heads on top of a frozen `beat_this` backbone to answer these questions without retraining the base model.

## Approach

```
mel [1,T,128] --> beat_this backbone (frozen) --> hidden states [1,T,D]
                                                       |
                   +---------------+---------------+---+
                   |               |               |
             beat logits    downbeat logits    NEW HEADS:
             [1,T]          [1,T]             |
                                        +-----+-----+     +------+------+
                                        | Confidence |     | Tempo Dist  |
                                        | per-frame  |     | BPM bins    |
                                        | [1,T]      |     | [1,T,B]     |
                                        +------------+     +-------------+
```

The backbone is frozen (or LoRA-adapted). Only the heads train — roughly 40K parameters total. Training takes hours on a single GPU, not days on a cluster.

### What the heads predict

| Head | Output | Purpose |
|------|--------|---------|
| Confidence | Per-frame `[0,1]` | Flags unreliable regions for manual review |
| Tempo distribution | Per-frame softmax over BPM bins | Captures half-time / double-time ambiguity |

### Decoder

A confidence-aware structured decoder replaces standard peak-picking:

- **High-confidence regions:** Standard tempo-regularized decode
- **Low-confidence regions:** Maintains multiple hypotheses (half-time, double-time, alternate phase)
- **Transitions:** Selects best hypothesis by compatibility with surrounding high-confidence anchors

## Output format

```json
{
  "beats": [0.48, 0.96, 1.44, ...],
  "downbeats": [0.48, 1.92, 3.36, ...],
  "bpm": 125.0,
  "confidence_curve": [0.95, 0.94, 0.91, ...],
  "low_confidence_regions": [
    {"start": 32.1, "end": 40.8, "type": "breakdown"},
    {"start": 64.2, "end": 66.5, "type": "fill"}
  ],
  "alternate_hypotheses": [
    {"bpm": 125.0, "downbeat_offset": 0, "score": 0.87},
    {"bpm": 62.5, "downbeat_offset": 0, "score": 0.41}
  ]
}
```

## Evaluation

Standard MIR metrics (Beat F1, Downbeat F1, CMLt, AMLt) plus product-relevant metrics:

| Metric | What it measures |
|--------|-----------------|
| **Relock latency** | Frames to recover after breakdown/fill |
| **Correction effort** | Manual anchors needed to fix a track |
| **Continuity span** | Longest uninterrupted correct-beat run |
| **Confidence calibration** | Brier score of confidence vs actual error |
| **Hypothesis recall** | Is correct grid in top-K when top-1 is wrong? |

## Quick start

```bash
# Clone
git clone https://github.com/Rowdi-Labs/beat-grid-confidence.git
cd beat-grid-confidence

# Install
pip install -e ".[dev,experiment]"

# Train confidence head (Phase 1)
python scripts/train.py --config configs/confidence_only.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/best.ckpt

# Export to ONNX
python scripts/export_onnx.py --checkpoint outputs/best.ckpt --output model.onnx
```

## Data

See [`data/README.md`](data/README.md) for the full dataset inventory and license audit.

Training uses only commercially clean data:
- [beat_this annotations](https://github.com/CPJKU/beat_this_annotations) (MIT)
- [Groove MIDI](https://magenta.tensorflow.org/datasets/groove) (CC BY 4.0)
- Synthetic augmentations (silence injection, energy drops, tempo drift)

## Project status

This project is in active development (Phase 0). See the [roadmap](#roadmap) for current status.

## Roadmap

- [ ] **Phase 0** — Reproduce beat_this baselines, build evaluation harness
- [ ] **Phase 1** — Train confidence head, curate hard-case corpus
- [ ] **Phase 2** — Tempo distribution head, confidence-aware decoder, paper draft
- [ ] **Phase 3** — HuggingFace release, ONNX export, ISMIR submission

## Citation

If you use this work, please cite both this project and the original beat_this:

```bibtex
@software{rowdilabs2026beatgridconfidence,
  title={beat-grid-confidence: Confidence-Aware Beat Grid Prediction},
  author={Rowdi Labs},
  year={2026},
  url={https://github.com/Rowdi-Labs/beat-grid-confidence}
}

@inproceedings{beatthis2024,
  title={Beat This! Accurate Beat Tracking Without DBN Postprocessing},
  author={Foscarin, Francesco and Praher, Jan and Widmer, Gerhard},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2024}
}
```

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Rowdi Labs](https://github.com/Rowdi-Labs).
