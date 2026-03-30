# Dataset Inventory & License Audit

No audio is stored in this repo. This document tracks what datasets are used, their licenses, and how they may be used.

## Commercially Clean (OK for training)

| Dataset | License | Tracks | Content | Use |
|---------|---------|--------|---------|-----|
| [beat_this annotations](https://github.com/CPJKU/beat_this_annotations) | MIT | ~15K across 9 datasets | Beat/downbeat timestamps | Primary training annotations |
| [Groove MIDI](https://magenta.tensorflow.org/datasets/groove) | CC BY 4.0 | 1,150 | Drum MIDI performances | Rhythm pattern training |

## Evaluation Only (Non-commercial)

| Dataset | License | Tracks | Use |
|---------|---------|--------|-----|
| [Ballroom](https://mirdata.readthedocs.io/en/1.0.0/_modules/mirdata/datasets/ballroom.html) | CC BY-NC-SA 4.0 | 698 | Benchmark comparison only |
| [Hainsworth](https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/hainsworth.html) | CC BY-NC-SA 4.0 | 222 | Benchmark comparison only |

## Monitor for Future Use

| Dataset | Status | Notes |
|---------|--------|-------|
| [Osu2MIR](https://ismir2025program.ismir.net/lbd_421.html) | Promising | ISMIR 2025 LBD, needs license review |
| [FMA](https://github.com/mdeff/fma) | CC variants | Usable subset may exist, needs per-track audit |

## Synthetic Augmentations

Generated programmatically during training (no external data needed):

- **Silence injection**: Zero out random 1-4 second segments (simulates breakdowns)
- **Energy drop**: Attenuate segments by 20-40 dB (simulates sparse intros)
- **Spectral masking**: Zero random frequency bands (simulates missing instruments)
- **Tempo drift**: Gradual tempo ramp within a segment

## Hard Cases (Curated from E11even)

Target: 200-500 segments labeled by failure type:

| Failure Type | Description | Source |
|-------------|-------------|--------|
| `sparse_intro` | Minimal rhythmic content at track start | E11even beat audit |
| `breakdown` | Energy drop mid-track | E11even beat audit |
| `fill` | Drum fill disrupting regular pattern | E11even beat audit |
| `half_time_ambiguity` | Plausible at both tempi | Manual curation |
| `deceptive_downbeat` | Strong accent on non-downbeat | Manual curation |

## Rules

1. **Never commit audio files** to this repo
2. **Never train on NC-licensed data** — evaluation only
3. **Document every dataset** used in training with its license
4. **Verify provenance** before adding new datasets — see [E11even's research monitoring guide](https://github.com/Rowdi-Labs/E11even) for the full audit checklist
