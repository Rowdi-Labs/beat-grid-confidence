# Paper Gaps Checklist

Analysis of what's missing from `paper/main.tex` before it's submission-ready.

## Must fix before submission

- [ ] **Table 1: fill continuity span** for both decoders (data exists in eval, just not in the table)
- [ ] **Relock latency: report or acknowledge** — defined as a contribution in abstract + Section 3.3 but never reported. All evals returned `inf` because datasets lack enough low-activation regions. Either measure on harder data or explicitly note this limitation
- [ ] **"Seableton" → "Ableton"** typo in intro (line 50)
- [ ] **Fix wrong citations:**
  - Holzapfel [8] is "Beat Tracking Using Group Delay Based Onset Detection" — not about confidence measures. Find the actual confidence paper or remove the claim
  - PLP [5] cites Grosche & Müller 2011, but the text references a "2024 real-time PLP tracker." Either cite the correct 2024 paper or fix the text
- [ ] **Use actual ISMIR LaTeX template** (currently generic `article` class)
- [ ] **Add statistical significance tests** — paired Wilcoxon signed-rank on decoder comparison (F1 and correction effort). Report p-values

## Should fix for credibility

- [ ] **More datasets** — 785 tracks across 2 datasets is thin for ISMIR. Add at least Harmonix (breakdowns), full GTZAN (999 tracks), HJDB. Target 3-5 datasets
- [ ] **DBN decoder comparison** — paper argues decoders matter but never tests the dominant paradigm. Use madmom's DBN or beat_this's own optional DBN mode
- [ ] **Validate correction effort metric** — claim that "1 incorrect region ≈ 1 manual anchor" is unvalidated. Even an informal user study (5 DJs correcting 20 tracks) would help
- [ ] **Error type breakdown** — are failures at breakdowns? Tempo changes? Genre-specific? This matters for the "product-oriented" framing
- [ ] **F1/effort divergence analysis** — the cases where F1 and effort disagree ARE the paper's argument. Show specific examples and explain why they diverge
- [ ] **Discuss downbeat errors** — Downbeat F1 is reported in Table 2 but never discussed. Bar-level errors affect loops, cue points, phrase alignment — arguably more impactful than beat errors for DJ use

## Nice to have

- [ ] **Reframe confidence section** — the negative result ("trained head ≈ heuristic") could be a positive finding: "beat tracking models already encode confidence in their activation strength, and this signal should be surfaced in product UIs"
- [ ] **Support "90% Problem" with data** — "tracks with F1 > 0.90 still require a mean of X correction anchors" (compute from our eval data)
- [ ] **Future work section** — currently just a limitations paragraph. Could discuss: better decoders (Viterbi, beam search), harder datasets, integration with source separation, online/streaming decoding
- [ ] **Per-genre analysis on GTZAN** — metal and jazz are much harder than pop/rock. Show this breakdown
