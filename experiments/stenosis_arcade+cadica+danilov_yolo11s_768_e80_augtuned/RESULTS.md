# Stenosis retrain — domain-tuned augmentation (P1.4) + the first per-video SPECIFICITY measurement

**Run tag:** `arcade+cadica+danilov_yolo11s_768_e80` · **Date:** 2026-08-16
**Kernel:** `jugalmodi0111/stenosis-new` (scriptVersionId 342775221, Kaggle T4, Save & Run All, COMPLETE)
**Config:** `configs/stenosis_yolo.yaml` — 80 epochs, patience 30, and the domain-tuned `augment:`
block, **verified applied** in the trainer args: `mosaic 0.0` (was 1.0), `scale 0.2` (was 0.5),
`erasing 0.0` (was 0.4), `box 9.0` (was 7.5), `dfl 2.0` (was 1.5), `cos_lr True` (was False).
**Split:** identical to the baseline's — `train 2609 / val 800`, 0 ungrouped on both tripwires. A
clean single-variable A/B: augmentation is the only thing that changed.
**Artifacts:** [`phase1/`](phase1/)

---

## Headline

**Per-frame the augmentation worked. Per-video it made the model worse, and the per-video metric is
the one the product uses.** More importantly, the first specificity measurement shows **neither
model has usable per-video discrimination** — the earlier "sensitivity 0.902" was largely an
artifact of a trigger-happy detector, not evidence of a working screening flag.

## Per-frame — the augmentation lever DID work

| metric | baseline (same val) | retrained | Δ |
|---|---|---|---|
| best F1 | 0.272 | **0.298** (ep 53) | **+0.026** |
| arcade R | 0.262 | **0.293** | +0.031 |
| cadica R | 0.244 | **0.304** | +0.060 |
| danilov R | 0.275 | **0.300** | +0.025 |
| arcade mAP50 | 0.136 | **0.188** | +0.052 |
| danilov mAP50 | 0.092 | **0.210** | +0.118 |
| danilov mAP50-95 | 0.023 | **0.056** | +0.033 |

Recall rose on **all three sources** and localization improved most where the annotation-geometry
mismatch was worst (Danilov mAP50 more than doubled — consistent with `box 9.0`/`dfl 2.0` helping
tiny boxes). Ran the full 80 epochs; best at ep 53, last ep 80 slightly lower (F1 0.280) — mild
over-fit tail, patience never fired.

**P1.4 verdict: the domain-tuned augmentation is a real per-frame improvement.** It is also the
only Phase-1/2 lever that has produced one.

## Per-video — the measurement that matters, and it is bad

First run to measure **specificity**. Truth from CADICA's own `lesionVideos.txt` manifests (not
inferred from groundtruth presence). 102 lesion / 153 non-lesion held-out videos.

**Retrained**

```
conf   sensitivity        false-flag rate    specificity   Youden J
0.05   0.961 ( 98/102)    1.000 (153/153)     0.000         -0.039
0.10   0.912 ( 93/102)    1.000 (153/153)     0.000         -0.088
0.15   0.882 ( 90/102)    1.000 (153/153)     0.000         -0.118
0.25   0.843 ( 86/102)    0.974 (149/153)     0.026         -0.131
0.50   0.667 ( 68/102)    0.752 (115/153)     0.248         -0.085
```

**Baseline**

```
0.05   0.931 ( 95/102)    0.980 (150/153)     0.020         -0.049
0.15   0.882 ( 90/102)    0.902 (138/153)     0.098         -0.020
0.35   0.745 ( 76/102)    0.693 (106/153)     0.307         +0.052
0.50   0.608 ( 62/102)    0.444 ( 68/153)     0.556         +0.163
```

**Youden J is NEGATIVE at every threshold for the retrained model.** J = sensitivity − false-flag
rate; J < 0 means the model flags negatives *more readily than positives* — worse than chance as a
per-video discriminator. The baseline is barely positive, and only at conf ≥ 0.20 where sensitivity
has already collapsed to 0.85 → 0.61.

**A/B: the retrain traded a lot of specificity for almost no sensitivity.**

```
conf   d-sens   d-fpr
0.05   +0.029   +0.020
0.15   +0.000   +0.098
0.35   +0.039   +0.209
0.50   +0.059   +0.307
```

`d-fpr` is positive at **every** threshold. The tuned model is simply more trigger-happy: it fires
more often on everything, which nudges sensitivity up and wrecks specificity.

### PPV — worse than flagging everything

| operating point | sens | spec | PPV @5% | @10% | @20% | @30% | @40% |
|---|---|---|---|---|---|---|---|
| retrained @0.50 | 0.667 | 0.248 | 0.045 | 0.090 | 0.181 | 0.275 | 0.372 |
| baseline @0.50 | 0.608 | 0.556 | 0.067 | 0.132 | 0.255 | 0.370 | 0.477 |
| *base rate (flag everything)* | — | — | *0.050* | *0.100* | *0.200* | *0.300* | *0.400* |

**The retrained model's PPV is below the base rate at every prevalence** — it is worse than a rule
that flags every study. The baseline's best point clears the base rate only marginally (0.255 vs
0.200 at 20% prevalence) and costs sensitivity down to 0.608.

Against Dr. Reddy's "highly precise" requirement, both models fail comprehensively.

## Why — the mechanism, and why 0.902 was misleading

The decision rule is "**flag the clip if the detector fires anywhere in it**". A CADICA clip is
100–300 frames. At per-frame precision ≈ 0.27, the chance of *at least one* false positive somewhere
in a clip approaches 1:

```
P(no false flag) ≈ (1 − p_fp_per_frame) ^ n_frames  →  ~0
```

**Both halves of the earlier result come from the same trigger-happiness.** Sensitivity 0.902 looked
strong for exactly the reason specificity is ~0: the detector fires almost everywhere. Reporting
0.902 without specificity overstated the model — that is precisely the failure the specificity
measurement was added to catch, and it caught it.

## What this changes

1. **The per-study reframe is still conceptually right** — a screening flag fires per study. What is
   wrong is the *aggregation rule*, not the framing.
2. **"Fires anywhere in the clip" is a bad rule.** It is maximally amplifying of per-frame false
   positives — the worst possible choice for a detector with precision ~0.27.
3. **The next lever is the per-video decision rule, not the detector.** Candidates, all testable
   offline from a single inference pass if richer per-video statistics are stored:
   - **k-th highest confidence** instead of the max (requires k frames to agree)
   - **count/fraction of frames** with a detection above threshold
   - **spatial consistency** — detections recurring in the same image region across frames
   - IoU-linked persistence at `min_hits=1` (gap recovery only; `min_hits=2` regressed again this
     run: −0.069 retrained, −0.059 baseline)
   The current run stored only per-video **max** confidence, so none of these can be evaluated
   retrospectively. Storing the top-k confidences and per-frame detection counts is the cheap fix.
4. **Dr. Reddy was given 0.902 before specificity existed.** That number needs correcting with him:
   the reframe stands, the model does not yet support it.
