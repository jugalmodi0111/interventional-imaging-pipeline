# Stenosis Phase 1 — diagnostic run on the baseline weights (P1.0 / P1.1a / P1.1b)

**Date:** 2026-08-16 · **Kernel:** `jugalmodi0111/stenosis-new` (Kaggle T4, Save & Run All, status COMPLETE)
**Weights scored:** `best.pt` from this experiment — **no retraining** (`SKIP_TRAIN=True`)
**Repo HEAD in the kernel:** `d7f9fb5` · ultralytics 8.4.120 · python 3.12.13
**Plan:** [`docs/STAGE2_PHASE1_POA.md`](../../docs/STAGE2_PHASE1_POA.md) §P1.0, §P1.1
**Raw artifacts:** [`phase1/`](phase1/)

Run was clean: no `[WARN]`, no traceback, all three steps produced output.

---

## Split actually scored

```
train 2609 imgs / 1377 groups | val 800 imgs / 229 groups (val ~14% by group)
danilov: 320 frames -> 64 patients,  0 ungrouped (0%)
cadica:  1589 frames -> 42 patients, 0 ungrouped (0%)
```

Both grouping tripwires reported **0 ungrouped**, so the split is honest. It is also **byte-identical
to the split the baseline was trained on** — verified before the run by replaying the 2026-07-13
`group_key`/`split_of` against today's over 1,798 stems: 0 changed. The val patients are patients
these weights never saw, so every number below is a genuine held-out measurement.

Two differences from the 2026-07-16 baseline run, both intentional and neither a leak:

- CADICA is now capped at 40 frames/patient (`0fb7390`, landed one day *after* that run), so the val
  set is a strict evenly-spaced subset — 800 val images here vs 1907 then. Noisier estimate, same
  patients.
- Aggregate F1 therefore reads **0.272** (best over the conf sweep) rather than 0.291. Not a
  regression; a smaller val set.

**Val composition is skewed** — this is the quota-split problem, measured:

| source | val / total | share of the val set |
|---|---|---|
| arcade | 207 / 1500 = 13.8% | 26% |
| danilov | 40 / 320 = 12.5% | 5% |
| **cadica** | **553 / 1589 = 34.8%** | **69%** |

CADICA is 47% of the corpus but **69% of val**, so the aggregate metric is effectively CADICA's
number. Independent per-patient hashing cannot hit the 15% target with only 42 patients (14 of 42
landed in val = 33%). This distorts the *estimate*; it does not leak.

---

## P1.0 — per-source val · **the diagnostic that aims everything**

```
arcade   n=  207  P 0.267  R 0.262  mAP50 0.136  mAP50-95 0.051
cadica   n=  553  P 0.375  R 0.244  mAP50 0.208  mAP50-95 0.078
danilov  n=   40  P 0.184  R 0.275  mAP50 0.092  mAP50-95 0.023
```

**Recall failure is uniform: 0.24–0.28 on every source.** That single fact retires three of the four
Phase-2 levers:

- **Not an ARCADE annotation-convention problem.** arcade R 0.262 ≈ cadica R 0.244.
- **Not Danilov dilution.** Danilov has the *highest* recall of the three (0.275). The "drop Danilov"
  lever (P2.1) cannot lift recall.
- **Not source imbalance.** Oversampling a minority source (P2.2 balance) cannot fix a deficit that
  is equally present in all sources.

What *does* vary is precision and localization — cadica mAP50 0.208 vs danilov 0.092 (2.3×),
precision 0.375 vs 0.184. That tracks the annotation-geometry mismatch measured the same run:

```
source      n_img  n_box  box/img   w_p50   h_p50  area_p50 tiny_frac
arcade       1293   2077   1.6063  0.1113  0.1060    0.0108    0.0563
cadica       1036   1453   1.4025  0.0820  0.0703    0.0058    0.1253
danilov       280    280   1.0000  0.0583  0.0491    0.0029    0.3607
```

Danilov boxes are ~half ARCADE's linear size, **3.7× smaller by area**, and 36% of them are "tiny"
vs ARCADE's 5.6%. Box harmonization (P2.1) is therefore worth something for **localization**
(mAP50-95 0.023 → ?), and nothing for recall.

**Conclusion: the model misses ~75% of lesions everywhere.** This is a data-scale / capability
limit, not a bad-source problem. No cheap re-weighting of the existing corpus fixes it.

## P1.1a — operating-point sweep

```
conf   P      R      F1     mAP50
0.05   0.252  0.289  0.269  0.155
0.10   0.229  0.336  0.272  0.142      <- best
0.15   0.265  0.272  0.268  0.127
0.20   0.285  0.230  0.255  0.116
0.25   0.314  0.204  0.248  0.107
```

F1 is flat across the whole sweep (0.248–0.272) — there is no knee to exploit, the PR curve is
simply low. Best operating point is **conf 0.10** (R 0.336). Recall only moves 0.204 → 0.336 across
the entire range, so operating-point selection is not a lever either.

*(Minor oddity: R at conf 0.05 (0.289) is below R at 0.10 (0.336). `b.mr` is read at the PR curve's
max-F1 point, which shifts as low-confidence boxes enter the curve; treat as sweep noise.)*

## P1.1b — per-video sensitivity · **the result that reframes the gate**

```
CADICA val videos: 255 (patients hashed to the val split)
lesion videos scored: 102
raw   per-video sensitivity: 92/102 = 0.902
voted per-video sensitivity: 85/102 = 0.833
```

**Per-video sensitivity 0.902 against per-frame recall 0.27.** Run over the FULL raw CADICA cine of
val patients (every input frame, not just annotated keyframes), scoring a hit if the lesion is
caught anywhere in the clip.

This is the evidence the gate-reframe proposal has been waiting on since 2026-07-17. A screening
flag — which is what the orchestrator actually deploys — fires per *study*, not per *frame*. On that
metric the same weights that look hopeless (F1 0.291 vs a 0.57 floor) catch the lesion in **90% of
lesion videos**. The per-frame floor has been measuring something the product does not do.

**Temporal voting HURT: 0.902 → 0.833 (−7 videos).** The notebook labelled this line
"(temporal-voting lift)" — it is not a lift. `min_hits=2` discards detections that appear in a single
frame, and on keyframe-annotated CADICA the true positive is frequently exactly one frame, so the
persistence filter deletes real detections. **Raw beats voted at these settings.** Label corrected in
the notebook 2026-08-16. If temporal voting is pursued, `min_hits=1` with IoU-linking for
gap-recovery only is the variant to test — but the honest baseline to beat is now raw 0.902.

---

## What this changes

1. **The clinical gate question is now answerable.** Take 0.902 per-video sensitivity to Dr. Reddy
   rather than F1 0.291 vs 0.57. Pending since 2026-07-17.
2. **Phase-2 levers are retired by evidence, not opinion.** P1.0 shows harmonize / balance /
   drop-Danilov cannot move recall. Do not spend GPU on them.
3. **The quota split is still worth landing** for a clean, comparable aggregate — but it will not
   clear 0.57 either, and that is no longer the interesting question.
