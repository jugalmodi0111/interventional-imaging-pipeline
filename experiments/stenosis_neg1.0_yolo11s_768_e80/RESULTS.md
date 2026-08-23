# Negative-sampling sweep — conclusion (audit A1)

**Date:** 2026-08-23 · **Runs:** Kaggle `jugalmodi0111/stenosis-new` v343513735 (0.25) and v344221183 (1.0)
**Question:** the A1 fix added background frames to a corpus that had none. How many?
**Answer:** background ratio is a real, monotonic lever — **and it is not enough to make this model usable.**

Raw artifacts: [`phase1/`](phase1/) · run 1: [`../stenosis_neg0.25_yolo11s_768_e80/`](../stenosis_neg0.25_yolo11s_768_e80/)

---

## The sweep

| corpus | `negatives_per_positive` | background frac | train imgs |
|---|---|---|---|
| baseline | 0 (pre-fix) | **0%** | 2609 |
| run 1 | 0.25 | **10.5%** | 2856 |
| run 2 | 1.0 | **31.9%** | 3566 |

All three scored on the identical per-video metric (raw CADICA cine of the 14 val patients, 102 lesion
/ 153 non-lesion videos). That metric is corpus-independent, so the three are directly comparable
even though each built a different training corpus.

## Result 1 — the dose works, monotonically

Compared at **matched sensitivity 0.882** (matched `conf` would compare different operating points):

| corpus | false-flag rate @ sens 0.882 |
|---|---|
| baseline | 0.902 |
| 0.25 | 0.850 |
| **1.0** | **0.745** |

Youden J is monotonic in the dose at every threshold:

| conf | baseline | 0.25 | 1.0 |
|---|---|---|---|
| 0.05 | −0.049 | +0.007 | **+0.137** |
| 0.10 | −0.052 | +0.033 | **+0.212** |
| 0.20 | +0.003 | +0.039 | **+0.242** |

**The baseline was worse than chance** (negative J) at the recall-first operating points. A1 fixed
that. This is a genuine, reproducible improvement and the audit finding was correct.

## Result 2 — the sensitivity CEILING erodes, and that decides it

| corpus | max achievable sensitivity | can it reach 0.90? | false-flag there |
|---|---|---|---|
| baseline | 0.931 | yes | 0.980 |
| **0.25** | **0.902** | **yes (just)** | **0.895** |
| 1.0 | 0.882 | **NO — at any threshold** | — |

Dr. Reddy's bar is *above* 0.902 ("good to go but have to improve a bit"); the proposal reads that as
0.95. **At `negatives_per_positive: 1.0` the model cannot reach 0.90 sensitivity at any confidence
threshold**, so it fails the sensitivity floor before specificity is even discussed.

That is why **`negatives_per_positive` is set back to 0.25** as the shipped default: it is the only
swept value that both improves discrimination AND retains a sensitivity ceiling at the floor. 1.0
buys better separation by forfeiting the axis that is clinically non-negotiable.

## Result 3 — where that leaves the actual product

At the sensitivity the gate requires, the best corpus (0.25) delivers **specificity 0.105**. The gate
needs ~0.95. PPV at the achievable operating points, for the strongest model (1.0):

| operating point | sens | spec | PPV @20% prev | PPV @10% |
|---|---|---|---|---|
| conf 0.05 | 0.882 | 0.255 | 0.228 | 0.116 |
| conf 0.20 | 0.706 | 0.536 | 0.276 | 0.145 |
| conf 0.35 | 0.412 | 0.817 | 0.360 | 0.200 |
| conf 0.50 | 0.147 | 0.948 | 0.414 | 0.239 |

Base rate is 0.200 / 0.100. **At usable sensitivity the flag is worth ~2.8 points over guessing.** The
one row where specificity finally reaches 0.948 costs 85% of the lesions.

## Result 4 — 1.0 overshot on per-frame, and the two metrics disagree

| | baseline | **0.25** | 1.0 |
|---|---|---|---|
| per-frame F1 | ~0.272 | **0.288** | 0.252 |
| CADICA per-frame recall | 0.244 | **0.309** | 0.249 |

0.25 is the per-frame optimum; 1.0 is the per-video-discrimination optimum. They disagree, which is
itself the finding: **32% background suppresses genuine detections, not only spurious ones.** Pushing
to 2.0 would erode the sensitivity ceiling further in the direction that already fails the gate.
**Do not run 2.0.**

---

## Conclusion

**A1 was necessary and is now closed. It is not sufficient, and the remaining gap is not a
data-balance problem.**

The sweep bought ~16 points of false-flag at matched sensitivity and turned a worse-than-chance
detector into a discriminative one. It did not move the model within reach of a clinical screen, and
the dose-response shows why more of the same will not: the two axes trade against each other, and the
sensitivity floor binds first.

What is left is detector capability and patient diversity, not corpus composition:

1. **More patients.** P1.0 already said diversity is the bottleneck and CADICA's 42 patients is thin.
   The AngioCAD proxy (413 patients, ~10x CADICA) is the obvious next lever.
2. **Stronger detector / angiography pretraining.** The SSL backbone path is now safe to use — A2 and
   A2b closed the zero-negative defects that would have poisoned every pseudo-label round.
3. **Reformulate.** Per-frame detection may be the wrong task shape for a per-study screen; a
   study-level classifier over the whole cine is a different, possibly better-posed problem.

None of these is a tuning change. All three are multi-day work, and (1) needs a dataset download.
That is the honest state of the stenosis track.
