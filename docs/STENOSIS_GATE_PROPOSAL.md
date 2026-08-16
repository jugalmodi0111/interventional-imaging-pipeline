# Stenosis acceptance gate — reframe proposal for clinical sign-off

**Status:** PROPOSED, unsigned · **Created:** 2026-08-16 · **Owner:** tech@manufex.io
**Decision needed from:** Dr. Reddy
**Evidence:** [`PHASE1_RESULTS.md`](../experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/PHASE1_RESULTS.md) · [`STAGE2_PHASE1_POA.md`](STAGE2_PHASE1_POA.md)

---

## 1. Where this stands

The question "is per-frame F1 the right gate?" has been open since 2026-07-17. It is now answered
with data: **no.** On the metric the product actually deploys — a screening flag that fires per
*study*, not per *frame* — the current weights score **per-video sensitivity 0.902** (92/102 held-out
lesion videos) while per-frame recall is 0.27.

Taken to Dr. Reddy 2026-08-16, two answers came back:

| Question | Answer as relayed | Status |
|---|---|---|
| Is per-study sensitivity ~0.90 with abstention the right acceptance criterion? | *"good to go but have to improve a bit"* | **Reframe ACCEPTED.** Bar is above 0.90; exact value not given. |
| What false-flag rate is tolerable? | *"should be very optimal and highly precise as real patient in the future will be used"* | **Direction given, no number.** |

**Neither answer is yet a floor.** This repo's convention is that an unquantified target is *not*
signed off — `configs/avf_fistulography.yaml` ships `sensitivity: null` for exactly this reason, and
the code treats null as "defer to human". So the reframe is accepted in principle and the gate stays
unsigned until §4 below is answered with numbers.

---

## 2. The measurement problem with "highly precise"

**Per-video precision (PPV) is not a property of the model alone — it depends on disease
prevalence.** The validation set is 102 lesion videos and 153 non-lesion videos, a prevalence of
~40%. That is an artifact of how CADICA was assembled, not a clinical rate. Quoting a PPV measured
at 40% prevalence and implying it will hold in a catheter lab running a different case mix would be
misleading.

The prevalence-independent quantities are **sensitivity** and **specificity**. Those are what the
model should be gated on. PPV then follows from whatever prevalence the deployment actually sees:

```
PPV = (sens × prev) / (sens × prev + (1 − spec) × (1 − prev))
```

So the right form of the question to Dr. Reddy is not "how precise?" but **"at what expected
prevalence, and what PPV do you need there?"** — from which the required specificity falls out.

## 3. What "highly precise" costs, in specificity

Holding the measured sensitivity at **0.902**:

**Specificity required to reach a target PPV**

| expected prevalence | PPV 0.70 | PPV 0.80 | PPV 0.90 | PPV 0.95 |
|---|---|---|---|---|
| 5% | 0.980 | 0.988 | 0.995 | 0.998 |
| 10% | 0.957 | 0.975 | 0.989 | 0.995 |
| 20% | 0.903 | 0.944 | 0.975 | 0.988 |
| 30% | 0.834 | 0.903 | 0.957 | 0.980 |
| 40% | 0.742 | 0.850 | 0.933 | 0.968 |

**PPV obtained at a given specificity**

| specificity | prev 5% | prev 10% | prev 20% | prev 30% | prev 40% |
|---|---|---|---|---|---|
| 0.80 | 0.192 | 0.334 | 0.530 | 0.659 | 0.750 |
| 0.90 | 0.322 | 0.501 | 0.693 | 0.794 | 0.857 |
| 0.95 | 0.487 | 0.667 | 0.819 | 0.885 | 0.923 |
| 0.98 | 0.704 | 0.834 | 0.919 | 0.951 | 0.968 |
| 0.99 | 0.826 | 0.909 | 0.958 | 0.975 | 0.984 |

**Read-out: "highly precise" is expensive.** At a plausible 10–20% prevalence, a PPV of 0.80 demands
specificity **0.944–0.975**. That is a demanding target for a detector whose per-frame precision is
0.23–0.31, and it is entirely possible the current weights cannot reach it at any operating point.

**This is not yet measured.** P1.1b scored only the 102 lesion videos; the 153 non-lesion videos it
had already run inference on were discarded, so the false-flag rate is unknown. The notebook cell was
rewritten 2026-08-16 to emit the full operating table (sensitivity / false-flag rate / specificity /
Youden J across conf 0.05–0.50) from a single inference pass. **That re-run is the blocker.**

There is a real possibility the answer is "these weights cannot be both 0.90+ sensitive and 0.95+
specific". If so, that is a legitimate finding and the honest response is to say so rather than to
quietly pick a threshold that flatters one axis.

---

## 4. Proposed gate — for signature

Replaces `configs/stenosis_yolo.yaml` `target: {f1: 0.57, recall: 0.60}` as the **primary** gate.

```yaml
target_per_video:
  sensitivity: 0.95        # PROPOSED — Dr. Reddy said "improve a bit" on the measured 0.902
  specificity: null        # BLOCKED  — needs §5 Q2 answered, then the §3 table sets it
  operating_conf: null     # chosen from the P1.1c sweep once specificity is signed off
  signed_off: false
```

Sensitivity 0.95 is proposed as a concrete reading of "improve a bit" from 0.902 — it is a
*proposal*, not something Dr. Reddy said. It needs confirming or replacing.

### 5. Three questions that close this out

1. **Sensitivity floor.** Is **0.95** per-video the bar, or is 0.90 acceptable given abstention
   (i.e. an uncertain study is referred rather than called normal)? A higher bar costs specificity.
2. **Expected prevalence.** What fraction of studies reaching this tool are expected to have a
   significant stenosis? Without this, "precise" cannot be converted into a specificity floor.
3. **PPV target at that prevalence.** Of the studies the tool flags, what fraction must be true
   positives for the flag to be clinically useful rather than noise?

Answering (2) and (3) sets the specificity floor directly from the §3 table. Answering (1) fixes the
other axis. Then `floor_ok` for stenosis becomes a real, evaluable question rather than a permanent
`false`.

### 6. Notes for whoever implements the signed gate

- `train_detector.qualifies_det` reads `cfg['target']['f1']` and **defaults to 0.57 when the key is
  absent**. Deleting `target.f1` does not retire the per-frame gate — it silently re-imposes it. The
  per-frame block must be explicitly demoted, not removed.
- No per-video gate function exists yet. The per-video numbers are produced by the notebook's
  P1.1b/c cell, not by `train()`, so wiring `target_per_video` into an automated gate is real work
  and should wait until the floors are signed.
- Do not gate on PPV directly. Gate on sensitivity + specificity; report PPV as a function of the
  agreed prevalence.
