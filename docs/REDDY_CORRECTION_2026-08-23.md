# Correction to Dr. Reddy — the stenosis sensitivity figure

**Date:** 2026-08-23 · **Owner:** tech@manufex.io
**Supersedes:** the 0.902 per-study sensitivity presented on 2026-08-16
**Backing:** [`stenosis_neg1.0_yolo11s_768_e80/RESULTS.md`](../experiments/stenosis_neg1.0_yolo11s_768_e80/RESULTS.md) · [`STENOSIS_GATE_PROPOSAL.md`](STENOSIS_GATE_PROPOSAL.md)

---

## What to say

> On 16 August I brought you a per-study sensitivity of **0.902** and asked whether that was the right
> acceptance criterion for a screening flag. That number was incomplete in a way that materially
> overstated the model, and I want to correct it before it informs any decision.
>
> **We had measured sensitivity without measuring the false-alarm rate.** When we measured the other
> half, the same model flagged **98% of lesion-free studies**. A detector that flags almost everything
> will of course catch 90% of the disease — the sensitivity was largely an artifact of that, not
> evidence of discrimination. On the standard balance measure it scored *worse than chance*.
>
> The cause was a defect on our side, now fixed: the training set contained **no normal frames at
> all** — every image we trained on contained a lesion, so the model was never shown what "no
> stenosis" looks like and learned to always fire.
>
> After fixing that and tuning it, the honest position today is:
>
> | | sensitivity | flags this share of NORMAL studies |
> |---|---|---|
> | what I showed you on 16 Aug | 0.90 | **98%** |
> | best we have now | 0.90 | **90%** |
> | if we accept much lower sensitivity | 0.41 | 18% |
>
> At a case mix of roughly 1 in 5 studies having a significant stenosis, a flag from the current model
> is correct about **23% of the time** — against 20% if you flagged studies at random. It is not yet
> a useful screen, and I do not want to present it as one.
>
> **The reframe you approved still holds** — per-study sensitivity with abstention is the right way to
> judge this, and per-frame F1 was the wrong measure. What has changed is only my claim about how
> close we are to meeting it. We are not close.
>
> The remaining gap is not something we can tune away. It needs more patients (our current set is 42),
> a stronger model, or a different formulation. Those are weeks, not days, and I would rather tell you
> that now than after another round of tuning.

## The three questions still open

Unchanged from the proposal, and now more consequential — the sweep showed the two axes trade
directly against each other, so setting one floor constrains the other:

1. **Sensitivity floor.** Is 0.95 per-study the bar, or is 0.90 acceptable given abstention?
   *New evidence:* pushing specificity up erodes the sensitivity ceiling. At our most aggressive
   setting the model **cannot reach 0.90 at all**. If the floor is 0.95, no configuration we have
   measured qualifies — which is a legitimate answer to record.
2. **Expected prevalence.** What share of studies reaching this tool would have a significant
   stenosis? Without this, "precise" cannot be turned into a number.
3. **Target PPV at that prevalence.** Of the studies the tool flags, what share must be true positives
   for the flag to be worth acting on?

## Why this is being raised rather than quietly re-measured

The 0.902 was already used to answer a question ("is per-study sensitivity ~0.90 the right
criterion?") and got an approving answer. If the number behind that approval is wrong, the approval
was given on a false premise, even though the *criterion* it endorsed is still correct. Correcting it
costs one conversation; not correcting it risks a clinical sign-off resting on a model that flags 9
out of 10 normal studies.
