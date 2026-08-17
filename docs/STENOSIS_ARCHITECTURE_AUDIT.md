# Stenosis pipeline — architecture audit

**Date:** 2026-08-16 · **Trigger:** per-video specificity measured at 0.00–0.25 with negative Youden J
([retrain RESULTS](../experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e80_augtuned/RESULTS.md))
**Scope:** data construction → training → evaluation → serve, for the stenosis detector only.

---

## A1 — CRITICAL: the training set contains **zero negative frames**

**Every image in the stenosis dataset has at least one stenosis box.** The model has never once been
shown what "no stenosis here" looks like.

**Evidence (two independent sources, both from the live runs):**

1. Ultralytics reports `0 backgrounds` on **every** scan, train and val, across both runs — 2609
   train / 800 val images, zero label-less images. "Background" is ultralytics' term for an image
   with an empty or missing label file, i.e. a negative example.
2. The annotation QA table shows `box/img` = **1.6063** (arcade), **1.4025** (cadica), **1.0000**
   (danilov). All ≥ 1.0 — no image has zero boxes.

**Cause** — `src/data_prep/cadica_to_yolo.py::_iter_frames`:

```python
gt_dir = _sibling_dir(video_dir, "groundtruth")
if not gt_dir:
    continue          # discards ENTIRE non-lesion videos (they ship no groundtruth dir)
...
gp = os.path.join(gt_dir, stem + ".txt")
if not os.path.isfile(gp):
    continue          # discards every non-keyframe
```

CADICA ships groundtruth directories **only for lesion videos**. So every non-lesion video is
dropped whole, before a single frame is considered.

The other two sources are not converter bugs — they are dataset properties. `io_utils.coco_to_yolo`
iterates *all* COCO images and writes an empty label file for an unannotated one, so it *would*
preserve negatives; ARCADE task-2 simply annotates a stenosis in every image. Danilov ships one XML
per frame, each with exactly one object. **CADICA is the only source that actually holds negatives,
and the converter throws them away.**

**Why this explains the results.** A single-class detector trained exclusively on positives has no
gradient signal teaching it to *not* fire. It learns "always propose a box". That produces, in order:

- per-frame precision ~0.27 — it fires constantly, and since every val image *does* contain a
  lesion, a useful fraction of those fires happen to land
- per-video false-flag rate ≈ 1.0 — on a non-lesion clip it fires just as constantly with nothing to
  hit, and the "flag if it fires anywhere in 100–300 frames" rule converts that into a certainty
- negative Youden J — no discriminative capacity at all
- "recall fails uniformly across all three sources" (P1.0) — consistent: this is not a per-source
  annotation problem, it is a model with no concept of a negative

Ultralytics' own guidance is ~0–10% background images to suppress false positives. This dataset has
**0%**.

**Scale of the unused data.** CADICA is 668 videos / 42 patients. The val split alone (14 patients)
held 102 lesion and **153 non-lesion** videos — ~40% lesion. Extrapolating, roughly **400 non-lesion
videos** exist across the corpus, at ~100–300 frames each: on the order of **40,000–120,000 negative
frames, none of which have ever been used.**

### A1a — correctness constraint on the fix

**Only frames from NON-LESION videos are valid negatives.** An unannotated frame inside a *lesion*
video is not a negative — the lesion is physically present, CADICA just annotates keyframes only.
Feeding those in as background would actively teach the model to suppress true positives, i.e. it
would destroy recall. Any negative-sampling implementation must partition on the video's
lesion/non-lesion label (CADICA's `lesionVideos.txt` / `nonlesionVideos.txt` manifests, which the
P1.1b/c evaluation cell already reads), never on per-frame annotation presence.

---

## A2 — HIGH: the SSL pseudo-label round has the identical bug

`src/train/train_detector.py::_pseudo_label_round`:

```python
b = model.predict(frame, conf=conf, ...)[0].boxes
if b is None or len(b) == 0:
    continue          # frames where the detector found NOTHING are discarded
```

A frame the detector fires on becomes a positive training example; a frame it does not fire on is
thrown away rather than becoming the background example it should be. If SSL were enabled it would
**compound** A1 — every round would reinforce "there is always a lesion". Currently inert (no
disjoint `ssl.unlabeled_dir` attached, guard verified firing in both runs), so this is latent, not
active.

## A2b — HIGH: a THIRD instance of the same defect, in the Grounding-DINO seed round

Found while implementing the A2 fix. `_gdino_seed_round` carried the identical line:

```python
lines = boxes_labels_to_yolo_lines(boxes, labels, class_map, W, H)
if not lines:
    continue          # GD found nothing -> frame discarded instead of becoming a background
```

So the open-vocab cold start re-created exactly the all-positive corpus that A1/A2 remove. Three
independent code paths (`cadica_to_yolo._iter_frames`, `_pseudo_label_round`, `_gdino_seed_round`)
each independently encoded the same wrong assumption: *a frame is only worth keeping if something
was found in it*. That is the actual root pattern behind A1 — not a single typo, a shared mental
model. Fixing two of three would have let the corpus silently regress the moment the seed path ran.

**Fixed 2026-08-16** with the same shape as A2: zero-detection frames become backgrounds (image +
empty label), sharing `ssl.max_background_frac` so a cold start that fires on almost nothing cannot
flood the train split. Frame order sorted for determinism. 4 tests.

## A3 — MEDIUM: the SSL round retrains from COCO weights, not the current model

Same function: `model = YOLO(_detector(cfg)["name"] + ".pt")` re-initialises from the stock
pretrained checkpoint rather than continuing from `weights`. Defensible as confirmation-bias
avoidance, but it is undocumented and means the SSL round discards everything the base run learned
except what survives in the pseudo-labels.

## A4 — the per-video decision rule (already recorded)

"Flag if the detector fires anywhere in the clip" is maximally amplifying of per-frame false
positives — the worst possible aggregation for a low-precision detector. Tracked in the retrain
RESULTS. **Note the ordering: A1 is upstream of A4.** Fixing the decision rule on a model that
cannot produce a negative is treating the symptom.

## A5 — pre-existing, documented

- `qualifies_det` defaults to `f1: 0.57` when `target.f1` is absent, so deleting the key silently
  re-imposes the floor rather than retiring it (noted in `configs/stenosis_yolo.yaml`).
- Split-fraction distortion: CADICA is 47% of the corpus but 69% of val (42 patients hashed at 15%
  put 14 in val). Affects the estimate, not the model.

---

## Recommended order

1. **A1 — add negative sampling to `cadica_to_yolo`**, drawn *only* from non-lesion videos, at a
   configurable ratio (start ~1:1 with positives, i.e. ~1,500 negatives). This is the single change
   most likely to move per-video specificity off zero, and it is a data-construction fix, not a
   modelling gamble. Retrain and re-measure the operating table.
2. **A2** — write empty label files instead of skipping, so SSL cannot re-break it later.
3. **A4** — revisit the decision rule *after* A1, with richer per-video statistics stored so
   candidate rules can be swept offline from one inference pass.

Do not spend further GPU on augmentation, harmonization, balancing or SSL until A1 is fixed: every
one of those tunes a model that has never seen a negative example.


---

# Round 2 — deeper pass (2026-08-16)

Second sweep, verifying by execution rather than reading where possible.

## Verified CORRECT (no action)

- **Bounding-box conversion, all three sources.** Round-tripped a known box `(x=100, y=60, w=40,
  h=90)` in a deliberately NON-square 512×384 frame through each converter and compared against the
  hand-computed YOLO target. CADICA (`x y w h` absolute top-left) and Danilov (Pascal-VOC
  `xmin/ymin/xmax/ymax`) both **PASS**. `coco_to_yolo`'s math verified by inspection:
  `(x + w/2)/W, (y + h/2)/H, w/W, h/H` — correct for COCO's `[x_tl, y_tl, w, h]`.
  *(First fixture attempt was invalid — `w/IW` and `h/IH` both equalled 0.078125, so a w/h swap
  would have passed. Re-run with distinct components.)*
- **Anisotropic resize is safe.** Boxes are normalized by the ORIGINAL `W,H` before
  `cv2.resize(img, (size, size))`. Normalized coordinates are invariant under independent per-axis
  scaling, so labels still land on the lesion after the square resize.
- **`aggregate_sequence` is not buggy.** Executed against constructed sequences: `min_hits=2` drops
  isolated single-frame detections and recovers interior gaps (frames 0 and 2 hit → frame 1
  recovered) exactly as documented; `min_hits=1` keeps isolated hits. **The voting regression is a
  real property, not a defect.**
  **This is additional evidence for A1:** it means detections on annotated keyframes are frequently
  *temporally isolated*. A detector genuinely locking onto a persistent anatomical lesion would fire
  across adjacent cine frames. Incoherent frame-to-frame firing is what a detector that has never
  learned a negative looks like.
- **Preprocessing is consistent.** Processed PNGs are written CLAHE'd; `.val()` and the demo read
  those directly (no double-CLAHE), while the raw-cine paths (`P1.1b/c`, `_pseudo_label_round`)
  apply `clahe_unsharp` themselves. No train/inference mismatch found.

## B1 — MEDIUM: `coco_to_yolo` trusts COCO metadata dimensions, never the image

```python
W, H = img["width"], img["height"]     # from the JSON
...
g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)   # the actual file — dimensions never compared
```

Every box is normalized by the JSON's declared size. If that metadata disagrees with the file on
disk — a common COCO corruption — **every box for that image is silently mis-normalized**, with no
cross-check anywhere. Unvalidated assumption for ARCADE. A one-line `assert (H, W) == g.shape` (or a
logged mismatch) removes the entire class.

## B2 — MEDIUM: the converters drop images silently

`coco_to_yolo` has `if not ip: continue` (image path unresolvable) and the native readers have
`if g is None: continue` (unreadable file). Nothing is recorded; the only signal is a printed count
nobody diffs between runs.

The **ingest** pipeline was hardened against exactly this class in the 2026-08-03 audit — P0.6 added
`index_errors.jsonl`, P0.7 added the `os.walk(onerror=…)` callback, both closed on the grounds that
a silent drop produces a confident but incomplete dataset. **`src/data_prep/` never received the
same treatment.** Same failure mode, different directory.

## B3 — HIGH: the leakage auditor is blind to labels

`_split_stems` reads `images/<split>` only:

```python
d = os.path.join(out_dir, "images", split)
return {os.path.splitext(f)[0] for f in os.listdir(d) if ...}
```

`audit_split_leakage` therefore never verifies image↔label pairing, never counts label-less
(background) images, and never notices an orphan label. Given that **A1 is entirely about the
absence of background images**, an honesty gate that cannot see the label directory is a structural
blind spot.

**A background-fraction check in the auditor would have caught A1 at conversion time**, the first
time the corpus was built, instead of after two GPU runs and a clinical conversation. This is the
single highest-value addition to the existing tripwire family (`danilov_stems`, `cathaction_stems`,
`avf_stems`, `cadica_stems`): report `n_background / n_images`, and fail or warn when it is 0.

## B4 — MEDIUM: the ARCADE split is not stable across dataset configurations

`_disambiguated_stem` prefixes a split tag only when a basename collides across COCO jsons, and the
collision set `dupes` depends on **which json files happen to be present**. Attach only ARCADE's
train json and image `1.png` becomes stem `1`; attach train+val+test and the same physical image
becomes `train_1`. Different stem → different `split_of` hash → **the same image changes split**.

Not a leak within a run (grouping stays internally consistent), but splits are not reproducible
across dataset configurations, and any metric comparison between runs with different attachments is
invalid. The 2026-08-16 runs printed `300 names in >1 split`, so this path is live.

## B5 — MEDIUM: harmonization creates a train/eval target mismatch by construction

`harmonize` clamps boxes up to `min_box_wh` on the **train split only**, so val is scored against
original (smaller) GT. The docstring's justification — "val stays comparable to the un-harmonized
baseline" — is true about the *metric* but misses the consequence: the **model** is now trained to
emit boxes systematically larger than the evaluation target. At IoU 0.5 an over-sized prediction
loses IoU against a small GT box, so the lever can *lower* mAP50 while appearing conservative.
It is currently OFF (`min_box_wh: 0.0`), so this is latent — but it should not be switched on
without evaluating both harmonized and original val.

## Revised recommended order

1. **A1** — negative sampling from non-lesion CADICA videos. Unchanged, still the root cause.
2. **B3** — add a background-fraction check to `audit_split_leakage`. Cheap, and it makes A1
   impossible to reintroduce silently. Do it *with* A1 so the fix is self-verifying.
3. **A2** — write empty label files in `_pseudo_label_round`.
4. **B1 / B2** — dimension cross-check and a dropped-image record in the converters.
5. **B4** — make the ARCADE stem independent of which jsons are attached.
6. **A4 / B5** — decision rule and harmonization, both only after A1.


---

# Implementation status — verified 2026-08-16 (suite 702 passed)

Each item below was checked against the code, not against the implementer's report. A1 was
additionally re-verified with an independent adversarial fixture that reused none of its tests.

| ID | Severity | Status |
|---|---|---|
| **A1** zero negative frames | CRITICAL | **FIXED** — `cadica_to_yolo` negative sampling, 38 tests |
| **A2** SSL pseudo-label discards negatives | HIGH | **FIXED** — empty labels + `max_background_frac`, 17 tests |
| **A2b** GD seed round, third instance | HIGH | **FIXED** — same shape, 4 tests |
| **B3** auditor blind to labels | HIGH | **FIXED** — `require_backgrounds` + background counts, 4 tests |
| **A3** SSL retrains from COCO weights | MEDIUM | **FIXED** — `ssl.restart_from` knob, default unchanged, 20 tests |
| A4 per-video decision rule | — | open (deliberately: downstream of A1) |
| **B1** COCO metadata dimensions untrusted | MEDIUM | **FIXED** — corroborated against the file, fails closed |
| **B2** converters drop images silently | MEDIUM | **FIXED** — `convert_errors.jsonl`, ingest convention |
| **B4** ARCADE stems unstable across configs | MEDIUM | **FIXED** — tag iff `group_key(stem) == stem` |
| **B5** harmonize train/eval mismatch | MEDIUM | **FIXED** — warning + honest docstring, math untouched |

## Independent verification of A1 (the one that could destroy recall)

Built a fixture with lesion videos carrying GT on only 2 of 10 frames — the real CADICA shape, and
the exact trap: the other 8 frames still contain the lesion.

| check | result |
|---|---|
| negatives sourced from a LESION video | **none** — 0 of 4 |
| negative label files genuinely 0 bytes | pass |
| a patient's negatives and positives on the same split side | pass, 0 straddling |
| negatives spread across patients | pass, drawn from 3 of 3 |

Labelling is stricter than specified: a video the manifests list in *neither* file is `unknown` and
**fails closed** — never sampled — instead of falling back to the weaker groundtruth-presence rule.
Module-level imports remain stdlib-only (`argparse, glob, os, re, yaml`); no cv2/torch/numpy.
`negatives_per_positive: 0` reproduces the pre-fix behaviour exactly.

## Open judgement call: the shipped ratio is above guidance

`negatives_per_positive: 1.0` on the real corpus:

| ratio | negatives | corpus | background frac |
|---|---|---|---|
| 0.00 | 0 | 3409 | 0.0% |
| **0.25** | 397 | 3806 | **10.4%** |
| 0.50 | 794 | 4203 | 18.9% |
| **1.00** | 1589 | 4998 | **31.8%** |

Ultralytics guidance is 0–10%; the shipped default is **~3× that ceiling**. Defensible given a
false-flag rate of ~1.0, but recall is the clinically costly axis and the model already misses ~75%
of lesions, so over-dosing background is a real risk in the dangerous direction.

**This should be swept, not assumed.** Train at 0.25 and at 1.0, then score both in a single
per-video pass (the P1.1b/c cell already scores multiple weight sets) and compare operating tables.
Do not pick by per-frame F1.


---

# Round 3 — remaining findings closed, verified 2026-08-16 (suite 768 passed)

Every finding in this audit is now fixed except A4, which is deliberately deferred. Verification
below was done independently of the implementers' own tests.

## B4 adversarial re-check (the one that could reintroduce the F1 0.885 leak)

The obvious fix — always prefix the split tag — would have destroyed `group_key`'s sequence collapse
and produced a per-frame split, i.e. the exact 2026-07 leak. The implemented rule is **tag iff
`group_key(stem) == stem`**. Re-verified with a script reusing none of the shipped tests:

| check | result |
|---|---|
| Danilov / CADICA / CathAction / AVF stems left untagged, collapse intact | PASS (4/4) |
| 40 frames of one sequence still land on ONE split side | PASS (Danilov → val, CADICA → train) |
| ARCADE bare stem identical with train-only vs all-three jsons attached | PASS (`train_1` both ways) |
| tagging never FABRICATES a group | PASS |

That last row is a failure mode found by the implementer beyond the brief: json folder `14` +
image `002_5_0016.png` would naively tag to `14_002_5_0016`, which `group_key` reads as the
**fabricated patient `14_002`** — inventing a sequence that does not exist. Tagging is refused there.

**Known discontinuity:** B4 moves some ARCADE images between train and val. Metrics from a corpus
built after this change are NOT comparable to the 2026-08-16 numbers unless both sides are rebuilt.

## B1 / B2 spot-checks

`dim_mismatch` rows carry BOTH coordinate systems (`declared 512x512 vs actual 1024x1024`) rather
than silently preferring one. A corrupt image in the Danilov native path is converted-count 1,
drop-count 1, reason `unreadable`, persisted to `convert_errors.jsonl`. A torn/garbage append costs
one row and does not raise, matching `src/ingest/manifest.read_jsonl`'s degrade-don't-raise rule.

B1 also closed a latent crash found in passing: an undecodable file previously reached
`clahe_unsharp(None)` and raised `AttributeError` mid-conversion; it is now an `unreadable` drop.

## Still open — by design

**A4, the per-video decision rule.** "Flag if the detector fires anywhere in the clip" maximally
amplifies per-frame false positives. It stays open deliberately: it is downstream of A1, and tuning
aggregation is only meaningful once the retrain shows what a model that has actually seen negatives
does. Revisit with the operating table, not before.
