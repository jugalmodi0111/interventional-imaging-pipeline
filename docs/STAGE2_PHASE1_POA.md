# Stage 2 Stenosis — Phase 1 POA (Quick Wins)

**Owner:** tech@manufex.io · **Created:** 2026-07-17
**Baseline run:** `experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150` → F1 0.291 / R 0.271 / mAP50 0.209 / mAP50-95 0.080 (floor 0.57)
**Companion:** [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) §3.2 · [`STAGE_ACCURACY_RESEARCH.md`](STAGE_ACCURACY_RESEARCH.md) · run [`RESULTS.md`](../experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/RESULTS.md)

---

## 0. Scope

Phase 1 = the levers that are **testable now on the existing Kaggle/Colab T4**, are **cheap** (config/code + one retrain at most), and **cannot make things worse** (each is reversible). No new datasets, no architecture change — those are Phase 2/3.

**What Phase 1 is *not*:** it will very likely **not** clear the 0.57 per-frame F1 floor by itself. Its job is to (a) squeeze the free accuracy that's currently left on the table, (b) produce the diagnostics that aim Phase 2, and (c) stand up the **per-video, recall-first evaluation** that is the honest gate for a screening tool. Treat "clear the floor" as a Phase 2/3 outcome; treat "stop bleeding recall + know exactly where it bleeds" as the Phase 1 outcome.

### Where each step runs
- **Local (this laptop):** all code edits + unit tests. `best.pt` is here; `data/processed/stenosis` is **not** (GPU-side), so nothing that needs val images runs locally.
- **GPU (Kaggle/Colab):** every `.val()`, every retrain, every temporal-voting pass. The notebook imports `src/*`, so land code changes in the repo first, then run the notebook.

---

## 1. Baseline recap (what we're improving on)

| Signal | Value | Read |
|---|---|---|
| F1 / P / R | 0.291 / 0.314 / 0.271 | recall is the broken axis |
| mAP50 / mAP50-95 | 0.209 / 0.080 | 2.6× drop → loose localization (annotation-convention mismatch) |
| Confusion (true stenosis) | 26% caught / **74% missed** | pure miss problem, not class confusion |
| PR recall ceiling | ~0.67 at conf→0 | ~⅓ of lesions never proposed by any box |
| F1-conf peak | 0.29 @ conf 0.20 | already near-optimal operating point |
| Val plateau | epoch ~16; best ~39; stop 69 | data-limited, not optimization-limited |
| Augmentation | YOLO COCO defaults | mosaic never closed, scale/erasing/HSV mistuned for tiny faint grayscale |

---

## 2. Phase 1 objective & exit criteria

**Objective:** recover recall and localization that current defaults are throwing away, and replace the per-frame gate with a per-video recall gate that reflects deployment.

**Exit Phase 1 when all of these exist:**
1. Per-source val table (ARCADE / CADICA / Danilov) on the baseline `best.pt` — we know *where* it fails.
2. A retrained model with domain-tuned augmentation, evaluated honestly, with the F1/recall delta recorded (better, worse, or neutral — all are informative).
3. A per-video (temporal-voted) sensitivity number at a recall-first operating point.
4. A written, stakeholder-ready **gate reframe proposal** (per-video sensitivity + abstention vs per-frame F1 0.57).
5. Split accounting fixed (CADICA grouped correctly; val fraction ~15–20%).

---

## 3. Work items

Each item: **Why · Files · Change · Run · Expect · Effort · Risk · Done-when.**

---

### P1.0 — Per-source val diagnostic *(do first; no retrain)*

**Why.** The single highest-information, near-zero-cost step. The aggregate F1 0.291 hides whether the failure is concentrated in one source (e.g. ARCADE's tight/subjective boxes) or uniform. Everything downstream is aimed by this.

**Files.** New: `src/eval/val_by_source.py` (local edit + local unit test on `source_of`). Runs GPU-side.

**Change.** New script — splits the merged val set by origin (inferred from converter stem naming) and runs `.val()` per source:

```python
"""Per-source stenosis val: split the merged val set by dataset of origin and run ultralytics
.val() on each, so we see WHERE recall fails. Run GPU-side after a training run.

Source inferred from the stem naming the converters emit:
  CADICA  pXX_vYY_NNNNN                       -> starts 'p<digits>_v<digits>'
  Danilov <site>_<pat>_<seq>_<frame> (4 all-digit groups) -> io_utils._PATIENT_RE
  ARCADE  <split>_<n> or <n>                  -> everything else
"""
import os, re, glob, argparse, yaml
from ultralytics import YOLO

_CADICA = re.compile(r"^p\d+_v\d+")
_DANILOV = re.compile(r"^\d+_\d+_\d+_\d+$")

def source_of(stem):
    if _CADICA.match(stem):
        return "cadica"
    if _DANILOV.match(stem):
        return "danilov"
    return "arcade"

def _write_lists(proc):
    buckets = {}
    for ip in sorted(glob.glob(os.path.join(proc, "images", "val", "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        buckets.setdefault(source_of(stem), []).append(os.path.abspath(ip))
    lists = {}
    for src, paths in buckets.items():
        lp = os.path.join(proc, f"val_{src}.txt")
        open(lp, "w").write("\n".join(paths))
        lists[src] = (lp, len(paths))
    return lists

def main(weights, proc, conf=0.001):
    base = yaml.safe_load(open(os.path.join(proc, "data.yaml")))
    for src, (lp, n) in sorted(_write_lists(proc).items()):
        dy = os.path.join(proc, f"data_{src}.yaml")
        cfg = dict(base); cfg["val"] = os.path.abspath(lp)
        yaml.safe_dump(cfg, open(dy, "w"))
        b = YOLO(weights).val(data=dy, conf=conf, verbose=False).box
        print(f"{src:8s} n={n:5d}  P {b.mp:.3f}  R {b.mr:.3f}  mAP50 {b.map50:.3f}  mAP50-95 {b.map:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--proc", default="data/processed/stenosis")
    a = ap.parse_args(); main(a.weights, a.proc)
```

> Ultralytics resolves labels by swapping `/images/` → `/labels/` and `.png` → `.txt`, which matches our on-disk layout, so a `.txt` image-list as `val:` works with no relabeling.

**Run (GPU).**
```
python -m src.eval.val_by_source --weights runs/stenosis/.../weights/best.pt
```

**Expect.** A 3-row table. Likely pattern: CADICA best (most/most-consistent data), ARCADE worst (tight subjective boxes), Danilov noisy (few patients). Whatever it shows *redirects* Phase 2 spend.

**Effort** ~1h (local) + minutes (GPU). **Risk** none (read-only). **Done-when** the 3-row table is in the run's RESULTS.md.

---

### P1.1 — Recall-first operating point + temporal voting *(no retrain)*

**Why.** Two free wins on the *existing* `best.pt`: (a) the deployed threshold should sit at the recall-first knee, not the default 0.25; (b) stenosis is a physical lesion across a cine window — `temporal_vote.aggregate_sequence` recovers missed frames (recall↑) and drops one-frame flicker (precision↑). This is the metric that actually matters for a screening flag and it is **CoreML-neutral** (pure post-processing, already in `src/serve/`).

**Files.** No code change to train path. Uses `src/serve/temporal_vote.py` (`aggregate_sequence`) + `src/serve/stenosis_infer.py` (already wires it). Optional new: `src/eval/val_per_video.py`.

**Change / Run (GPU).**

1. **Operating-point sweep** on `best.pt` — tabulate P/R/F1 across confidences:
```python
from ultralytics import YOLO
m = YOLO("runs/stenosis/.../weights/best.pt")
for c in (0.05, 0.10, 0.15, 0.20, 0.25):
    b = m.val(data="data/processed/stenosis/data.yaml", conf=c, verbose=False).box
    p, r = float(b.mp), float(b.mr); f1 = 2*p*r/(p+r) if p+r else 0
    print(f"conf {c:.2f}  P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")
```
Pick the recall-first point (expect ~0.10 — trades precision for the costly-error axis).

2. **Per-video temporal sensitivity** — group val frames by video, predict in frame order, vote, compare recall before/after:
```python
# sketch: per CADICA video (pXX_vYY) / Danilov clip (<site>_<patient>_<seq>)
from src.serve.temporal_vote import aggregate_sequence
# frames_dets = [[{"box":(cx,cy,w,h),"conf":..}, ...], ...]  # per-frame YOLO preds in order
stab = aggregate_sequence(frames_dets, iou_thr=0.3, min_hits=2, conf_agg="mean")
# per-video HIT if any surviving track overlaps the GT lesion on >=1 frame
```
Report: raw per-frame recall vs voted per-frame recall vs **per-video sensitivity** (fraction of lesion-videos with ≥1 recovered detection). Tune `min_hits` (2 = balanced; 1 = max recall).

**Expect.** Per-video sensitivity materially higher than the 0.271 per-frame recall — this is the headline deployment number. Voting typically adds recall *and* precision on cine.

**Effort** ~half day (GPU). **Risk** none (post-processing). **Done-when** operating-point table + per-video sensitivity recorded, and `min_hits`/conf chosen for deployment.

---

### P1.2 — Augmentation retune *(code plumbing + config + one retrain)*

**Why.** `run/args.yaml` shows pure COCO defaults. For 23–115px faint grayscale lesions these are wrong or harmful: `mosaic 1.0` (never closed — run stopped at ep69, `close_mosaic` fires at ep141) shrinks/composites tiny lesions; `scale 0.5` shrinks them below detectability; `erasing 0.4` can erase the lesion; `hsv_*` is a no-op on grayscale. **These are not currently configurable** — `train_kwargs()` only forwards imgsz/epochs/batch/lr0/cache/workers/patience/amp. So step one is a config passthrough.

**Files.** `src/train/train_detector.py` (`train_kwargs`), `configs/stenosis_yolo.yaml`, `tests/test_train_detector.py`.

**Change 1 — passthrough (local, TDD).**
```python
def train_kwargs(cfg):
    m, tr = _detector(cfg) or {}, cfg.get("train", {})
    kw = {"imgsz": m.get("imgsz", 640),
          "epochs": tr.get("epochs", 100), "batch": tr.get("batch", 16),
          "lr0": tr.get("lr", 1e-3),
          "cache": tr.get("cache", True), "workers": tr.get("workers", 8),
          "patience": tr.get("patience", 30), "amp": tr.get("amp", True)}
    aug = cfg.get("augment") or {}          # domain-tuned overrides (small faint grayscale lesions)
    kw.update({k: v for k, v in aug.items() if v is not None})
    return kw
```
Unit test: `train_kwargs({"augment": {"mosaic": 0.0}})["mosaic"] == 0.0`; absent block leaves defaults untouched.

**Change 2 — config block.** Add to `configs/stenosis_yolo.yaml`:
```yaml
augment:                 # tuned for tiny, faint, grayscale stenosis targets
  mosaic: 0.0            # A/B this: 0.0 (safest for tiny objects) vs 0.3
  close_mosaic: 15       # if mosaic>0, ensure an off-tail actually fires
  scale: 0.2             # was 0.5 — heavy scale-down loses 23px lesions
  erasing: 0.0           # was 0.4 — random-erase can delete the lesion
  hsv_h: 0.0
  hsv_s: 0.0
  hsv_v: 0.0             # grayscale: HSV jitter is wasted
  fliplr: 0.5            # keep
  translate: 0.1         # keep
  degrees: 0.0
  cos_lr: true           # smoother decay; pairs with shorter schedule
  box: 9.0               # was 7.5 — push localization (mAP50-95 was 0.080)
  dfl: 2.0               # was 1.5
```
Also drop `train.epochs` 150 → **80** (saturates by ~40) to save GPU. All keys above are valid `model.train()` kwargs.

**Run (GPU).** Two runs, tag them: `..._augtuned_mosaic0` and `..._augtuned_mosaic03`. Same honest patient-grouped split + leakage audit as baseline.

**Expect.** +0.03–0.08 F1, concentrated in recall and mAP50-95 (tighter boxes). Possible neutral — record either way; a null result rules the lever out cheaply.

**Effort** ~2h local + 2 short GPU runs. **Risk** low (reversible; worst case = baseline). **Done-when** both A/B runs evaluated (incl. P1.0 per-source + P1.1 per-video) and best config chosen.

---

### P1.3 — Split rebalance + CADICA grouping fix *(code + reconvert)*

**Why.** Val ballooned to ~34% by group, starving training. Root cause: CADICA stems `pXX_vYY_NNNNN` don't match `group_key`'s regexes, so the auditor counts **each CADICA frame as its own group** and the 42-patient hash lands image-heavy patients in val. Two fixes: teach `group_key` about CADICA (accounting + belt-and-suspenders honesty), and cap CADICA frames/patient to cut redundancy and val variance.

**Files.** `src/data_prep/io_utils.py` (`group_key`), `src/data_prep/cadica_to_yolo.py` (optional cap), `tests/test_split_grouping.py`.

**Change 1 — group_key (local, TDD).**
```python
_CADICA_RE = re.compile(r"^(p\d+)_v\d+_\d+")   # CADICA pXX_vYY_NNNNN -> patient pXX

def group_key(name):
    m = _PATIENT_RE.match(name)
    if m: return m.group(1)
    m = _CADICA_RE.match(name)
    if m: return m.group(1)
    m = _CLIP_RE.match(name)
    if m: return m.group(1)
    return name
```
Test: `group_key("p12_v3_00045") == "p12"`; Danilov/CathAction/ARCADE unchanged.

**Change 2 — cap CADICA frames/patient (optional but recommended).** Wire `cap_frames_per_patient(..., k)` (already in `io_utils`) into `cadica_to_yolo` via a `cadica.max_frames_per_patient: 40` config knob, mirroring Danilov's cap. Cuts 3996 near-duplicate keyframes toward evenly-spaced ~40/patient.

**Run (GPU).** Re-run the converters (`cadica_to_yolo`, etc.) → `audit_split_leakage` should now report **~42 CADICA patient groups** and `val_frac_by_group` ~0.15–0.20. Then retrain (fold into the P1.2 run).

**Expect.** More training images retained → small recall/F1 lift; honest, interpretable split accounting. Note: leakage was already absent (conversion split on `pXX` directly) — this fixes *accounting* and *balance*, not a live leak.

**Effort** ~2h local + reconvert. **Risk** low. Re-run the leakage gate — must stay PASS. **Done-when** audit shows correct group counts + val fraction, leakage PASS, model retrained on the rebalanced split.

---

### P1.4 — Combined honest re-run + archive *(integration)*

**Why.** Fold P1.2 + P1.3 into one clean run so the deltas are attributable and the artifact is archivable like prior runs.

**Run (GPU).** One run with tuned augmentation + rebalanced/grouped split + `epochs 80` + `val.conf 0.001`. Keep `target: {f1: 0.57, recall: 0.60}` (uncomment) so the recall floor prints in the gate line.

**Then, on the resulting best.pt:** P1.0 (per-source) + P1.1 (operating point + per-video voting).

**Archive** to `experiments/stenosis_<tag>/` with a RESULTS.md carrying: the metric table, per-source table, per-video sensitivity, and the aug/split diff vs baseline. Update `PROJECT_TRACKER.md` §3.2 + changelog.

**Done-when** archived + tracker updated + Phase 1 exit criteria (§2) all satisfied.

---

## 4. Execution order & dependencies

```
P1.0 per-source val ──┐  (baseline best.pt; independent, do immediately)
P1.1 op-point+voting ─┤  (baseline best.pt; independent, parallel to P1.0)
                      │
P1.2 aug retune ──────┼──► P1.4 combined re-run ──► rerun P1.0 + P1.1 on new best.pt ──► archive
P1.3 split/group fix ─┘
```
- P1.0 and P1.1 run **now** on the existing `best.pt` — no dependency, no retrain.
- P1.2 and P1.3 are local code edits (TDD) that both feed the single P1.4 retrain.
- Do all local edits + `pytest tests/` green **before** the GPU run.

**Local pre-flight (one command):** `pytest tests/` must stay green (currently 150 passing) after the `train_kwargs`, `group_key`, and `val_by_source.source_of` edits.

---

## 5. Acceptance gates & decision → Phase 2

**Phase 1 succeeds** (regardless of whether the floor is cleared) when §2's five artifacts exist. Then decide with this rule:

| Post-P1 outcome | Decision |
|---|---|
| Per-video sensitivity ≥ stakeholder screening bar (e.g. ≥0.80) at acceptable precision | Ship as **deferred screening flag** under the reframed gate; Stage 2 is deployable even below per-frame 0.57. |
| Per-frame F1 lift is real but < 0.57, per-video short | Proceed to **Phase 2**: annotation harmonization (aim with P1.0 table) + pseudo-label SSL on XCAD. |
| Aug/split neutral, per-source shows one bad source | Phase 2 prioritizes fixing/dropping that source over adding data. |

**Do not** invest Phase 2/3 GPU before the P1.0 per-source table exists — it's the map.

---

## 6. Risks & rollback

- Every change is config/data-prep level and reversible; the baseline run is archived, so worst case is "no improvement," never regression of the record.
- **Leakage gate is the hard guard** — P1.3 reconvert must keep `audit_split_leakage` PASS; if CADICA grouping change ever makes a group span splits, it raises before training.
- **CoreML**: nothing in Phase 1 touches exportability (no new layers, same YOLO11s, same imgsz). Temporal voting is host-side post-processing, not in the CoreML graph.
- **Gate reframe is a proposal, not a unilateral change** — the 0.57 → per-video-sensitivity move needs clinical sign-off (Stage 5 intended-use). Phase 1 delivers the numbers to make that call, not the call itself.

---

## 7. Effort / impact summary

| Item | Effort | Expected impact | Retrain? |
|---|---|---|---|
| P1.0 per-source val | ~1h + mins GPU | **High info** (aims everything) | no |
| P1.1 op-point + temporal voting | ~½ day GPU | **High** on deployment metric (per-video recall) | no |
| P1.2 augmentation retune | ~2h + 2 GPU runs | **Med** (+0.03–0.08 F1, recall/localization) | yes |
| P1.3 split rebalance + CADICA grouping | ~2h + reconvert | **Low–med** (more train data, honest accounting) | yes (folds into P1.4) |
| P1.4 combined re-run + archive | 1 GPU run | consolidates the above | yes |
