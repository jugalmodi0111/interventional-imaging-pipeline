# Diagnostic Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current pile of task-specific cath-lab models into one pipeline where a clinician uploads an image *or* a cine clip, the system decides what modality/view it is, routes it to the right model(s), and returns a structured, calibrated, defer-aware **diagnostic report** naming the finding — instead of today's "operator pre-selects `TASK=det` and a weights path, gets raw boxes back."

**Architecture:** A thin **modality/view router** (RAD-DINO frozen-encoder + linear head on the build side, distilled to a MobileNetV3 student for the edge) classifies each frame into a modality. A **registry** (`configs/orchestrator.yaml`) maps modality → task model + finding metadata. A **diagnosis layer** turns per-frame detections/masks + the existing triage-with-abstention into typed `Finding`s. A **DiagnosticOrchestrator** ties router → registry → model → diagnosis together, aggregates a clip with the existing `temporal_vote.aggregate_sequence`, and emits a `StudyReport`. A FastAPI `/analyze` endpoint accepts image or video. Everything defers to a human when the router is unsure, the modality is unsupported, or the anchor detector is below its accuracy floor.

**Tech Stack:** Python 3.12, PyTorch ≥2.2, Ultralytics ≥8.2 (YOLO11), `transformers` ≥4.44 (RAD-DINO encoder, build-side only), `timm`/`torchvision` (MobileNetV3 student), coremltools (Apple-silicon edge export), FastAPI + uvicorn (serving), pytest (all pure logic torch-free). nnU-Net v2 teacher for seg. numpy/opencv for frame IO + CLAHE.

## Global Constraints

- **Golden invariant (verbatim from PROJECT_TRACKER):** heavy models are *teachers/labelers on the GPU build side only*. Only distilled/quantized students ship to edge (Mac / procedure-cart). RAD-DINO and nnU-Net obey this — the router that ships is the **distilled MobileNetV3 student**, never the RAD-DINO encoder.
- **Safety default is DEFER, not guess.** A missed stenosis is the deadly error. Any uncertainty — unknown modality, low router confidence, below-floor anchor model, low detector confidence, OOD — routes the case to a human. Never emit a confident diagnosis the models cannot support. `stenosis_triage` already encodes this; the orchestrator must not bypass it.
- **No autonomous diagnosis claim.** Intended use is **screening-with-abstention / second-read**, not autonomous Dx. Output copy must say "possible finding — clinician review required," never "diagnosis: X." Enforced in `docs/INTENDED_USE.md` (Phase F) and asserted in tests.
- **Import-safe modules.** No heavy import (torch, ultralytics, coremltools, transformers) at module top level in anything under `src/serve/` or `src/models/` — import lazily inside functions, so pure helpers stay unit-testable on this laptop with no GPU deps. This is an existing repo convention; every new module follows it.
- **Build side = Colab/Kaggle GPU** (thin notebooks import `src/*`); **deploy side = Mac CoreML**. Training tasks below run in a notebook on Kaggle/Colab; serving/logic tasks run and are tested locally.
- **Every prediction is audit-logged** via `src.eval.audit.record(version, input_array, meta)` — existing convention, preserved by the orchestrator.
- **Python version:** target 3.12 (matches Kaggle runtime `Python-3.12.13`).

---

## Dependency graph (read before picking a task order)

```
Phase A  Anchor: stenosis above floor ──┐  (unblocks a trustworthy finding to report)
Phase B  Modality/View router ──────────┼──► Phase D  Orchestrator ──► Phase E  /analyze serving
Phase C  Diagnosis / Finding layer ─────┘        (registry + report)        (image + video)
                                                        │
                                                        └──► Phase F  Intended-use gate + end-to-end verification
```

- **Phases B, C, and the report/registry parts of D are pure-logic + independent** — they can be built and tested with fake models on this laptop *before* Phase A's training finishes. Do these first; they need no GPU.
- **Phase A is experiment work** (runs on Kaggle GPU, gated by a metric, not a unit test). It runs in parallel with B/C/D.
- **Phase E** needs C + D. **Phase F** needs everything.

---

## File Structure

**New files:**
- `src/serve/report.py` — `Finding` + `StudyReport` dataclasses, `.to_dict()` JSON serialization. One responsibility: the output contract. Pure stdlib.
- `src/serve/registry.py` — `TaskEntry` dataclass, `load_registry(path)`, `resolve(registry, modality)`. Maps modality → model + finding metadata. Pure stdlib + pyyaml.
- `src/serve/diagnosis.py` — `det_to_findings(entry, triage)`, `seg_to_finding(entry, seg_res)`, `study_defer(decision, findings)`. Turns raw model output + triage into typed findings and the study-level defer decision. Pure numpy.
- `src/serve/router.py` — `ModalityDecision` dataclass, pure `decide_modality(probs, ...)`, and the `ModalityRouter` class (lazy-loads the edge classifier, calls `decide_modality`). Split: pure decision logic (tested) vs model wrapper (thin).
- `src/serve/orchestrator.py` — `DiagnosticOrchestrator` (`analyze_frame`, `analyze_video`). Wires router → registry → model_factory → diagnosis → temporal aggregation → `StudyReport`.
- `src/train/train_router.py` — build-side: RAD-DINO frozen-encoder + linear head teacher, then distill to MobileNetV3-small student; export student to CoreML. Pure config/label helpers unit-tested; heavy path lazy.
- `src/data_prep/build_router_manifest.py` — scan the per-modality datasets already on disk and emit a `router_manifest.csv` (path,label) for router training. Pure path→label helpers unit-tested.
- `configs/orchestrator.yaml` — the registry: modality → task/model_path/finding metadata + router thresholds.
- `configs/router.yaml` — router training config: modality label set, encoder id, student arch, thresholds.
- `notebooks/kaggle_router_build.ipynb` — thin GPU runner for `train_router.py`.
- `docs/INTENDED_USE.md` — intended-use / regulatory-posture gate (screening-with-abstention, not autonomous Dx).
- Tests: `tests/test_report.py`, `tests/test_registry.py`, `tests/test_diagnosis.py`, `tests/test_router.py`, `tests/test_orchestrator.py`, `tests/test_build_router_manifest.py`, `tests/test_train_router.py`, `tests/test_analyze_endpoint.py`.

**Modified files:**
- `src/serve/app.py` — add `POST /analyze` (image or video) returning a `StudyReport` dict; keep the old `/infer` for back-compat.
- `configs/edge_export.yaml` — add the router student as a shipped edge artifact.
- `requirements.txt` — add `timm` (MobileNetV3 student); `transformers` already present (RAD-DINO).
- `Makefile` — add `prep-router`, `train-router`, `export-router-coreml`, `serve-analyze` targets.
- `docs/PROJECT_TRACKER.md` — add Stage 6 (orchestrator) rows; flip when gates pass.

---

## Interface contract (locked — every task uses these exact names/types)

```python
# src/serve/report.py
@dataclass
class Finding:
    label: str                 # machine key, e.g. "coronary_stenosis"
    display_name: str          # clinician-facing, e.g. "Possible coronary artery stenosis (blockage)"
    confidence: float          # calibrated [0,1]; 0.0 when deferred with no positive
    deferred: bool             # this finding routed to a human
    reason: str                # "confident" | "clean" | "low-confidence" | "ood" | "no-detection-uncertain" | "below-floor" | "unsupported-modality" | "router-uncertain"
    severity: str | None = None        # optional band, None until Phase C severity is enabled
    boxes: list = field(default_factory=list)   # [(x1,y1,x2,y2,conf), ...] pixel coords, empty for seg/negatives

@dataclass
class StudyReport:
    modality: str              # "coronary_angiography" | ... | "unknown"
    view: str | None
    quality_ok: bool
    findings: list             # list[Finding]
    deferred: bool             # study-level: any finding deferred OR modality unknown/unsupported
    defer_reason: str          # top reason surfaced to the operator ("" if not deferred)
    frames_analyzed: int
    model_versions: dict       # {"router": "...", "coronary_stenosis": "best.pt@abc123", ...}
    def to_dict(self) -> dict: ...   # JSON-safe, tuples -> lists

# src/serve/registry.py
@dataclass
class TaskEntry:
    modality: str
    task: str                  # "det" | "seg"
    model_path: str
    display_name: str          # modality label for the report
    finding_label: str         # -> Finding.label
    finding_display: str       # -> Finding.display_name
    floor_ok: bool             # False if the model is below its accuracy floor (forces defer)
def load_registry(path: str) -> dict:          # {modality: TaskEntry}
def resolve(registry: dict, modality: str):    # -> TaskEntry | None

# src/serve/router.py
@dataclass
class ModalityDecision:
    modality: str              # or "unknown"
    view: str | None
    quality_ok: bool
    confidence: float          # top calibrated prob
    deferred: bool
    reason: str                # "confident" | "router-uncertain" | "low-quality"
def decide_modality(probs: dict, *, keep_thr: float = 0.60, margin: float = 0.15,
                    quality_prob: float | None = None, quality_thr: float = 0.5) -> ModalityDecision:
class ModalityRouter:
    def __init__(self, weights: str, labels: list, thresholds: dict | None = None, size: int = 224): ...
    def classify(self, frame_gray) -> ModalityDecision: ...   # CLAHE -> student -> softmax -> decide_modality

# src/serve/diagnosis.py
def det_to_findings(entry, triage: dict) -> list:   # triage = stenosis_triage.triage_decision(...) output
def seg_to_finding(entry, seg_res: dict) -> Finding: # seg_res = infer.SegModel(...) output
def study_defer(decision, findings: list) -> tuple:  # -> (deferred: bool, reason: str)

# src/serve/orchestrator.py
class DiagnosticOrchestrator:
    def __init__(self, router, registry: dict, model_factory, cfg: dict | None = None): ...
    def analyze_frame(self, frame_gray) -> StudyReport: ...
    def analyze_video(self, path: str, stride: int = 5, max_frames: int = 400) -> StudyReport: ...
    # model_factory(entry: TaskEntry) -> callable(frame_gray) -> dict   (injected; tests pass fakes)
```

`triage_decision` (existing, `src/serve/stenosis_triage.py`) returns
`{"prediction": [...], "calibrated_confs": [...], "deferred": bool, "reason": str}`.
`SegModel`/`DetModel.__call__` (existing, `src/serve/infer.py`) return dicts with `deferred`,
`confidence`/`top_conf`, `mask`/`boxes`. The orchestrator consumes those verbatim.

---

# PHASE A — Anchor: get stenosis above its accuracy floor

**Why first-in-parallel:** the orchestrator can only *report* a finding whose model clears its floor. Stenosis (F1 0.214 < 0.57) is the anchor. Until it clears, the registry marks it `floor_ok: false` and the orchestrator emits it as **deferred "below-floor"** — the plumbing (Phases B–E) is built and tested regardless, but the product isn't clinically useful until A lands. These are **experiment tasks gated by a metric**, not TDD units; each has an explicit accept/reject gate and an archived `RESULTS.md`.

### Task A1: CADICA run — measure the patient-diversity lift

**Files:**
- Run: `notebooks/kaggle_stenosis_plug_and_play.ipynb` (already fixed: single CADICA symlink cell, `--imgsz` handled)
- Produce: `experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/RESULTS.md`

- [ ] **Step 1:** On Kaggle: GPU + Internet ON; `+ Add Input` ARCADE, danilov, CADICA (`selectedVideos` present — verified: run tag shows `arcade+cadica+danilov`).
- [ ] **Step 2:** DRY_RUN=True → Run All. Confirm the §3b leakage audit prints `LEAKAGE CHECK PASSED` and the CADICA count is non-zero (`cadica: N frames`, N>0).
- [ ] **Step 3:** DRY_RUN=False → Run All. Full 150-epoch run.
- [ ] **Step 4 (GATE):** Read the `[PASS]/[FAIL] F1 vs floor 0.57` line and the recall.
  - **Accept** if F1 ≥ 0.57 AND recall ≥ 0.60 → set `floor_ok: true` for coronary_stenosis in `configs/orchestrator.yaml` (Phase D), archive best.pt, done with Phase A.
  - **Reject** (expected on first pass — 42 CADICA patients is still few) → record the delta vs 0.214 baseline, proceed to A2.
- [ ] **Step 5:** Save `experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/RESULTS.md` (copy the format of the existing `_grouped/RESULTS.md`: split counts, F1/P/R/mAP, read-out, next lever). Commit the RESULTS.md (not weights).

```bash
git add experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/RESULTS.md
git commit -m "exp(stenosis): archive arcade+cadica+danilov run (F1 <result> vs 0.57 floor)"
```

### Task A2: Pseudo-label SSL round to raise recall

**Files:**
- Config: `configs/stenosis_yolo.yaml:ssl` (`pseudo_label: true, conf: 0.4` present; needs a disjoint unlabeled dir)
- Data: attach an **unlabeled-only** angiography source (XCAD frames) as `ssl.unlabeled_dir` — must contain **no val patients** (the notebook auto-disables SSL otherwise to prevent re-leak).
- Run: same notebook, SSL enabled.

- [ ] **Step 1:** Attach XCAD (unlabeled) as a Kaggle dataset. Set `cfg['ssl']['unlabeled_dir']` to its path in a notebook cell before train.
- [ ] **Step 2:** Confirm the notebook does NOT print `SSL disabled (... no disjoint ssl.unlabeled_dir ...)` — if it does, the dir wasn't found or overlaps val; fix the path.
- [ ] **Step 3:** Run All (DRY_RUN=False).
- [ ] **Step 4 (GATE):** Compare **recall** and F1 to A1. Accept the SSL round only if recall rises without F1 dropping below A1. Archive `experiments/stenosis_arcade+cadica+danilov_ssl_yolo11s_768/RESULTS.md`.
- [ ] **Step 5:** Commit RESULTS.md.

### Task A3 (only if A1+A2 still below floor): bigger student / RT-DETR fallback

**Files:**
- Config: `configs/stenosis_yolo.yaml:model.name` → `yolo11m` (try), else RT-DETR-R18.

- [ ] **Step 1:** Set `model.name: yolo11m`, keep imgsz 768; if T4 OOMs, `train.batch: 8`.
- [ ] **Step 2:** Run All. Gate on F1 ≥ 0.57 recall-weighted.
- [ ] **Step 3 (GATE + honesty):** Per `STAGE_ACCURACY_RESEARCH.md`, S2 is data-limited — expect small gains from model size until patient count grows. If still below floor after A1–A3, **record that the anchor remains below floor**; the orchestrator keeps emitting stenosis as deferred "below-floor". Do NOT relabel it `floor_ok: true`. Log what was tried in `RESULTS.md` and update PROJECT_TRACKER Stage 2.
- [ ] **Step 4:** Commit RESULTS.md.

> **Phase A exit:** either `coronary_stenosis` is `floor_ok: true` (registry flips, real finding surfaces) or it is documented below-floor (orchestrator surfaces it as a deferred screening flag). Both are valid Phase-A completions — the orchestrator handles both by design.

---

# PHASE B — Modality / View router (the input-side "what is this image")

**Goal:** classify a frame into one of the supported modalities (+ a coarse view + a quality flag), so the orchestrator can pick the right model instead of the operator pre-selecting it. Build side: RAD-DINO frozen encoder + linear head (label-efficient, per PROJECT_TRACKER's explicit "use a DINOv2/RAD-DINO encoder + linear head, not Grounding DINO" decision). Edge: distill to MobileNetV3-small (golden invariant).

Label set (start small, extend later): `coronary_angiography`, `cerebral_dsa`, `other_xray`, `non_medical`. `other_xray` and `non_medical` are the **reject buckets** — anything not a supported interventional modality must land there so the orchestrator defers instead of mis-routing a chest X-ray to the stenosis model.

## B0 — pure decision logic first (no GPU, fully TDD)

### Task B0: `decide_modality` — thresholded, margin-aware, defers on ambiguity

**Files:**
- Create: `src/serve/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Produces: `ModalityDecision`, `decide_modality(probs, keep_thr, margin, quality_prob, quality_thr)` — consumed by `ModalityRouter` (B4) and `DiagnosticOrchestrator` (Phase D).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py
import pytest
from src.serve.router import decide_modality, ModalityDecision

def test_confident_top_class_is_kept():
    d = decide_modality({"coronary_angiography": 0.9, "cerebral_dsa": 0.05, "other_xray": 0.05})
    assert isinstance(d, ModalityDecision)
    assert d.modality == "coronary_angiography"
    assert d.deferred is False
    assert d.reason == "confident"

def test_below_keep_threshold_defers_unknown():
    d = decide_modality({"coronary_angiography": 0.5, "cerebral_dsa": 0.3, "other_xray": 0.2})
    assert d.modality == "unknown"
    assert d.deferred is True
    assert d.reason == "router-uncertain"

def test_thin_margin_between_top_two_defers():
    # top prob clears keep_thr but the runner-up is within `margin` -> ambiguous -> defer
    d = decide_modality({"coronary_angiography": 0.62, "cerebral_dsa": 0.55, "other_xray": 0.0},
                        keep_thr=0.60, margin=0.15)
    assert d.modality == "unknown"
    assert d.deferred is True
    assert d.reason == "router-uncertain"

def test_low_quality_flag_defers_even_if_confident_class():
    d = decide_modality({"coronary_angiography": 0.95, "other_xray": 0.05},
                        quality_prob=0.2, quality_thr=0.5)
    assert d.quality_ok is False
    assert d.deferred is True
    assert d.reason == "low-quality"

def test_reject_bucket_class_is_returned_not_unknown():
    # a confident non-medical image is a real, keepable classification (-> orchestrator will defer as unsupported)
    d = decide_modality({"non_medical": 0.97, "coronary_angiography": 0.03})
    assert d.modality == "non_medical"
    assert d.deferred is False
    assert d.reason == "confident"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `ImportError: cannot import name 'decide_modality'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/serve/router.py
"""Modality/view router decision layer + edge classifier wrapper.

The pure `decide_modality` is torch-free and unit-tested. `ModalityRouter` lazy-loads the distilled
MobileNetV3 student (edge) and delegates the keep/defer call to `decide_modality`. Safety default:
DEFER (modality 'unknown') whenever the top class is weak or the top-two margin is thin — a wrong
route sends a frame to the wrong disease model, so ambiguity must never resolve to a guess.
"""
from dataclasses import dataclass


@dataclass
class ModalityDecision:
    modality: str
    view: str | None
    quality_ok: bool
    confidence: float
    deferred: bool
    reason: str


def decide_modality(probs, *, keep_thr=0.60, margin=0.15,
                    quality_prob=None, quality_thr=0.5, view=None):
    """Softmax dict -> keep/defer decision. Defers to 'unknown' on weak top prob or thin margin."""
    quality_ok = quality_prob is None or quality_prob >= quality_thr
    if not probs:
        return ModalityDecision("unknown", view, quality_ok, 0.0, True, "router-uncertain")
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    (top_label, top_p) = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
    if not quality_ok:
        return ModalityDecision(top_label, view, False, float(top_p), True, "low-quality")
    if top_p < keep_thr or (top_p - runner_p) < margin:
        return ModalityDecision("unknown", view, quality_ok, float(top_p), True, "router-uncertain")
    return ModalityDecision(top_label, view, quality_ok, float(top_p), False, "confident")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_router.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/serve/router.py tests/test_router.py
git commit -m "feat(router): pure modality decision layer with margin + quality defer"
```

## B1 — router training data manifest

### Task B1: `build_router_manifest` — path→label from the on-disk modality datasets

**Files:**
- Create: `src/data_prep/build_router_manifest.py`
- Test: `tests/test_build_router_manifest.py`

**Interfaces:**
- Produces: `label_for_path(path, rules)` (pure), `build_manifest(roots, rules, out_csv)` (IO) — consumed by `train_router.py` (B2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_router_manifest.py
from src.data_prep.build_router_manifest import label_for_path

RULES = {
    "coronary_angiography": ["arcade", "danilov", "cadica", "dca1"],
    "cerebral_dsa": ["dsa", "cerebral"],
    "other_xray": ["chestxray", "mura"],
}

def test_maps_coronary_source_dirs():
    assert label_for_path("/data/raw/cadica/selectedVideos/p1/v1/input/x.png", RULES) == "coronary_angiography"
    assert label_for_path("/kaggle/input/arcade/stenosis/train/img/9.png", RULES) == "coronary_angiography"

def test_maps_cerebral_and_other():
    assert label_for_path("/data/raw/cerebral_dsa/seq3/f10.png", RULES) == "cerebral_dsa"
    assert label_for_path("/data/raw/chestxray14/000001.png", RULES) == "other_xray"

def test_unmatched_path_returns_none():
    assert label_for_path("/data/raw/mystery/x.png", RULES) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_build_router_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_prep.build_router_manifest'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_prep/build_router_manifest.py
"""Scan the per-modality dataset roots already on disk and emit a (path,label) manifest CSV for
router training. Labels come from substring rules on the path (dataset name -> modality), so adding
a modality = adding one rule. Frames that match no rule are dropped (logged), never mislabeled."""
import argparse, csv, glob, os, yaml

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def label_for_path(path, rules):
    """First modality whose any substring appears in the lowercased path; else None."""
    p = path.lower()
    for label, subs in rules.items():
        if any(s in p for s in subs):
            return label
    return None


def build_manifest(roots, rules, out_csv, per_class_cap=4000):
    counts, rows = {}, []
    for root in roots:
        for ext in _EXTS:
            for fp in glob.glob(os.path.join(root, "**", "*" + ext), recursive=True):
                lab = label_for_path(fp, rules)
                if lab is None:
                    continue
                if counts.get(lab, 0) >= per_class_cap:      # cap dominant classes -> balanced-ish
                    continue
                counts[lab] = counts.get(lab, 0) + 1
                rows.append((fp, lab))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows([("path", "label"), *rows])
    print("router manifest:", out_csv, "| per-class:", counts)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    build_manifest(cfg["roots"], cfg["rules"], cfg["manifest_csv"],
                   per_class_cap=cfg.get("per_class_cap", 4000))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_build_router_manifest.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write `configs/router.yaml` and commit**

```yaml
# configs/router.yaml
task: modality_view_routing
labels: [coronary_angiography, cerebral_dsa, other_xray, non_medical]
roots:
  - data/raw/arcade
  - data/raw/danilov
  - data/raw/cadica
  - data/raw/dca1
  - data/raw/cerebral_dsa
  - data/raw/other_xray        # e.g. a public chest-xray sample as the 'not-ours' bucket
manifest_csv: data/processed/router/manifest.csv
per_class_cap: 4000
encoder: microsoft/rad-dino          # build-side teacher encoder (frozen)
student: mobilenetv3_small_100       # timm id; the shipped edge classifier
size: 224
train: {epochs: 30, batch: 64, lr: 3.0e-4, val_frac: 0.15}
thresholds: {keep_thr: 0.60, margin: 0.15, quality_thr: 0.5}
```

```bash
git add src/data_prep/build_router_manifest.py tests/test_build_router_manifest.py configs/router.yaml
git commit -m "feat(router): dataset->label manifest builder + router config"
```

## B2 — router training (build side, experiment-gated)

### Task B2: `train_router.py` — RAD-DINO+linear head teacher → MobileNetV3 student

**Files:**
- Create: `src/train/train_router.py`
- Test: `tests/test_train_router.py` (pure helpers only — no GPU in CI)

**Interfaces:**
- Produces: `split_manifest(rows, val_frac, seed)` (pure), `class_weights(counts)` (pure), `train(cfg)` (heavy, lazy imports) → writes `runs/router/student.pt` + label list.

- [ ] **Step 1: Write the failing test (pure helpers)**

```python
# tests/test_train_router.py
from src.train.train_router import split_manifest, class_weights

def test_split_is_disjoint_and_sized():
    rows = [(f"/x/{i}.png", "coronary_angiography") for i in range(100)]
    tr, va = split_manifest(rows, val_frac=0.2, seed=0)
    assert len(va) == 20 and len(tr) == 80
    assert set(p for p, _ in tr).isdisjoint(p for p, _ in va)

def test_class_weights_upweight_rare_classes():
    w = class_weights({"coronary_angiography": 900, "cerebral_dsa": 100})
    assert w["cerebral_dsa"] > w["coronary_angiography"]   # rarer -> heavier
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_train_router.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation (pure helpers real; heavy path lazy)**

```python
# src/train/train_router.py
"""Build-side router training: RAD-DINO frozen encoder + linear head (label-efficient teacher),
distilled into a MobileNetV3-small student that ships to the edge (golden invariant). Pure split/
weight helpers are torch-free + unit-tested; all torch/transformers/timm imports are lazy so this
module imports on a CPU laptop with no GPU deps."""
import csv


def read_manifest(path):
    with open(path) as f:
        return [(r["path"], r["label"]) for r in csv.DictReader(f)]


def split_manifest(rows, val_frac=0.15, seed=0):
    """Deterministic shuffle-free split: stride by hash so it's reproducible without Random()."""
    val = [r for i, r in enumerate(rows) if (i * 2654435761 + seed) % 1000 < val_frac * 1000]
    val_set = set(id(r) for r in val)
    train = [r for r in rows if id(r) not in val_set]
    return train, val


def class_weights(counts):
    """Inverse-frequency weights, normalized to mean 1.0."""
    n = sum(counts.values()); k = len(counts)
    raw = {c: n / (k * v) for c, v in counts.items()}
    m = sum(raw.values()) / len(raw)
    return {c: w / m for c, w in raw.items()}


def train(cfg):
    """Heavy path (GPU): fit RAD-DINO+head teacher, distill to timm student, save student.pt."""
    import torch, timm, numpy as np, cv2
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel
    from src.data_prep.preprocess import clahe_unsharp

    labels = cfg["labels"]; lab2i = {l: i for i, l in enumerate(labels)}
    rows = read_manifest(cfg["manifest_csv"])
    tr, va = split_manifest(rows, cfg["train"]["val_frac"])
    counts = {}
    for _, l in tr:
        counts[l] = counts.get(l, 0) + 1
    cw = torch.tensor([class_weights(counts).get(l, 1.0) for l in labels])

    size = cfg["size"]

    class DS(Dataset):
        def __init__(self, rr): self.rr = rr
        def __len__(self): return len(self.rr)
        def __getitem__(self, i):
            p, l = self.rr[i]
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            x = cv2.resize(clahe_unsharp(g), (size, size)).astype("float32") / 255.0
            return torch.from_numpy(x)[None].repeat(3, 1, 1), lab2i[l]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AutoModel.from_pretrained(cfg["encoder"]).to(dev).eval()   # frozen teacher encoder
    for p in enc.parameters(): p.requires_grad = False
    head = torch.nn.Linear(enc.config.hidden_size, len(labels)).to(dev)
    student = timm.create_model(cfg["student"], num_classes=len(labels), in_chans=3).to(dev)

    trL = DataLoader(DS(tr), batch_size=cfg["train"]["batch"], shuffle=True, num_workers=2)
    vaL = DataLoader(DS(va), batch_size=cfg["train"]["batch"])
    ce = torch.nn.CrossEntropyLoss(weight=cw.to(dev))
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    optT = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"])
    optS = torch.optim.AdamW(student.parameters(), lr=cfg["train"]["lr"])

    def enc_feats(x):
        out = enc(pixel_values=x)
        return out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]

    for ep in range(cfg["train"]["epochs"]):
        for x, y in trL:
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad(): f = enc_feats(x)
            tlog = head(f); optT.zero_grad(); ce(tlog, y).backward(); optT.step()
            slog = student(x)
            loss = 0.5 * ce(slog, y) + 0.5 * kl(torch.log_softmax(slog, 1),
                                                torch.softmax(tlog.detach(), 1))
            optS.zero_grad(); loss.backward(); optS.step()
        acc = _eval_acc(student, vaL, dev)
        print(f"router ep{ep} student val_acc={acc:.3f}")
    import os, json
    os.makedirs("runs/router", exist_ok=True)
    torch.save(student.state_dict(), "runs/router/student.pt")
    json.dump(labels, open("runs/router/labels.json", "w"))
    return "runs/router/student.pt"


def _eval_acc(model, loader, dev):
    import torch
    model.eval(); n = c = 0
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(dev)).argmax(1).cpu()
            c += (p == y).sum().item(); n += len(y)
    model.train(); return c / max(n, 1)


if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); train(yaml.safe_load(open(a.config)))
```

- [ ] **Step 4: Run to verify pure tests pass**

Run: `pytest tests/test_train_router.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/train/train_router.py tests/test_train_router.py
git commit -m "feat(router): RAD-DINO teacher + MobileNetV3 student distillation trainer"
```

### Task B3: router notebook + train run (experiment gate)

**Files:**
- Create: `notebooks/kaggle_router_build.ipynb` (thin: `env.setup()` → `build_router_manifest.main` → `train_router.train` → CoreML export)
- Add to `Makefile`: `prep-router`, `train-router`.

- [ ] **Step 1:** Author the notebook mirroring `kaggle_coronary_build.ipynb` structure (clone repo to `/kaggle/tmp`, pip install `timm`, attach the modality datasets + a chest-xray sample for `other_xray`, run manifest → train).
- [ ] **Step 2:** Add Makefile targets:

```make
prep-router:
	$(PY) -m src.data_prep.build_router_manifest --config configs/router.yaml
train-router:
	$(PY) -m src.train.train_router --config configs/router.yaml
```

- [ ] **Step 3 (GATE):** Run on Kaggle GPU. Accept the student only if **val accuracy ≥ 0.90 overall AND per-class recall ≥ 0.85 on the reject buckets** (`other_xray`, `non_medical`) — a leaky reject bucket is what lets a wrong image reach a disease model. If reject recall is low, add more negative variety and re-run.
- [ ] **Step 4:** Archive `experiments/router_radino_mnv3/RESULTS.md` (per-class confusion + accepted thresholds). Commit notebook + RESULTS.md.

### Task B4: `ModalityRouter` class wrapper (thin, lazy)

**Files:**
- Modify: `src/serve/router.py` (append the class)
- Test: `tests/test_router.py` (append a wrapper test that monkeypatches the model)

**Interfaces:**
- Consumes: `decide_modality` (B0).
- Produces: `ModalityRouter(weights, labels, thresholds, size).classify(frame_gray) -> ModalityDecision` — consumed by the orchestrator (Phase D).

- [ ] **Step 1: Write the failing test (inject a fake forward so no torch needed)**

```python
# tests/test_router.py  (append)
from src.serve.router import ModalityRouter

def test_router_classify_uses_decide_modality(monkeypatch):
    r = ModalityRouter.__new__(ModalityRouter)          # bypass __init__ (no model load)
    r.labels = ["coronary_angiography", "other_xray"]
    r.thresholds = {"keep_thr": 0.60, "margin": 0.15, "quality_thr": 0.5}
    r.size = 224
    monkeypatch.setattr(r, "_probs", lambda frame: {"coronary_angiography": 0.92, "other_xray": 0.08})
    d = r.classify(frame=object())
    assert d.modality == "coronary_angiography" and d.deferred is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_router.py::test_router_classify_uses_decide_modality -v`
Expected: FAIL — `AttributeError: ... 'ModalityRouter'` / no such class.

- [ ] **Step 3: Append implementation to `src/serve/router.py`**

```python
class ModalityRouter:
    """Edge modality classifier. Lazy-loads the distilled MobileNetV3 student; `classify` returns a
    keep/defer ModalityDecision. Torch is imported only in `_probs`."""
    def __init__(self, weights, labels, thresholds=None, size=224):
        self.weights, self.labels, self.size = weights, labels, size
        self.thresholds = thresholds or {"keep_thr": 0.60, "margin": 0.15, "quality_thr": 0.5}
        self._model = None

    def _load(self):
        import torch, timm
        m = timm.create_model("mobilenetv3_small_100", num_classes=len(self.labels), in_chans=3)
        m.load_state_dict(torch.load(self.weights, map_location="cpu")); m.eval()
        return m

    def _probs(self, frame):
        import torch, cv2, numpy as np
        from src.data_prep.preprocess import clahe_unsharp
        if self._model is None:
            self._model = self._load()
        x = cv2.resize(clahe_unsharp(frame), (self.size, self.size)).astype("float32") / 255.0
        t = torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1)
        with torch.no_grad():
            p = torch.softmax(self._model(t), 1).squeeze(0).tolist()
        return {l: float(pi) for l, pi in zip(self.labels, p)}

    def classify(self, frame):
        return decide_modality(self._probs(frame),
                               keep_thr=self.thresholds["keep_thr"],
                               margin=self.thresholds["margin"],
                               quality_thr=self.thresholds["quality_thr"])
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_router.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/serve/router.py tests/test_router.py
git commit -m "feat(router): ModalityRouter edge wrapper over decide_modality"
```

---

# PHASE C — Diagnosis / Finding layer + report contract

**Goal:** the output contract (`Finding`, `StudyReport`) and the mapping from raw model output + existing triage into typed findings. All pure — no GPU.

### Task C1: `report.py` — `Finding` + `StudyReport` with JSON serialization

**Files:**
- Create: `src/serve/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces: `Finding`, `StudyReport` (+ `.to_dict()`) — consumed by diagnosis (C2), orchestrator (D), serving (E).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import json
from src.serve.report import Finding, StudyReport

def test_finding_defaults():
    f = Finding(label="coronary_stenosis", display_name="Possible coronary artery stenosis",
                confidence=0.8, deferred=False, reason="confident")
    assert f.severity is None and f.boxes == []

def test_studyreport_to_dict_is_json_safe():
    f = Finding("coronary_stenosis", "Possible coronary artery stenosis", 0.8, False, "confident",
                boxes=[(1, 2, 3, 4, 0.8)])
    r = StudyReport(modality="coronary_angiography", view=None, quality_ok=True, findings=[f],
                    deferred=False, defer_reason="", frames_analyzed=1,
                    model_versions={"coronary_stenosis": "best.pt"})
    d = r.to_dict()
    s = json.dumps(d)                    # must not raise (tuples -> lists)
    assert json.loads(s)["findings"][0]["boxes"][0] == [1, 2, 3, 4, 0.8]
    assert d["deferred"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/serve/report.py
"""Output contract for the diagnostic orchestrator. A StudyReport is what a clinician sees: modality,
per-finding screening flags with calibrated confidence, and an explicit study-level defer. `to_dict`
is JSON-safe (box tuples -> lists) for the /analyze endpoint."""
from dataclasses import dataclass, field, asdict


@dataclass
class Finding:
    label: str
    display_name: str
    confidence: float
    deferred: bool
    reason: str
    severity: str | None = None
    boxes: list = field(default_factory=list)


@dataclass
class StudyReport:
    modality: str
    view: str | None
    quality_ok: bool
    findings: list
    deferred: bool
    defer_reason: str
    frames_analyzed: int
    model_versions: dict

    def to_dict(self):
        d = asdict(self)
        for f in d["findings"]:
            f["boxes"] = [list(b) for b in f["boxes"]]     # tuples -> lists for json
        return d
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/serve/report.py tests/test_report.py
git commit -m "feat(serve): StudyReport/Finding output contract with json serialization"
```

### Task C2: `registry.py` — modality → model + finding metadata

**Files:**
- Create: `src/serve/registry.py`, `configs/orchestrator.yaml`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces: `TaskEntry`, `load_registry(path)`, `resolve(registry, modality)` — consumed by orchestrator (D).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
from src.serve.registry import load_registry, resolve, TaskEntry

def test_load_and_resolve(tmp_path):
    y = tmp_path / "orch.yaml"
    y.write_text(
        "modalities:\n"
        "  coronary_angiography:\n"
        "    task: det\n"
        "    model_path: runs/stenosis/best.pt\n"
        "    display_name: Coronary angiography\n"
        "    finding_label: coronary_stenosis\n"
        "    finding_display: Possible coronary artery stenosis (blockage)\n"
        "    floor_ok: false\n"
    )
    reg = load_registry(str(y))
    e = resolve(reg, "coronary_angiography")
    assert isinstance(e, TaskEntry) and e.task == "det" and e.floor_ok is False
    assert e.finding_label == "coronary_stenosis"

def test_resolve_unknown_returns_none(tmp_path):
    y = tmp_path / "orch.yaml"; y.write_text("modalities: {}\n")
    assert resolve(load_registry(str(y)), "cerebral_dsa") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/serve/registry.py
"""Modality -> task-model + finding metadata registry, loaded from configs/orchestrator.yaml.
`floor_ok=False` means the model exists but is below its accuracy floor -> the orchestrator surfaces
its finding as a deferred screening flag, never as a confident positive."""
from dataclasses import dataclass
import yaml


@dataclass
class TaskEntry:
    modality: str
    task: str
    model_path: str
    display_name: str
    finding_label: str
    finding_display: str
    floor_ok: bool = False


def load_registry(path):
    cfg = yaml.safe_load(open(path)) or {}
    reg = {}
    for mod, d in (cfg.get("modalities") or {}).items():
        reg[mod] = TaskEntry(modality=mod, task=d["task"], model_path=d["model_path"],
                             display_name=d["display_name"], finding_label=d["finding_label"],
                             finding_display=d["finding_display"], floor_ok=bool(d.get("floor_ok", False)))
    return reg


def resolve(registry, modality):
    return registry.get(modality)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write `configs/orchestrator.yaml` and commit**

```yaml
# configs/orchestrator.yaml — the registry the orchestrator loads
router:
  weights: runs/router/student.pt
  labels: [coronary_angiography, cerebral_dsa, other_xray, non_medical]
  thresholds: {keep_thr: 0.60, margin: 0.15, quality_thr: 0.5}
modalities:
  coronary_angiography:
    task: det
    model_path: runs/stenosis/arcade+cadica+danilov_yolo11s_768_e150/base/weights/best.pt
    display_name: Coronary angiography
    finding_label: coronary_stenosis
    finding_display: Possible coronary artery stenosis (blockage) — clinician review required
    floor_ok: false        # flip to true ONLY when Phase A clears F1>=0.57 recall>=0.60
  # cerebral_dsa / tavr_ct / avf: add here as their models clear their floors (Phase F extension)
```

```bash
git add src/serve/registry.py tests/test_registry.py configs/orchestrator.yaml
git commit -m "feat(serve): modality->model registry + orchestrator config"
```

### Task C3: `diagnosis.py` — raw output + triage → typed findings + study defer

**Files:**
- Create: `src/serve/diagnosis.py`
- Test: `tests/test_diagnosis.py`

**Interfaces:**
- Consumes: `TaskEntry` (C2), `Finding` (C1), `triage_decision` output shape (existing).
- Produces: `det_to_findings(entry, triage)`, `seg_to_finding(entry, seg_res)`, `study_defer(decision, findings)` — consumed by orchestrator (D).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnosis.py
from src.serve.registry import TaskEntry
from src.serve.diagnosis import det_to_findings, seg_to_finding, study_defer
from src.serve.router import ModalityDecision

ENTRY = TaskEntry("coronary_angiography", "det", "best.pt", "Coronary angiography",
                  "coronary_stenosis", "Possible coronary artery stenosis", floor_ok=True)

def test_confident_detection_becomes_kept_finding():
    triage = {"prediction": [(0, 0, 10, 10, 0.9)], "calibrated_confs": [0.9],
              "deferred": False, "reason": "confident"}
    fs = det_to_findings(ENTRY, triage)
    assert len(fs) == 1
    assert fs[0].label == "coronary_stenosis" and fs[0].deferred is False
    assert fs[0].boxes == [(0, 0, 10, 10, 0.9)] and fs[0].confidence == 0.9

def test_below_floor_entry_forces_defer_even_if_triage_confident():
    below = TaskEntry("coronary_angiography", "det", "best.pt", "Coronary angiography",
                      "coronary_stenosis", "Possible coronary artery stenosis", floor_ok=False)
    triage = {"prediction": [(0, 0, 10, 10, 0.95)], "calibrated_confs": [0.95],
              "deferred": False, "reason": "confident"}
    fs = det_to_findings(below, triage)
    assert fs[0].deferred is True and fs[0].reason == "below-floor"

def test_clean_negative_is_a_nonpositive_kept_finding():
    triage = {"prediction": [], "calibrated_confs": [], "deferred": False, "reason": "clean"}
    fs = det_to_findings(ENTRY, triage)
    assert fs[0].confidence == 0.0 and fs[0].deferred is False and fs[0].reason == "clean"

def test_study_defer_true_when_any_finding_deferred():
    dec = ModalityDecision("coronary_angiography", None, True, 0.9, False, "confident")
    triage = {"prediction": [(0, 0, 5, 5, 0.5)], "calibrated_confs": [0.5],
              "deferred": True, "reason": "low-confidence"}
    fs = det_to_findings(ENTRY, triage)
    deferred, reason = study_defer(dec, fs)
    assert deferred is True and reason == "low-confidence"

def test_study_defer_true_when_modality_unknown():
    dec = ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain")
    deferred, reason = study_defer(dec, [])
    assert deferred is True and reason == "router-uncertain"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_diagnosis.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/serve/diagnosis.py
"""Turn raw model output + the existing triage-with-abstention into typed Findings, and compute the
study-level defer. A below-floor model (entry.floor_ok=False) ALWAYS defers its finding, regardless
of how confident the detector was — a model that hasn't cleared its own bar cannot assert a positive."""
from src.serve.report import Finding


def det_to_findings(entry, triage):
    """Detector triage dict -> [Finding]. One finding per modality (the anchor disease), carrying the
    kept boxes. Below-floor forces deferred + reason 'below-floor'."""
    preds = triage.get("prediction") or []
    confs = triage.get("calibrated_confs") or []
    top = max(confs) if confs else 0.0
    deferred = bool(triage.get("deferred")) or (not entry.floor_ok)
    reason = "below-floor" if not entry.floor_ok else triage.get("reason", "clean")
    return [Finding(label=entry.finding_label, display_name=entry.finding_display,
                    confidence=float(top), deferred=deferred, reason=reason,
                    boxes=[tuple(p) for p in preds])]


def seg_to_finding(entry, seg_res):
    """Segmentation result dict -> a single anatomy/finding entry. Seg is context, not a positive Dx,
    so it never asserts disease; it defers if the model deferred (low confidence)."""
    deferred = bool(seg_res.get("deferred")) or (not entry.floor_ok)
    reason = "below-floor" if not entry.floor_ok else ("low-confidence" if deferred else "confident")
    return Finding(label=entry.finding_label, display_name=entry.finding_display,
                   confidence=float(seg_res.get("confidence", 0.0)), deferred=deferred, reason=reason)


def study_defer(decision, findings):
    """Study defers if the router deferred/modality unknown, or ANY finding deferred. Returns the
    most-fundamental reason first (router distrust > per-finding distrust)."""
    if decision.deferred or decision.modality in ("unknown", ""):
        return True, decision.reason
    for f in findings:
        if f.deferred:
            return True, f.reason
    return False, ""
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_diagnosis.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/serve/diagnosis.py tests/test_diagnosis.py
git commit -m "feat(serve): diagnosis layer — triage->findings + study defer, below-floor guard"
```

---

# PHASE D — Orchestrator (router → registry → model → diagnosis → report)

**Goal:** the glue. `analyze_frame` routes one frame and produces a `StudyReport`; `analyze_video` samples a clip, routes+infers per frame, and aggregates detections with the existing `temporal_vote.aggregate_sequence`. Models are injected via `model_factory` so the whole thing is testable with fakes (no torch).

### Task D1: `DiagnosticOrchestrator.analyze_frame`

**Files:**
- Create: `src/serve/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `ModalityRouter.classify` (B4), `resolve` (C2), `det_to_findings`/`seg_to_finding`/`study_defer` (C3), `triage_decision` (existing), `StudyReport` (C1).
- Produces: `DiagnosticOrchestrator(router, registry, model_factory, cfg)`, `.analyze_frame(frame_gray) -> StudyReport`.

- [ ] **Step 1: Write the failing test (fakes for router + model)**

```python
# tests/test_orchestrator.py
from src.serve.registry import TaskEntry
from src.serve.router import ModalityDecision
from src.serve.orchestrator import DiagnosticOrchestrator

class FakeRouter:
    def __init__(self, decision): self.d = decision
    def classify(self, frame): return self.d

def _reg(floor_ok):
    return {"coronary_angiography": TaskEntry(
        "coronary_angiography", "det", "best.pt", "Coronary angiography",
        "coronary_stenosis", "Possible coronary artery stenosis", floor_ok=floor_ok)}

def _det_factory(boxes):
    def factory(entry):
        return lambda frame: {"boxes": boxes, "top_conf": max([b[4] for b in boxes], default=0.0),
                              "deferred": False}
    return factory

def test_confident_coronary_frame_reports_kept_finding():
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([(0, 0, 9, 9, 0.9)]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.modality == "coronary_angiography" and r.deferred is False
    assert r.findings[0].label == "coronary_stenosis" and r.findings[0].deferred is False

def test_unknown_modality_defers_with_no_disease_finding():
    router = FakeRouter(ModalityDecision("unknown", None, True, 0.4, True, "router-uncertain"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([(0, 0, 9, 9, 0.9)]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.defer_reason == "router-uncertain" and r.findings == []

def test_supported_modality_with_no_registry_entry_defers_unsupported():
    router = FakeRouter(ModalityDecision("cerebral_dsa", None, True, 0.9, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), _det_factory([]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.defer_reason == "unsupported-modality"

def test_below_floor_model_defers_finding():
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=False), _det_factory([(0, 0, 9, 9, 0.95)]))
    r = orch.analyze_frame(frame_gray=object())
    assert r.deferred is True and r.findings[0].reason == "below-floor"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/serve/orchestrator.py
"""Diagnostic orchestrator: route a frame/clip -> the right task model -> typed findings -> StudyReport.
Models are injected via `model_factory(entry) -> callable(frame)->dict` so the control flow is unit-
tested with fakes (no torch). Safety: unknown/unsupported modality or below-floor model -> DEFER."""
from src.serve.report import StudyReport
from src.serve.registry import resolve
from src.serve.diagnosis import det_to_findings, seg_to_finding, study_defer


class DiagnosticOrchestrator:
    def __init__(self, router, registry, model_factory, cfg=None):
        self.router, self.registry, self.model_factory, self.cfg = router, registry, model_factory, cfg or {}
        self._models = {}

    def _model_for(self, entry):
        if entry.modality not in self._models:
            self._models[entry.modality] = self.model_factory(entry)
        return self._models[entry.modality]

    def _report(self, decision, findings, frames, versions):
        deferred, reason = study_defer(decision, findings)
        return StudyReport(modality=decision.modality, view=decision.view,
                           quality_ok=decision.quality_ok, findings=findings,
                           deferred=deferred, defer_reason=reason, frames_analyzed=frames,
                           model_versions=versions)

    def analyze_frame(self, frame_gray):
        dec = self.router.classify(frame_gray)
        versions = {"router": getattr(self.router, "weights", "router")}
        if dec.deferred or dec.modality in ("unknown", ""):
            return self._report(dec, [], 1, versions)
        entry = resolve(self.registry, dec.modality)
        if entry is None:                                    # supported class, no wired model
            dec.deferred, dec.reason = True, "unsupported-modality"
            return self._report(dec, [], 1, versions)
        versions[entry.finding_label] = entry.model_path
        out = self._model_for(entry)(frame_gray)
        if entry.task == "det":
            from src.serve.stenosis_triage import triage_decision
            triage = triage_decision(out.get("boxes", []),
                                     temperature=self.cfg.get("temperature", 1.0))
            findings = det_to_findings(entry, triage)
        else:
            findings = [seg_to_finding(entry, out)]
        return self._report(dec, findings, 1, versions)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/serve/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(serve): DiagnosticOrchestrator.analyze_frame — route->infer->report with defer"
```

### Task D2: `analyze_video` — sample, per-frame route, temporal-vote aggregate

**Files:**
- Modify: `src/serve/orchestrator.py` (add `analyze_video` + a helper)
- Test: `tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `aggregate_sequence(frames, iou_thr, min_hits, conf_agg)` (existing `src/serve/temporal_vote.py`).
- Produces: `analyze_video(path, stride, max_frames) -> StudyReport`.

- [ ] **Step 1: Write the failing test (inject a frame iterator so no video file needed)**

```python
# tests/test_orchestrator.py  (append)
from src.serve.router import ModalityDecision
from src.serve.orchestrator import DiagnosticOrchestrator

def test_video_majority_modality_and_temporal_vote(monkeypatch):
    # 5 coronary frames; a stenosis box present in >=2 -> aggregate keeps it, single-frame flicker dropped
    router = FakeRouter(ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident"))
    per_frame_boxes = [
        [(10, 10, 30, 30, 0.8)],           # f0 real
        [(10, 10, 30, 30, 0.82)],          # f1 real (2 hits -> kept)
        [],                                # f2
        [(200, 200, 210, 210, 0.4)],       # f3 flicker (1 hit -> dropped by min_hits=2)
        [(11, 11, 31, 31, 0.79)],          # f4 real
    ]
    calls = {"i": 0}
    def factory(entry):
        def model(frame):
            b = per_frame_boxes[calls["i"]]; calls["i"] += 1
            return {"boxes": b, "top_conf": max([x[4] for x in b], default=0.0), "deferred": False}
        return model
    orch = DiagnosticOrchestrator(router, _reg(floor_ok=True), factory)
    monkeypatch.setattr(orch, "_iter_frames", lambda path, stride, max_frames: iter([object()] * 5))
    r = orch.analyze_video("clip.mp4")
    assert r.modality == "coronary_angiography" and r.frames_analyzed == 5
    assert r.findings[0].label == "coronary_stenosis"
    # the flicker box (only 1 hit) must not survive aggregation into a kept detection
    assert all(not (abs(b[0] - 200) < 5) for b in r.findings[0].boxes)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_video_majority_modality_and_temporal_vote -v`
Expected: FAIL — `AttributeError: ... 'analyze_video'`.

- [ ] **Step 3: Append implementation to `src/serve/orchestrator.py`**

```python
    def _iter_frames(self, path, stride, max_frames):
        """Yield grayscale frames sampled every `stride` up to `max_frames`. cv2 imported lazily."""
        import cv2
        cap = cv2.VideoCapture(path)
        i = kept = 0
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                kept += 1
            i += 1
        cap.release()

    def analyze_video(self, path, stride=5, max_frames=400):
        frames = list(self._iter_frames(path, stride, max_frames))
        if not frames:
            from src.serve.router import ModalityDecision
            dec = ModalityDecision("unknown", None, True, 0.0, True, "router-uncertain")
            return self._report(dec, [], 0, {"router": getattr(self.router, "weights", "router")})
        decisions = [self.router.classify(f) for f in frames]
        kept = [d for d in decisions if not d.deferred and d.modality not in ("unknown", "")]
        versions = {"router": getattr(self.router, "weights", "router")}
        if not kept:                                          # no frame confidently routed -> defer whole study
            return self._report(decisions[0], [], len(frames), versions)
        # majority modality across confidently-routed frames
        counts = {}
        for d in kept:
            counts[d.modality] = counts.get(d.modality, 0) + 1
        modality = max(counts, key=counts.get)
        dec = next(d for d in kept if d.modality == modality)
        entry = resolve(self.registry, modality)
        if entry is None:
            dec.deferred, dec.reason = True, "unsupported-modality"
            return self._report(dec, [], len(frames), versions)
        versions[entry.finding_label] = entry.model_path
        model = self._model_for(entry)
        if entry.task == "det":
            from src.serve.temporal_vote import aggregate_sequence
            from src.serve.stenosis_triage import triage_decision
            seq = [model(f).get("boxes", []) for f in frames]         # per-frame detections
            voted = aggregate_sequence(seq, iou_thr=self.cfg.get("iou_thr", 0.3),
                                       min_hits=self.cfg.get("min_hits", 2),
                                       conf_agg=self.cfg.get("conf_agg", "mean"))
            triage = triage_decision(voted, temperature=self.cfg.get("temperature", 1.0))
            findings = det_to_findings(entry, triage)
        else:
            # seg: run on the modality's representative (highest-confidence) frame
            best_frame = frames[decisions.index(dec)] if dec in decisions else frames[0]
            findings = [seg_to_finding(entry, model(best_frame))]
        return self._report(dec, findings, len(frames), versions)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/serve/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(serve): analyze_video — per-frame route + temporal-vote aggregation"
```

### Task D3: real `model_factory` binding router + task models (integration wiring)

**Files:**
- Modify: `src/serve/orchestrator.py` (add module-level `build_orchestrator(cfg_path)` factory)
- Test: `tests/test_orchestrator.py` (append — assert wiring resolves paths; still no model load, monkeypatch loaders)

**Interfaces:**
- Produces: `build_orchestrator(cfg_path) -> DiagnosticOrchestrator` using real `ModalityRouter` + real `DetModel`/`SegModel` from `src.serve.infer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py  (append)
from src.serve import orchestrator as O

def test_build_orchestrator_wires_router_and_registry(tmp_path, monkeypatch):
    cfg = tmp_path / "orch.yaml"
    cfg.write_text(
        "router: {weights: runs/router/student.pt, labels: [coronary_angiography, other_xray],\n"
        "         thresholds: {keep_thr: 0.6, margin: 0.15, quality_thr: 0.5}}\n"
        "modalities:\n"
        "  coronary_angiography: {task: det, model_path: best.pt, display_name: Coronary,\n"
        "    finding_label: coronary_stenosis, finding_display: Possible stenosis, floor_ok: false}\n")
    monkeypatch.setattr(O, "_load_det", lambda p: (lambda f: {"boxes": [], "deferred": False}))
    monkeypatch.setattr(O, "_load_seg", lambda p: (lambda f: {"deferred": False, "confidence": 0.0}))
    orch = O.build_orchestrator(str(cfg))
    assert "coronary_angiography" in orch.registry
    assert orch.router.labels == ["coronary_angiography", "other_xray"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_build_orchestrator_wires_router_and_registry -v`
Expected: FAIL — `AttributeError: ... 'build_orchestrator'`.

- [ ] **Step 3: Append implementation to `src/serve/orchestrator.py`**

```python
def _load_det(model_path):
    from src.serve.infer import DetModel
    m = DetModel(model_path)
    return lambda frame: m(frame)


def _load_seg(model_path):
    from src.serve.infer import SegModel
    m = SegModel(model_path)
    return lambda frame: m(frame)


def build_orchestrator(cfg_path):
    """Wire a real orchestrator from configs/orchestrator.yaml: ModalityRouter + DetModel/SegModel."""
    import yaml
    from src.serve.router import ModalityRouter
    from src.serve.registry import load_registry
    cfg = yaml.safe_load(open(cfg_path)) or {}
    rc = cfg["router"]
    router = ModalityRouter(rc["weights"], rc["labels"], rc.get("thresholds"))
    registry = load_registry(cfg_path)

    def model_factory(entry):
        return _load_det(entry.model_path) if entry.task == "det" else _load_seg(entry.model_path)

    return DiagnosticOrchestrator(router, registry, model_factory, cfg=cfg.get("runtime", {}))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/serve/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(serve): build_orchestrator factory wiring router+registry+infer models"
```

---

# PHASE E — Serving: `POST /analyze` for image or video

**Goal:** one HTTP endpoint a clinician-facing app calls with an uploaded image or clip; returns the `StudyReport` JSON.

### Task E1: `/analyze` endpoint

**Files:**
- Modify: `src/serve/app.py`
- Test: `tests/test_analyze_endpoint.py` (uses FastAPI `TestClient`; orchestrator monkeypatched to a fake so no models load)

**Interfaces:**
- Consumes: `build_orchestrator` (D3), `StudyReport.to_dict` (C1).
- Produces: `POST /analyze` (multipart file + optional `?kind=image|video`) → `StudyReport` dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyze_endpoint.py
import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
import src.serve.app as app_mod
from src.serve.report import StudyReport, Finding

class FakeOrch:
    def analyze_frame(self, frame):
        return StudyReport("coronary_angiography", None, True,
                           [Finding("coronary_stenosis", "Possible stenosis", 0.0, True, "below-floor")],
                           True, "below-floor", 1, {"router": "r"})
    def analyze_video(self, path, **kw):
        return StudyReport("coronary_angiography", None, True, [], True, "router-uncertain", 0, {"router": "r"})

def test_analyze_image_returns_report(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "_orch", FakeOrch())
    monkeypatch.setattr(app_mod, "_decode_image", lambda raw: object())
    c = TestClient(app_mod.app)
    resp = c.post("/analyze?kind=image", files={"file": ("f.png", b"x", "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deferred"] is True and body["defer_reason"] == "below-floor"
    assert body["findings"][0]["label"] == "coronary_stenosis"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_analyze_endpoint.py -v`
Expected: FAIL — no `/analyze` route / `_orch` attr.

- [ ] **Step 3: Add to `src/serve/app.py`** (append inside the `if FastAPI is not None:` block; add lazy orchestrator singleton)

```python
    import os as _os
    from src.serve.orchestrator import build_orchestrator

    _orch = None
    _ORCH_CFG = _os.environ.get("ORCH_CONFIG", "configs/orchestrator.yaml")

    def _get_orch():
        global _orch
        if _orch is None:
            _orch = build_orchestrator(_ORCH_CFG)
        return _orch

    def _decode_image(raw):
        import cv2, numpy as np
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...), kind: str = "image"):
        raw = await file.read()
        orch = _orch or _get_orch()
        if kind == "video":
            import tempfile, os as _o
            fd, p = tempfile.mkstemp(suffix=_o.path.splitext(file.filename or "")[1] or ".mp4")
            with _o.fdopen(fd, "wb") as f:
                f.write(raw)
            try:
                return orch.analyze_video(p).to_dict()
            finally:
                _o.remove(p)
        return orch.analyze_frame(_decode_image(raw)).to_dict()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_analyze_endpoint.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Add Makefile target + commit**

```make
serve-analyze:            # ORCH_CONFIG=configs/orchestrator.yaml
	uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
```

```bash
git add src/serve/app.py tests/test_analyze_endpoint.py Makefile
git commit -m "feat(serve): POST /analyze endpoint (image|video) returning StudyReport"
```

### Task E2: export the router student to CoreML (edge artifact)

**Files:**
- Modify: `configs/edge_export.yaml` (register `runs/router/student.pt`), `Makefile`
- Reuse: `src/export/to_coreml.py` (state_dict → CoreML) — router student is a plain classifier, so add a thin `--arch mobilenetv3_small_100 --classes N` path if `to_coreml.py` assumes a seg net.

- [ ] **Step 1:** Inspect `src/export/to_coreml.py` — if it hardcodes `TinyUNet`, add a `--arch`/`--classes` branch that builds a `timm` classifier instead; keep the seg path default. (One conditional; TDD it if the file has pure helpers, else smoke-export on Mac.)
- [ ] **Step 2:** Add Makefile target:

```make
export-router-coreml:     # -> runs/router/student.mlpackage
	$(PY) -m src.export.to_coreml --weights runs/router/student.pt --arch mobilenetv3_small_100 --classes 4 --method palettize --nbits 6
```

- [ ] **Step 3:** On Mac, run it; confirm a `.mlpackage` is produced and `ModalityRouter` can be pointed at it (add a CoreML branch to `ModalityRouter._load` mirroring `infer._CoreMLBase`). Gate: router val accuracy on a held-out set within 1% of the torch student.
- [ ] **Step 4:** Commit config + Makefile + any export branch.

---

# PHASE F — Intended-use gate, tracker, and end-to-end verification

### Task F1: `docs/INTENDED_USE.md` — regulatory posture (blocking gate for any non-research use)

**Files:**
- Create: `docs/INTENDED_USE.md`
- Test: `tests/test_report.py` (append a copy-guard test)

- [ ] **Step 1:** Write `docs/INTENDED_USE.md` stating: **intended use = screening-with-abstention / second-read decision support**, NOT autonomous diagnosis; population + modality scope (coronary angiography today); the defer contract; the below-floor disclosure; that display copy must say "possible … — clinician review required"; and an explicit "not cleared for autonomous diagnostic use" line. Reference the accuracy floors per stage.
- [ ] **Step 2: Add a copy-guard test** so the product can't silently start asserting diagnoses:

```python
# tests/test_report.py  (append)
from src.serve.registry import load_registry

def test_finding_display_copy_never_asserts_autonomous_diagnosis():
    reg = load_registry("configs/orchestrator.yaml")
    for entry in reg.values():
        low = entry.finding_display.lower()
        assert "clinician review" in low or "review required" in low, entry.finding_display
        assert not low.startswith("diagnosis:"), entry.finding_display
```

- [ ] **Step 3: Run** `pytest tests/test_report.py -v` → PASS. **Commit.**

```bash
git add docs/INTENDED_USE.md tests/test_report.py
git commit -m "docs: intended-use gate (screening-with-abstention) + copy-guard test"
```

### Task F2: full-suite verification + tracker update

**Files:**
- Modify: `docs/PROJECT_TRACKER.md`

- [ ] **Step 1:** Run the whole suite: `pytest tests/ -q`. Expected: all prior tests plus the ~7 new test files pass; report the exact count. Fix any regression before proceeding (do not edit tests to pass).
- [ ] **Step 2:** Run the import-safety guard — every new `src/serve/*` and `src/train/train_router.py` must import on this laptop with no GPU deps:

```bash
python -c "import src.serve.report, src.serve.registry, src.serve.diagnosis, src.serve.router, src.serve.orchestrator, src.train.train_router, src.data_prep.build_router_manifest; print('import-safe OK')"
```

Expected: `import-safe OK` (no torch/transformers/timm imported at module load).
- [ ] **Step 3:** Add Stage 6 (Diagnostic Orchestrator) rows to `docs/PROJECT_TRACKER.md`: router trained + gated, registry, orchestrator, `/analyze`, intended-use gate; mark each with its real state.
- [ ] **Step 4:** End-to-end smoke (Mac, real models, once Phase A + router exist): point `ORCH_CONFIG` at the real registry, `make serve-analyze`, POST a known coronary frame and a known non-medical image; confirm the coronary frame returns a `coronary_stenosis` finding (deferred "below-floor" until Phase A clears) and the non-medical image returns study-level defer "unsupported-modality"/"router-uncertain". Record the two JSON responses in the tracker.
- [ ] **Step 5: Commit.**

```bash
git add docs/PROJECT_TRACKER.md
git commit -m "docs(tracker): add Stage 6 diagnostic orchestrator status"
```

---

## Self-Review (run against the vision before executing)

**Spec coverage** — the vision was "doctor uploads any image/video → model says which disease, else defer":
- *Accepts image OR video* → Phase E `/analyze` (`kind=image|video`), Phase D `analyze_frame`/`analyze_video`. ✅
- *Figures out what it's looking at* → Phase B router (`decide_modality` + `ModalityRouter`). ✅
- *Routes to the right model* → Phase C registry + Phase D orchestrator. ✅
- *Names the finding* → Phase C `Finding`/`StudyReport` + `diagnosis.py`. ✅
- *Defers when unsure* → margin/quality defer (B0), below-floor guard (C3), unknown/unsupported guard (D1), study-level defer (C3). ✅
- *Doesn't over-claim* → Phase F intended-use gate + copy-guard test. ✅
- *"Any" disease* → **explicitly scoped down**: only modalities with a floor-cleared model surface a finding; everything else defers. The reject buckets (`other_xray`, `non_medical`) exist precisely so "any image" fails safe. Extending coverage = add a modality row + a floor-cleared model (cerebral DSA, TAVR, AVF are future rows). This is the honest boundary, documented, not a gap.

**Placeholder scan:** no "TBD"/"add error handling"/"similar to Task N" — every code step has full code; every experiment step (Phase A, B3, E2) has an explicit accept/reject metric gate because those are metric-gated, not unit-testable.

**Type consistency:** `Finding`/`StudyReport` fields, `TaskEntry` fields, `ModalityDecision` fields, and `triage_decision`'s dict shape are used identically across C, D, E. `model_factory(entry) -> callable(frame)->dict` and `det_to_findings(entry, triage)` signatures match between D1 tests and C3 definition. `floor_ok` gate name is consistent registry↔diagnosis↔config.

**Known assumptions to verify during execution (not blockers):**
- RAD-DINO output attr (`pooler_output` vs `last_hidden_state[:,0]`) — handled with a fallback in B2; confirm on first run.
- `to_coreml.py` may assume `TinyUNet`; E2 Step 1 adds a classifier branch if so.
- `aggregate_sequence` expects per-frame lists of `(x,y,w,h,conf)`-style dets; confirm the box format matches `DetModel` output (xywh-normalized vs xyxy-pixel) and adapt in D2 if needed — the test pins behavior, so a format mismatch surfaces there.

---

## Execution notes specific to this repo

- **Build vs deploy:** Phases A, B2/B3 run on Kaggle/Colab GPU (thin notebooks import `src/*`). Phases B0/B1/C/D and all `pytest` run locally. Phase E2 + F4 smoke run on the Mac.
- **Frequent commits:** each task ends in a commit; experiment tasks commit the `RESULTS.md`, never weights (weights live on the GPU box / release assets).
- **Don't relax a safety gate to make a test pass.** If `study_defer` or the below-floor guard makes an integration test defer, that's usually correct — fix the test's expectation, not the guard.
