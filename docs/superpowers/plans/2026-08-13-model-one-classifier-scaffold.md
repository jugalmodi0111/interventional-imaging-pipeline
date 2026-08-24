# Model One AVF Classifier Scaffold — Implementation Plan

> **EXECUTED 2026-08-24 — suite 800 → 836. Three of this plan's assumptions were wrong; the code
> blocks below are the AS-WRITTEN plan, not the as-shipped repo. Read `src/` for what shipped.**
>
> 1. **Task 3's implementation fails Task 3's own test, and the failure is a leakage bug. DO NOT
>    COPY IT.** `group_key` is anchored against FRAME stems (`_AVF_RE`: `..._s\d+_\d+$`); the
>    labels-JSONL `key` is a SERIES stem, so `group_key(stem)` returns the **series**, splitting one
>    patient's two studies across train and val — the P0.2 / CADICA-2026-08-16 bug, third occurrence.
>    Shipped fix: reconstruct the frame stem via `src.ingest.extract.frame_stem` before grouping,
>    with a fallback to the series key so an unknown grammar never degrades to one group per frame.
> 2. `src.serve.router` was deleted 2026-08-16 — `ModalityDecision` is in `src/serve/validity.py`.
>    Tasks 6/7 test imports below are stale.
> 3. Task 6's `else: seg_to_finding(...)` fallthrough relabels a cls `"defer-band"` as seg's
>    `"low-confidence"`. Shipped as an explicit three-way branch, pinned by a test.
>
> Also: the "621 passed" baseline was stale (actual 800). `temperature_scale` returns a bare float,
> resolving the plan's flagged Task 4 risk. See PROJECT_TRACKER §4.7 + the 2026-08-24 changelog.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete code path for Dialygo Model One — a frozen-backbone binary classifier (normal vs. significant juxta-anastomotic stenosis) — training, classification metrics, calibrated+thresholded inference, and orchestrator wiring, all runnable and tested on synthetic data before any real frame exists.

**Architecture:** A frozen foundation backbone (`dinov2_vitb14` via timm at runtime; a tiny deterministic torch backbone for offline tests) feeds a trained linear head (B4). Training is patient-grouped (B5/B6) over a PNG frame store + labels JSONL (the shapes `src/ingest/` already produces), fits a temperature on val logits and selects an operating threshold at a target sensitivity (B3: never a false normal). Serving loads the head checkpoint into a torch `ClsModel` whose output flows through a new `cls_to_finding` into the existing `StudyReport`/defer machinery; the event bus mirrors it automatically.

**Tech Stack:** torch (installed, ≥2.2, CPU-sufficient for tests), timm (in requirements; lazily imported, never in tests), numpy, existing `src/eval/calibration.py`, pytest.

**Spec:** No single spec file. Binding sources, in precedence order: `docs/Dialygo_Orientation_and_Requirements.md` B1–B9 (esp. B3, B4, B5, B6), `configs/avf_fistulography.yaml` (all values quoted below), `docs/superpowers/plans/2026-08-03-audit-remediation-plan.md` §P3 "Build new" list.

## Global Constraints

- **No git commits by the implementer.** User's standing order. Every task ends at "suite green"; the user commits. (Commit steps are deliberately absent.)
- **B5:** no real patient data touched anywhere; tests use synthetic arrays/files only.
- **Offline tests:** no test may download weights. Real backbones (`dinov2_vitb14`, …) load only behind lazy imports at runtime; tests use the `"test-tiny"` backbone defined in Task 2.
- **Patient-level splits only (B5/B6):** grouping via `src.data_prep.io_utils.group_key`; a train/val patient overlap is an assertion failure, not a warning.
- **Floors are unset** (`target: {sensitivity: null, specificity: null}` in `configs/avf_fistulography.yaml`) — code must treat `None` as "floor not signed off": `floor_ok: false` in the registry entry, threshold falls back to 0.5 with an explicit `"threshold-unsigned"` marker in metrics.json.
- **Suite baseline: 621 passed** (`python -m pytest tests/ -q` from repo root). Every task leaves it green; each adds tests.
- **Import-safety:** `src/serve/*` and `src/eval/*` modules must import without torch/timm installed-or-loaded (lazy imports inside functions), matching the repo convention tested by the existing subprocess guardrails.
- Match repo style: plain functions + small dataclasses, docstrings that explain *why*, no type-annotation ceremony beyond what neighbors use.

## File Structure

| File | Responsibility |
|---|---|
| `src/eval/cls_metrics.py` (new) | Pure-numpy binary classification metrics: confusion counts, sensitivity, specificity, threshold selection at target sensitivity, bootstrap CIs. No torch. |
| `src/models/frozen_backbone.py` (new) | `make_backbone(name)` factory (timm-lazy; `"test-tiny"` offline backbone) + `FrozenBackboneClassifier` (frozen features → linear head). |
| `src/train/train_classifier.py` (new) | Dataset over frames dir + labels JSONL, patient-grouped split, head-only training loop, temperature fit, threshold selection, artifact writing (`head.pt`, `metrics.json`). CLI. |
| `src/serve/infer_cls.py` (new) | `ClsModel(path)` — hosted torch inference callable returning the dict contract the orchestrator consumes. |
| `src/serve/diagnosis.py` (modify) | Add `cls_to_finding(entry, cls_res)` — classifier output → `Finding`, fail-safe on malformed input and below-floor. |
| `src/serve/orchestrator.py` (modify) | `_model_factory` gains `task == "cls"` branch (`_load_cls`); `analyze_frame` routes cls findings through `cls_to_finding`. |
| `configs/orchestrator.yaml` (modify) | Add `avf_fistulography` modality entry (`task: cls`, `floor_ok: false`). |
| `Makefile` (modify) | `train-avf-cls` target. |
| Tests | `tests/test_cls_metrics.py`, `tests/test_frozen_backbone.py`, `tests/test_train_classifier.py`, `tests/test_infer_cls.py`, plus additions to `tests/test_diagnosis.py`-style file `tests/test_cls_finding.py` and `tests/test_orchestrator.py`. |

---

### Task 1: `src/eval/cls_metrics.py` — pure classification metrics

**Files:**
- Create: `src/eval/cls_metrics.py`
- Test: `tests/test_cls_metrics.py`

**Interfaces:**
- Consumes: numpy only. (`auroc`/`ece` already exist in `src/eval/calibration.py` — do NOT duplicate them here.)
- Produces (later tasks call these exactly):
  - `confusion_counts(probs, labels, thr) -> dict` with int keys `"tp","fp","tn","fn"`
  - `sensitivity(probs, labels, thr) -> float` (recall on positives; `0.0` when no positives)
  - `specificity(probs, labels, thr) -> float` (`0.0` when no negatives)
  - `threshold_at_sensitivity(probs, labels, target) -> float` — the HIGHEST threshold whose sensitivity ≥ target; `0.5` when `target is None`; `0.0` when even threshold 0 can't reach target
  - `bootstrap_ci(metric_fn, probs, labels, n_boot=1000, seed=0, alpha=0.05) -> (lo, hi)`

- [x] **Step 1: Write the failing tests**

```python
"""Binary classification metrics for Model One (B3). Pure numpy — no torch anywhere.

sensitivity/specificity are the B-requirement vocabulary (screening triage), deliberately not
precision/recall aliases. threshold_at_sensitivity implements B3's 'never a false normal' posture:
pick the operating point from a *sensitivity* target, then report the specificity you got.
"""
import numpy as np
import pytest

from src.eval.cls_metrics import (bootstrap_ci, confusion_counts, sensitivity, specificity,
                                  threshold_at_sensitivity)

PROBS = np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1])
LABELS = np.array([1, 1, 0, 1, 0, 0])


def test_confusion_counts_at_half():
    c = confusion_counts(PROBS, LABELS, 0.5)
    assert c == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}


def test_sensitivity_and_specificity_at_half():
    assert sensitivity(PROBS, LABELS, 0.5) == pytest.approx(2 / 3)
    assert specificity(PROBS, LABELS, 0.5) == pytest.approx(2 / 3)


def test_degenerate_inputs_return_zero_not_nan():
    assert sensitivity(PROBS, np.zeros(6), 0.5) == 0.0      # no positives to be sensitive to
    assert specificity(PROBS, np.ones(6), 0.5) == 0.0       # no negatives


def test_threshold_at_sensitivity_hits_target():
    thr = threshold_at_sensitivity(PROBS, LABELS, 1.0)      # must catch ALL positives
    assert sensitivity(PROBS, LABELS, thr) == 1.0
    assert thr <= 0.4                                        # the 0.4 positive must clear it


def test_threshold_none_target_falls_back_to_half():
    assert threshold_at_sensitivity(PROBS, LABELS, None) == 0.5


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    lo, hi = bootstrap_ci(sensitivity, PROBS, LABELS, n_boot=200, seed=7)
    lo2, hi2 = bootstrap_ci(sensitivity, PROBS, LABELS, n_boot=200, seed=7)
    assert (lo, hi) == (lo2, hi2)
    assert lo <= sensitivity(PROBS, LABELS, 0.5) <= hi
```

- [x] **Step 2: Run to verify failure** — `python -m pytest tests/test_cls_metrics.py -q` → collection error `No module named 'src.eval.cls_metrics'`.

- [x] **Step 3: Implement**

```python
"""Binary classification metrics for Model One triage (B3 vocabulary: sensitivity/specificity).

Pure numpy, no torch: importable by eval scripts, the trainer, and tests without the heavy stack.
AUROC and ECE deliberately live in src/eval/calibration.py already — import them from there, this
module only adds what the repo lacked (confusion vocabulary, operating-point selection, CIs).

Operating-point rule (B3 'low confidence defaults to uncertain, never false normal'): choose the
threshold FROM a sensitivity target — the highest cut that still catches the required fraction of
positives — then report the specificity that follows. A None target means the clinical floor is
not signed off (configs/avf_fistulography.yaml target: null); callers fall back to 0.5 and must
mark the result unsigned rather than invent a floor.
"""
import numpy as np


def confusion_counts(probs, labels, thr):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = probs >= thr
    pos = labels == 1
    return {"tp": int(np.sum(pred & pos)), "fp": int(np.sum(pred & ~pos)),
            "tn": int(np.sum(~pred & ~pos)), "fn": int(np.sum(~pred & pos))}


def sensitivity(probs, labels, thr):
    c = confusion_counts(probs, labels, thr)
    denom = c["tp"] + c["fn"]
    return c["tp"] / denom if denom else 0.0


def specificity(probs, labels, thr):
    c = confusion_counts(probs, labels, thr)
    denom = c["tn"] + c["fp"]
    return c["tn"] / denom if denom else 0.0


def threshold_at_sensitivity(probs, labels, target):
    """Highest threshold with sensitivity >= target, scanning candidate cuts at the observed
    positive probabilities (any threshold between two observed values behaves identically).
    None target -> 0.5 (floor unsigned); unreachable target -> 0.0 (call everything positive)."""
    if target is None:
        return 0.5
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    for thr in sorted(np.unique(probs[labels == 1]), reverse=True):
        if sensitivity(probs, labels, thr) >= target:
            return float(thr)
    return 0.0


def bootstrap_ci(metric_fn, probs, labels, n_boot=1000, seed=0, alpha=0.05, thr=0.5):
    """Percentile bootstrap CI for metric_fn(probs, labels, thr). Deterministic under a seed so
    metrics.json is reproducible run-to-run."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(probs)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(probs[idx], labels[idx], thr))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))
```

- [x] **Step 4: Run to verify pass** — `python -m pytest tests/test_cls_metrics.py -q` → 6 passed.
- [x] **Step 5: Full suite** — `python -m pytest tests/ -q` → 627 passed (621 + 6).

---

### Task 2: `src/models/frozen_backbone.py` — backbone factory + frozen classifier

**Files:**
- Create: `src/models/frozen_backbone.py`
- Test: `tests/test_frozen_backbone.py`

**Interfaces:**
- Consumes: torch (installed). timm ONLY inside `make_backbone` for non-test names.
- Produces:
  - `make_backbone(name, imgsz=224) -> (torch.nn.Module, int)` — module maps `[B,1,imgsz,imgsz]` float in [0,1] → `[B, feat_dim]`; returns `(module, feat_dim)`. `name="test-tiny"` = deterministic offline backbone (seeded conv+pool, feat_dim 32). Any other name: `import timm`, `timm.create_model(name, pretrained=True, num_classes=0, in_chans=1)`.
  - `FrozenBackboneClassifier(backbone_name, imgsz=224)` — torch module; `.backbone` frozen (`requires_grad=False`, eval mode), `.head = nn.Linear(feat_dim, 1)`; `forward(x) -> logits [B]`; `.trainable_parameters()` yields head params only.

- [x] **Step 1: Write the failing tests**

```python
"""Frozen-backbone classifier (B4): the backbone never trains, only the linear head does.
All tests use the 'test-tiny' backbone -- no timm import, no network, CPU-only."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.frozen_backbone import FrozenBackboneClassifier, make_backbone


def test_test_tiny_backbone_shape_and_determinism():
    b1, d1 = make_backbone("test-tiny", imgsz=32)
    b2, d2 = make_backbone("test-tiny", imgsz=32)
    x = torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(0))
    assert d1 == d2
    assert torch.equal(b1(x), b2(x))          # seeded init: same weights, same features
    assert b1(x).shape == (2, d1)


def test_backbone_is_frozen_and_head_is_trainable():
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    assert all(not p.requires_grad for p in m.backbone.parameters())
    trainable = list(m.trainable_parameters())
    assert trainable and all(p.requires_grad for p in trainable)
    assert {id(p) for p in trainable} == {id(p) for p in m.head.parameters()}


def test_forward_returns_one_logit_per_sample():
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    out = m(torch.rand(3, 1, 32, 32))
    assert out.shape == (3,)


def test_head_learns_while_backbone_stays_fixed():
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    before = [p.clone() for p in m.backbone.parameters()]
    x, y = torch.rand(8, 1, 32, 32), torch.tensor([0., 1.] * 4)
    opt = torch.optim.SGD(m.trainable_parameters(), lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), y)
        loss.backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, m.backbone.parameters()))
```

- [x] **Step 2: Run to verify failure** — `python -m pytest tests/test_frozen_backbone.py -q` → `No module named 'src.models.frozen_backbone'`.

- [x] **Step 3: Implement**

```python
"""Frozen foundation backbone + lightweight trained head (Dialygo B4).

The backbone is a feature extractor and NOTHING trains inside it -- sample efficiency on a small
institutional cohort is the fixed requirement, and a frozen backbone is how the design meets it.
Real backbones (dinov2_vitb14 default; dinov3/rad-dino/biomedclip bake-off candidates per
configs/avf_fistulography.yaml) come from timm, imported lazily so this module -- and every test --
works with timm absent. 'test-tiny' is a seeded, deterministic conv backbone for offline tests:
NEVER use it for a real run.
"""
import torch
import torch.nn as nn


def make_backbone(name, imgsz=224):
    """-> (module, feat_dim). Module: [B,1,H,W] float in [0,1] -> [B, feat_dim] features."""
    if name == "test-tiny":
        gen = torch.Generator().manual_seed(1234)
        conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        with torch.no_grad():
            conv.weight.copy_(torch.rand(conv.weight.shape, generator=gen) - 0.5)
            conv.bias.zero_()
        backbone = nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        return backbone, 32
    import timm                                    # lazy: only real runs pay for this
    model = timm.create_model(name, pretrained=True, num_classes=0, in_chans=1)
    feat_dim = model.num_features
    return model, feat_dim


class FrozenBackboneClassifier(nn.Module):
    def __init__(self, backbone_name, imgsz=224):
        super().__init__()
        self.backbone_name = backbone_name
        self.imgsz = imgsz
        self.backbone, feat_dim = make_backbone(backbone_name, imgsz)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.head = nn.Linear(feat_dim, 1)

    def trainable_parameters(self):
        return self.head.parameters()

    def train(self, mode=True):
        """Head follows train/eval; the frozen backbone stays in eval so norm layers never update."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.head(feats).squeeze(-1)
```

- [x] **Step 4: Run to verify pass** — 4 passed.
- [x] **Step 5: Full suite** — 631 passed.

---

### Task 3: `src/train/train_classifier.py` (part 1) — dataset + patient-grouped split

**Files:**
- Create: `src/train/train_classifier.py` (dataset/split half)
- Test: `tests/test_train_classifier.py` (dataset/split tests)

**Interfaces:**
- Consumes: `src.data_prep.io_utils.group_key` (collapses `avf_inu_<pid>_s01_00012` → `avf_inu_<pid>`); frame store layout `frames/<stem_prefix>/f%05d.png`; labels JSONL rows `{"key": <stem_prefix>, "label": 0|1}` (the shape `src.ingest.labels.write_labels_jsonl` emits via its `key`/`label` fields).
- Produces:
  - `load_examples(frames_root, labels_path) -> list[dict]` — one dict per FRAME: `{"path": str, "stem": str, "group": str, "label": int}`; stems present in labels but missing on disk are skipped with a count; frames with no label are skipped.
  - `grouped_split(examples, val_frac=0.2, seed=0) -> (train, val)` — split by `group`, never by frame; deterministic; raises `AssertionError` on any group overlap (self-check, B5).

- [x] **Step 1: Write the failing tests** (build a tiny synthetic frame store with PIL-free cv2 writes)

```python
"""Trainer scaffolding: dataset loading + the patient-grouped split (B5: split by patient, never
by frame). Synthetic frame store on tmp_path -- no real data, no network, CPU only."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.train.train_classifier import grouped_split, load_examples


def _store(tmp_path, stems_frames_labels):
    frames = tmp_path / "frames"
    rows = []
    for stem, n, label in stems_frames_labels:
        d = frames / stem
        d.mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(d / f"f{i:05d}.png"), np.full((32, 32), 60 + 60 * label, np.uint8))
        rows.append({"key": stem, "label": label})
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return frames, labels


def test_load_examples_one_row_per_frame_with_patient_group(tmp_path):
    frames, labels = _store(tmp_path, [("avf_inu_aaaaaaaaaa_s01", 3, 1),
                                       ("avf_inu_bbbbbbbbbb_s01", 2, 0)])
    ex = load_examples(frames, labels)
    assert len(ex) == 5
    assert {e["group"] for e in ex} == {"avf_inu_aaaaaaaaaa", "avf_inu_bbbbbbbbbb"}
    assert all(e["label"] in (0, 1) and e["path"].endswith(".png") for e in ex)


def test_load_examples_skips_labels_with_no_frames_on_disk(tmp_path):
    frames, labels = _store(tmp_path, [("avf_inu_aaaaaaaaaa_s01", 2, 1)])
    labels.write_text(labels.read_text() + json.dumps({"key": "avf_inu_gone_s01", "label": 0}) + "\n")
    assert len(load_examples(frames, labels)) == 2


def test_grouped_split_never_splits_a_patient(tmp_path):
    trips = [(f"avf_inu_{i:010x}_s01", 4, i % 2) for i in range(10)]
    frames, labels = _store(tmp_path, trips)
    train, val = grouped_split(load_examples(frames, labels), val_frac=0.3, seed=1)
    tg, vg = {e["group"] for e in train}, {e["group"] for e in val}
    assert tg and vg and not (tg & vg)


def test_grouped_split_is_deterministic(tmp_path):
    trips = [(f"avf_inu_{i:010x}_s01", 2, i % 2) for i in range(6)]
    frames, labels = _store(tmp_path, trips)
    ex = load_examples(frames, labels)
    a = grouped_split(ex, val_frac=0.5, seed=42)
    b = grouped_split(ex, val_frac=0.5, seed=42)
    assert [e["path"] for e in a[1]] == [e["path"] for e in b[1]]
```

- [x] **Step 2: Verify failure** — `No module named 'src.train.train_classifier'`.

- [x] **Step 3: Implement (module top + these two functions)**

```python
"""Train Model One: frozen-backbone binary classifier over the de-identified AVF frame store.

Consumes exactly what src/ingest/ produces: frames/<stem_prefix>/f%05d.png and a labels JSONL of
{"key": <stem_prefix>, "label": 0|1} rows (src.ingest.labels join output). Split is by PATIENT
(io_utils.group_key on the stem prefix) -- the F1 0.885->0.214 leakage incident is why this module
refuses to split any other way, and why group overlap is an assertion, not a log line (B5/B6).
"""
import hashlib
import json
import os
from pathlib import Path

from src.data_prep.io_utils import group_key
from src.ingest.manifest import read_jsonl


def load_examples(frames_root, labels_path):
    frames_root = Path(frames_root)
    examples, missing = [], 0
    for row in read_jsonl(labels_path):
        stem, label = row.get("key"), row.get("label")
        if stem is None or label not in (0, 1):
            continue
        d = frames_root / stem
        pngs = sorted(d.glob("f*.png")) if d.is_dir() else []
        if not pngs:
            missing += 1
            continue
        group = group_key(stem)
        for p in pngs:
            examples.append({"path": str(p), "stem": stem, "group": group, "label": int(label)})
    if missing:
        print(f"[train_cls] skipped {missing} labeled stems with no frames on disk")
    return examples


def grouped_split(examples, val_frac=0.2, seed=0):
    """Deterministic per-GROUP assignment: hash(group|seed) -> [0,1) < val_frac => val."""
    train, val = [], []
    for e in examples:
        h = hashlib.sha256(f"{e['group']}|{seed}".encode()).hexdigest()
        (val if int(h[:8], 16) / 0xFFFFFFFF < val_frac else train).append(e)
    overlap = {e["group"] for e in train} & {e["group"] for e in val}
    assert not overlap, f"patient groups in BOTH splits (leakage): {sorted(overlap)[:5]}"
    return train, val
```

- [x] **Step 4: Verify pass** — 4 passed.
- [x] **Step 5: Full suite green.**

---

### Task 4: `src/train/train_classifier.py` (part 2) — training loop, calibration, artifacts, CLI

**Files:**
- Modify: `src/train/train_classifier.py`
- Test: append to `tests/test_train_classifier.py`
- Modify: `Makefile` (add `train-avf-cls` target, mirroring the `# --- Dialygo ingest ---` section's variable style)

**Interfaces:**
- Consumes: Task 2 `FrozenBackboneClassifier`; Task 1 `threshold_at_sensitivity`, `sensitivity`, `specificity`, `bootstrap_ci`; `src.eval.calibration.temperature_scale(logits, labels) -> T` and `apply_temperature(logits, T) -> probs`; `src.eval.calibration.auroc`, `ece`.
- Produces:
  - `train(frames_root, labels_path, out_dir, *, backbone="test-tiny", imgsz=32, epochs=2, lr=1e-2, val_frac=0.2, seed=0, target_sensitivity=None) -> dict` (the metrics dict it also writes)
  - Artifacts in `out_dir`: `head.pt` (torch save of `{"backbone": name, "imgsz": int, "head_state": state_dict, "temperature": float, "threshold": float, "defer_band": [lo, hi]}`) and `metrics.json` (sens/spec/auroc/ece at the chosen threshold + `threshold`, `threshold_policy` = `"at-target-sensitivity"` or `"threshold-unsigned"`, bootstrap CIs, split sizes, group counts)
  - CLI: `python -m src.train.train_classifier --frames F --labels L --out O [--backbone B] [--imgsz N] [--epochs N] [--val-frac F] [--seed N] [--target-sensitivity F]`
- `defer_band` is copied verbatim from `configs/avf_fistulography.yaml` `defer.band` (`[0.3, 0.6]`) — read the YAML; if unreadable, default `[0.3, 0.6]` with a warning.

- [x] **Step 1: Write the failing tests** (tiny end-to-end run on `test-tiny`, separable classes so it learns)

```python
def test_train_end_to_end_writes_artifacts_and_learns(tmp_path):
    torch = pytest.importorskip("torch")
    from src.train.train_classifier import train
    trips = ([(f"avf_inu_{i:010x}_s01", 6, 1) for i in range(4)]
             + [(f"avf_inu_{i + 8:010x}_s01", 6, 0) for i in range(4)])
    frames, labels = _store(tmp_path, trips)      # label 1 -> bright frames, 0 -> dark (separable)
    m = train(frames, labels, tmp_path / "run", backbone="test-tiny", imgsz=32,
              epochs=8, val_frac=0.45, seed=3)
    assert (tmp_path / "run" / "head.pt").exists()
    assert (tmp_path / "run" / "metrics.json").exists()
    assert m["auroc"] >= 0.9                       # brightness is trivially separable
    assert m["threshold_policy"] == "threshold-unsigned"     # no target given (floor unsigned)
    ckpt = torch.load(tmp_path / "run" / "head.pt", weights_only=False)
    assert ckpt["backbone"] == "test-tiny" and 0 < ckpt["temperature"]
    assert ckpt["defer_band"] == [0.3, 0.6]


def test_train_with_target_sensitivity_selects_threshold_from_val(tmp_path):
    pytest.importorskip("torch")
    from src.train.train_classifier import train
    trips = ([(f"avf_inu_{i:010x}_s01", 6, 1) for i in range(4)]
             + [(f"avf_inu_{i + 8:010x}_s01", 6, 0) for i in range(4)])
    frames, labels = _store(tmp_path, trips)
    m = train(frames, labels, tmp_path / "run", backbone="test-tiny", imgsz=32,
              epochs=8, val_frac=0.45, seed=3, target_sensitivity=1.0)
    assert m["threshold_policy"] == "at-target-sensitivity"
    assert m["sensitivity"] == 1.0                 # by construction of the threshold rule


def test_cli_smoke(tmp_path, capsys):
    pytest.importorskip("torch")
    from src.train.train_classifier import main
    trips = [("avf_inu_aaaaaaaaaa_s01", 4, 1), ("avf_inu_bbbbbbbbbb_s01", 4, 0),
             ("avf_inu_cccccccccc_s01", 4, 1), ("avf_inu_dddddddddd_s01", 4, 0)]
    frames, labels = _store(tmp_path, trips)
    rc = main(["--frames", str(frames), "--labels", str(labels), "--out", str(tmp_path / "o"),
               "--backbone", "test-tiny", "--imgsz", "32", "--epochs", "2", "--val-frac", "0.5"])
    assert rc == 0 and (tmp_path / "o" / "metrics.json").exists()
```

- [x] **Step 2: Verify failure** — `ImportError: cannot import name 'train'`.

- [x] **Step 3: Implement** (append to module; key structure below — implementer writes it exactly)

```python
def _load_gray(path, imgsz):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    return img.astype("float32") / 255.0


def _tensorize(examples, imgsz):
    import numpy as np
    import torch
    x = np.stack([_load_gray(e["path"], imgsz) for e in examples])[:, None]   # [N,1,H,W]
    y = np.array([e["label"] for e in examples], dtype="float32")
    return torch.from_numpy(x), torch.from_numpy(y)


def _defer_band(cfg_path="configs/avf_fistulography.yaml"):
    try:
        import yaml
        with open(cfg_path) as f:
            return list(yaml.safe_load(f)["defer"]["band"])
    except Exception:
        print(f"[train_cls] warning: could not read defer.band from {cfg_path}; using [0.3, 0.6]")
        return [0.3, 0.6]


def train(frames_root, labels_path, out_dir, *, backbone="test-tiny", imgsz=32, epochs=2,
          lr=1e-2, val_frac=0.2, seed=0, target_sensitivity=None):
    import numpy as np
    import torch
    from src.eval.calibration import apply_temperature, auroc, ece, temperature_scale
    from src.eval.cls_metrics import (bootstrap_ci, sensitivity, specificity,
                                      threshold_at_sensitivity)
    from src.models.frozen_backbone import FrozenBackboneClassifier

    torch.manual_seed(seed)
    examples = load_examples(frames_root, labels_path)
    train_ex, val_ex = grouped_split(examples, val_frac=val_frac, seed=seed)
    assert train_ex and val_ex, "both splits must be non-empty (add patients or adjust val_frac)"
    xt, yt = _tensorize(train_ex, imgsz)
    xv, yv = _tensorize(val_ex, imgsz)

    model = FrozenBackboneClassifier(backbone, imgsz=imgsz)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(xv).numpy()
    labels_np = yv.numpy().astype(int)
    T = float(temperature_scale(val_logits, labels_np))
    probs = apply_temperature(val_logits, T)

    thr = threshold_at_sensitivity(probs, labels_np, target_sensitivity)
    policy = "threshold-unsigned" if target_sensitivity is None else "at-target-sensitivity"
    band = _defer_band()
    metrics = {
        "n_train_frames": len(train_ex), "n_val_frames": len(val_ex),
        "n_train_groups": len({e["group"] for e in train_ex}),
        "n_val_groups": len({e["group"] for e in val_ex}),
        "backbone": backbone, "temperature": T, "threshold": float(thr),
        "threshold_policy": policy,
        "sensitivity": sensitivity(probs, labels_np, thr),
        "specificity": specificity(probs, labels_np, thr),
        "auroc": float(auroc(probs, labels_np)), "ece": float(ece(probs, labels_np)),
        "sensitivity_ci": bootstrap_ci(sensitivity, probs, labels_np, n_boot=200,
                                       seed=seed, thr=thr),
        "specificity_ci": bootstrap_ci(specificity, probs, labels_np, n_boot=200,
                                       seed=seed, thr=thr),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone, "imgsz": imgsz, "head_state": model.head.state_dict(),
                "temperature": T, "threshold": float(thr), "defer_band": band}, out / "head.pt")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_cls] {json.dumps({k: metrics[k] for k in ('auroc', 'sensitivity', 'specificity', 'threshold_policy')})}")
    return metrics
```

CLI `main(argv=None)`: argparse mirroring the signature (`--target-sensitivity` default None → floor-unsigned path), calls `train`, returns 0. `if __name__ == "__main__": raise SystemExit(main())`. Makefile target:

```make
train-avf-cls:  ## Train Model One head (synthetic/test data until B5 clears real frames)
	$(PY) -m src.train.train_classifier --frames $(FRAMES) --labels $(LABELS) --out $(OUT) $(ARGS)
```

**Note on `temperature_scale`:** read its actual return in `src/eval/calibration.py:85` before wiring — if it returns `(T, …)` rather than bare `T`, unpack accordingly; the test asserting `0 < ckpt["temperature"]` catches a mis-wire.

- [x] **Step 4: Verify pass** — 3 more passed (7 total in file).
- [x] **Step 5: Full suite green.**

---

### Task 5: `src/serve/infer_cls.py` — hosted torch inference

**Files:**
- Create: `src/serve/infer_cls.py`
- Test: `tests/test_infer_cls.py`

**Interfaces:**
- Consumes: Task 4's `head.pt` schema; Task 2's `FrozenBackboneClassifier`.
- Produces: `ClsModel(path)` — construction loads checkpoint + backbone (raises on failure; the orchestrator's `_load_cls` wraps that into `ModelUnavailable`, Task 6). `__call__(frame_gray_uint8) -> dict`:
  `{"prob": float, "confidence": float, "deferred": bool, "reason": str}` where `confidence = max(prob, 1-prob)`, `deferred=True` with `reason="defer-band"` when `defer_band[0] <= prob <= defer_band[1]`, else `reason="confident"`. Threshold is carried through as `"threshold"` in the dict for the diagnosis layer.

- [x] **Step 1: Write the failing tests**

```python
"""Hosted torch inference for Model One. Trains nothing; loads Task 4's head.pt and mirrors B3:
a calibrated probability inside the defer band NEVER becomes a confident call."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.frozen_backbone import FrozenBackboneClassifier
from src.serve.infer_cls import ClsModel


def _ckpt(tmp_path, band=(0.3, 0.6), thr=0.5):
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    p = tmp_path / "head.pt"
    torch.save({"backbone": "test-tiny", "imgsz": 32, "head_state": m.head.state_dict(),
                "temperature": 1.0, "threshold": thr, "defer_band": list(band)}, p)
    return p


def test_returns_contract_keys_and_types(tmp_path):
    model = ClsModel(_ckpt(tmp_path))
    out = model(np.zeros((64, 64), dtype=np.uint8))
    assert set(out) >= {"prob", "confidence", "deferred", "reason", "threshold"}
    assert 0.0 <= out["prob"] <= 1.0 and isinstance(out["deferred"], bool)


def test_prob_inside_defer_band_defers(tmp_path):
    model = ClsModel(_ckpt(tmp_path, band=(0.0, 1.0)))     # band swallows everything
    out = model(np.zeros((64, 64), dtype=np.uint8))
    assert out["deferred"] is True and out["reason"] == "defer-band"


def test_missing_checkpoint_raises_at_construction(tmp_path):
    with pytest.raises(Exception):
        ClsModel(tmp_path / "absent.pt")
```

- [x] **Step 2: Verify failure** — `No module named 'src.serve.infer_cls'`.

- [x] **Step 3: Implement**

```python
"""Hosted torch inference for Model One (B8: central serving; this is NOT the CoreML edge path).

Loads the trainer's head.pt (backbone name + head weights + temperature + threshold + defer band)
and answers one de-identified still frame at a time. B3 posture is enforced here, closest to the
model: a calibrated probability inside the defer band is returned deferred -- downstream layers
may defer MORE, never less. Construction failures raise; src.serve.orchestrator._load_cls turns
them into ModelUnavailable so one bad checkpoint defers studies instead of crashing the service.
"""
import numpy as np


class ClsModel:
    def __init__(self, path):
        import torch
        from src.models.frozen_backbone import FrozenBackboneClassifier
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self._torch = torch
        self.imgsz = int(ckpt["imgsz"])
        self.temperature = float(ckpt["temperature"])
        self.threshold = float(ckpt["threshold"])
        self.defer_band = tuple(ckpt["defer_band"])
        self.model = FrozenBackboneClassifier(ckpt["backbone"], imgsz=self.imgsz)
        self.model.head.load_state_dict(ckpt["head_state"])
        self.model.eval()

    def _prep(self, frame_gray):
        import cv2
        img = cv2.resize(np.asarray(frame_gray), (self.imgsz, self.imgsz),
                         interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        return self._torch.from_numpy(img[None, None])

    def __call__(self, frame_gray):
        with self._torch.no_grad():
            logit = float(self.model(self._prep(frame_gray))[0])
        prob = float(1.0 / (1.0 + np.exp(-logit / self.temperature)))
        lo, hi = self.defer_band
        deferred = lo <= prob <= hi
        return {"prob": prob, "confidence": float(max(prob, 1.0 - prob)),
                "deferred": bool(deferred), "reason": "defer-band" if deferred else "confident",
                "threshold": self.threshold}
```

- [x] **Step 4: Verify pass** — 3 passed.
- [x] **Step 5: Full suite green.** Also confirm import-safety: `python -c "import src.serve.infer_cls"` must not import torch at module scope (it doesn't — torch only inside methods).

---

### Task 6: diagnosis + orchestrator + config wiring

**Files:**
- Modify: `src/serve/diagnosis.py` (add `cls_to_finding`)
- Modify: `src/serve/orchestrator.py` (`_load_cls`, `_model_factory` cls branch, `analyze_frame` cls findings branch)
- Modify: `configs/orchestrator.yaml` (add `avf_fistulography` entry)
- Test: `tests/test_cls_finding.py` + append 2 tests to `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: Task 5's output dict; existing `Finding` dataclass (`label, display_name, confidence, deferred, reason, boxes`); existing `TaskEntry` (`modality, task, model_path, display_name, finding_label, finding_display, floor_ok`); existing `ModelUnavailable` pattern in `_load_det`/`_load_seg`.
- Produces:
  - `cls_to_finding(entry, cls_res) -> Finding` — malformed input (missing `prob`/`deferred`) → deferred `reason="malformed-cls"`; `entry.floor_ok` False → deferred `reason="below-floor"` (mirrors `det_to_findings`); positive call (`prob >= threshold`) or defer-band → deferred/kept accordingly, `confidence=cls_res["confidence"]`, `boxes=[]`.
  - Orchestrator: `entry.task == "cls"` → `findings = [cls_to_finding(entry, out)]`; `_model_factory` builds `ClsModel` lazily with the same fail-safe closure shape as `_load_det`.

- [x] **Step 1: Write failing tests for `cls_to_finding`** (`tests/test_cls_finding.py`)

```python
"""cls output -> Finding. Same fail-safe grammar as det/seg: malformed input and below-floor both
defer -- a classifier that didn't demonstrably run + clear its floor never emits a confident call."""
from src.serve.diagnosis import cls_to_finding
from src.serve.registry import TaskEntry


def _entry(floor_ok=True):
    return TaskEntry("avf_fistulography", "cls", "head.pt", "AVF fistulography",
                     "avf_ja_stenosis", "Possible juxta-anastomotic stenosis", floor_ok=floor_ok)


def test_confident_positive_becomes_kept_finding():
    f = cls_to_finding(_entry(), {"prob": 0.9, "confidence": 0.9, "deferred": False,
                                  "reason": "confident", "threshold": 0.5})
    assert not f.deferred and f.confidence == 0.9 and f.label == "avf_ja_stenosis"


def test_defer_band_result_stays_deferred():
    f = cls_to_finding(_entry(), {"prob": 0.45, "confidence": 0.55, "deferred": True,
                                  "reason": "defer-band", "threshold": 0.5})
    assert f.deferred and f.reason == "defer-band"


def test_below_floor_forces_defer_even_when_confident():
    f = cls_to_finding(_entry(floor_ok=False), {"prob": 0.95, "confidence": 0.95,
                                                "deferred": False, "reason": "confident",
                                                "threshold": 0.5})
    assert f.deferred and f.reason == "below-floor"


def test_malformed_output_defers():
    f = cls_to_finding(_entry(), {"confidence": 0.9})       # no prob, no deferred
    assert f.deferred and f.reason == "malformed-cls"
```

Orchestrator tests (append to `tests/test_orchestrator.py`, reusing its fakes):

```python
def test_cls_modality_flows_through_analyze_frame(monkeypatch):
    monkeypatch.setattr(orch_mod, "record", lambda *a, **k: None)
    entry = TaskEntry("avf_fistulography", "cls", "head.pt", "AVF fistulography",
                      "avf_ja_stenosis", "Possible juxta-anastomotic stenosis", floor_ok=True)
    router = FakeRouter(ModalityDecision("avf_fistulography", None, True, 0.9, False, "confident"))

    def cls_factory(e):
        return lambda frame: {"prob": 0.9, "confidence": 0.9, "deferred": False,
                              "reason": "confident", "threshold": 0.5}
    orch = DiagnosticOrchestrator(router, {"avf_fistulography": entry}, cls_factory)
    report = orch.analyze_frame(FRAME)
    assert report.findings[0].label == "avf_ja_stenosis" and not report.deferred


def test_unknown_cls_checkpoint_defers_model_unavailable():
    from src.serve.orchestrator import _model_factory
    entry = TaskEntry("avf_fistulography", "cls", "definitely/absent/head.pt",
                      "AVF fistulography", "avf_ja_stenosis", "x", floor_ok=True)
    model = _model_factory(entry)
    import numpy as np, pytest as _pytest
    with _pytest.raises(orch_mod.ModelUnavailable):
        model(np.zeros((8, 8), dtype=np.uint8))
```

- [x] **Step 2: Verify failures** — `cannot import name 'cls_to_finding'`; orchestrator cls test fails on unknown-task `ModelUnavailable`.

- [x] **Step 3: Implement.** `diagnosis.py` (mirror the docstring register of `det_to_findings`/`seg_to_finding`):

```python
def cls_to_finding(entry, cls_res):
    """Classifier result dict -> one Finding. Same fail-safe grammar as det/seg: a result missing
    'prob' or 'deferred' is malformed input (defer, never a confident negative); a below-floor
    entry defers regardless of confidence (the floor is the clinical gate, not the model's mood)."""
    if not entry.floor_ok:
        return Finding(label=entry.finding_label, display_name=entry.finding_display,
                       confidence=0.0, deferred=True, reason="below-floor", boxes=[])
    if "prob" not in cls_res or "deferred" not in cls_res:
        return Finding(label=entry.finding_label, display_name=entry.finding_display,
                       confidence=0.0, deferred=True, reason="malformed-cls", boxes=[])
    return Finding(label=entry.finding_label, display_name=entry.finding_display,
                   confidence=float(cls_res.get("confidence", 0.0)),
                   deferred=bool(cls_res["deferred"]),
                   reason=str(cls_res.get("reason", "confident")), boxes=[])
```

`orchestrator.py` — import `cls_to_finding` beside the other two; in `analyze_frame` replace the det/else chain with:

```python
        if entry.task == "det":
            findings = self._det_findings(entry, out)
        elif entry.task == "cls":
            findings = [cls_to_finding(entry, out)]
        else:
            findings = [seg_to_finding(entry, out)]
```

Wait — the current `else` treats every non-det as seg. Keep exact current behavior for unknown tasks: the existing `_model_factory` already routes unknown task types to a `ModelUnavailable` raiser, so the `else: seg` line is only ever reached for task == "seg". Preserve that invariant with the three-way branch above. Add `_load_cls` beside `_load_det`/`_load_seg` (same closure pattern, same comment register):

```python
def _load_cls(model_path):
    try:
        from src.serve.infer_cls import ClsModel
        return ClsModel(model_path)
    except Exception as e:
        def _unavailable(frame, _path=model_path, _e=e):
            raise ModelUnavailable(f"cls model at {_path!r} failed to load: {_e}") from _e
        return _unavailable
```

and in `_model_factory` add the branch `if entry.task == "cls": return _load_cls(entry.model_path)` alongside det/seg. `configs/orchestrator.yaml` — append under `modalities:` (floor_ok false: floors unsigned, B3):

```yaml
  avf_fistulography:
    task: cls
    model_path: runs/avf_cls/head.pt
    display_name: AVF fistulography
    finding_label: avf_ja_stenosis
    finding_display: Possible juxta-anastomotic stenosis (triage aid; clinician review required)
    floor_ok: false
```

- [x] **Step 4: Verify pass** — 4 + 2 new tests pass.
- [x] **Step 5: Full suite green.** Also rerun the event tests (`tests/test_serve_events.py`) — `model.inferred` must fire for cls findings with `task: "cls"` in the payload (it will: the publish site is task-agnostic).

---

### Task 7: end-to-end proof on synthetic data + tracker

**Files:**
- Test: append one integration test to `tests/test_train_classifier.py`
- Modify: `docs/PROJECT_TRACKER.md` (changelog entry + T1.4 note)

**Interfaces:** none new — this task proves Tasks 1–6 compose.

- [x] **Step 1: Write the failing integration test**

```python
def test_trained_head_serves_through_the_orchestrator(tmp_path, monkeypatch):
    """train -> head.pt -> ClsModel -> analyze_frame: the full Model One path, synthetic only."""
    torch = pytest.importorskip("torch")
    import numpy as np
    from src.serve import orchestrator as orch_mod
    from src.serve.orchestrator import DiagnosticOrchestrator, _model_factory
    from src.serve.registry import TaskEntry
    from src.serve.router import ModalityDecision
    from src.train.train_classifier import train

    monkeypatch.setattr(orch_mod, "record", lambda *a, **k: None)
    trips = ([(f"avf_inu_{i:010x}_s01", 6, 1) for i in range(4)]
             + [(f"avf_inu_{i + 8:010x}_s01", 6, 0) for i in range(4)])
    frames, labels = _store(tmp_path, trips)
    train(frames, labels, tmp_path / "run", backbone="test-tiny", imgsz=32, epochs=8,
          val_frac=0.45, seed=3)

    entry = TaskEntry("avf_fistulography", "cls", str(tmp_path / "run" / "head.pt"),
                      "AVF fistulography", "avf_ja_stenosis", "Possible JA stenosis",
                      floor_ok=True)

    class R:
        def classify(self, frame):
            return ModalityDecision("avf_fistulography", None, True, 0.9, False, "confident")

    orch = DiagnosticOrchestrator(R(), {"avf_fistulography": entry}, _model_factory)
    report = orch.analyze_frame(np.full((64, 64), 120, dtype=np.uint8))   # bright = positive class
    assert report.modality == "avf_fistulography"
    assert report.findings and report.findings[0].label == "avf_ja_stenosis"
    # No assertion on deferred: with a defer band this small model may legitimately abstain.
    # The claim under test is the PLUMBING: trained artifact -> real factory -> typed finding.
```

- [x] **Step 2: Verify it fails only if plumbing is broken** — run it; with Tasks 1–6 done it should PASS immediately. If it fails, that's a real seam bug — fix the seam, not the test. (This is the one deliberately-green test in the plan: it's an integration proof, written after its parts were TDD'd individually.)
- [x] **Step 3: Tracker** — add a changelog entry to `docs/PROJECT_TRACKER.md`: date, "Model One scaffold: cls metrics, frozen-backbone trainer, hosted ClsModel, orchestrator cls path — all synthetic-tested; real training blocked on B5/B9 + backbone bake-off; floors unsigned so registry ships floor_ok: false". Update the §2 inventory row for `src/train/` (train_classifier.py real) and note T1.4 code-complete in the realignment cross-reference.
- [x] **Step 4: Full suite** — `python -m pytest tests/ -q` → expected ≈ 621 + 6 + 4 + 4 + 3 + 3 + 6 + 1 = 648 passed (exact count recorded by implementer; arithmetic is advisory, actual governs).

---

## Self-Review (done at authoring)

1. **Coverage:** B3 (defer band, threshold-from-sensitivity, never false normal) → Tasks 1/4/5/6; B4 (frozen backbone, timm-lazy, bake-off-ready via `--backbone`) → Task 2; B5/B6 (patient split as assertion, no real data) → Task 3; remediation P3 build-new list (train_cls, infer_cls, cls_metrics) → Tasks 1/4/5; registry/orchestrator integration → Task 6; end-to-end proof → Task 7. NOT in scope (deliberate): validity-gate model (needs its own training set — separate plan), router AVF label + router training (blocked on router trainer decision), real backbone bake-off (needs GPU + real frames), external validation (B6, blocked on second site).
2. **Placeholders:** none — every step carries runnable code. The one advisory number (final test count) is labeled advisory.
3. **Type consistency:** `head.pt` schema identical in Tasks 4/5/6 tests; `cls_res` dict keys identical in Tasks 5/6; `TaskEntry` positional order matches existing `tests/test_orchestrator.py` usage; `bootstrap_ci(..., thr=...)` keyword matches Task 1 signature.
4. **Known risk, called out to the implementer:** `temperature_scale`'s exact return shape (Task 4 note) — verify at `src/eval/calibration.py:85` before wiring.
