# Stage 2 Stenosis — Phase 2 POA (data + model levers)

**Owner:** tech@manufex.io · **Created:** 2026-07-18 · **Companion:** [`STAGE2_PHASE1_POA.md`](STAGE2_PHASE1_POA.md) · [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md)
**Baseline:** honest `best.pt` F1 0.291 / R 0.271 / mAP50 0.209 / mAP50-95 0.080 (floor 0.57).

---

## 0. Scope & the ordering rule

Phase 2 attacks the accuracy ceiling itself (raise the PR curve), not just the operating point. Three levers, **all tooling landed + unit-tested 2026-07-18**; the runs are GPU-side.

**Ordering rule (hard):** which lever to emphasize is decided by the **Phase-1 §5b outputs** — the per-source val table and the temporal-voting per-video sensitivity. Do **not** spend Phase-2 GPU before those exist:
- per-source table shows **one source is the recall sink** → P2.1 (fix/harmonize/drop it) first.
- recall low **uniformly** + mAP50-95 collapse persists → P2.1 harmonize convention, then P2.2 add data.
- per-video sensitivity already clears the screening bar → ship as deferred flag (Phase 1 outcome); Phase 2 becomes optional polish.

---

## 1. P2.1 — Annotation QA + harmonization

**Why.** mAP50 0.209 vs mAP50-95 0.080 = loose localization = the three sources box stenosis differently (tightness, size, what counts). This caps IoU-sensitive metrics and injects label noise.

**Measured (dry-run §3c annotation QA, 2026-07-18 — label geometry, model-independent so valid despite 3 epochs):**

| source | median box area | tiny_frac (<38px) | boxes/img |
|---|---|---|---|
| arcade | 0.0108 | 0.056 | 1.61 |
| cadica | 0.0058 | 0.125 | 1.40 |
| **danilov** | **0.0029** | **0.36** | 1.00 |

ARCADE boxes are ~**4× Danilov's area**; a third of Danilov boxes are sub-38px. Mismatch confirmed → Danilov is the outlier. (Prior ablation agrees: arcade-only 0.246 F1 **>** +danilov 0.214 — Danilov dilutes the honest metric even capped.)

**Tooling (landed 2026-07-18, TDD, 9 tests).**
- `src/eval/annotation_qa.py` — the per-source QA above (notebook §3c).
- `src/data_prep/harmonize.py` — **clamps every box up to a min w/h floor** (recall-preserving; no positive dropped), config `harmonize.min_box_wh` (default 0 = off), notebook **§3f** (`HARMONIZE` flag → 0.04 ≈ 30px, TRAIN-ONLY so val stays comparable to baseline).
- **Drop-Danilov** lever — `DROP_DANILOV=True` in notebook §3 convert cell (zero code, strongest-evidenced single experiment).

**Run (when GPU frees).** Two cheap A/Bs vs the un-harmonized baseline:
1. `DROP_DANILOV=True` → the model trains on ARCADE+CADICA only (both consistent conventions).
2. `HARMONIZE=True` (keep Danilov) → clamp tiny boxes to a common floor.
Compare each per-source (§5b P1.0) + mAP50-95 vs baseline 0.080.

**Effort** low (both are toggles) · **Uplift** med–high (directly on the localization collapse) · **Retrain** yes.

---

## 2. P2.2 — Pseudo-label SSL on XCAD (+ optional dataset balance)

**Why.** Patient diversity is the confirmed bottleneck (CADICA gave the biggest honest lift). XCAD (~1621 unlabeled frames) is a **disjoint cohort** → self-training on it adds effective patients without leaking val.

**Tooling (landed).**
- SSL self-training already implemented in `train_detector` (`_pseudo_label_round`, `_gdino_seed_round`), guarded by `ssl_enabled` (requires a disjoint `ssl.unlabeled_dir`).
- Notebook **§3e** auto-sets `ssl.unlabeled_dir=data/raw/xcad` when an *xcad* dataset is attached → SSL activates on the real (`DRY_RUN=False`) pass.
- **Dataset balance** — `src/data_prep/balance.py` (notebook **§3d**, `BALANCE=True`): oversamples the minority source with `bal_`-prefixed **train-only** copies. Runs *after* the leakage audit → cannot leak.

**Run.** Attach an XCAD unlabeled Kaggle dataset (title contains *xcad*) → Run All. §3e prints `ssl.unlabeled_dir=... (N png)`; §4 runs the pseudo-label round. Optional: set `BALANCE=True` in §3d once P1.0 names the target source. Optional GD cold-start: `ssl.seed: gdino` + `transformers` installed.

**Guardrails.** SSL stays OFF unless a disjoint on-disk unlabeled dir exists (notebook + `train()` both check) — a self-labeled val patient can't re-enter train. XCAD must be **.png** (SSL globs `**/*.png`); §3e warns if 0 found.

**Effort** high · **Uplift** high (raises the PR curve itself) · **Retrain** yes.

---

## 3. P2.3 — Model / resolution step (11m, 1024) + CoreML smoke-test

**Why.** Cheap to rule in/out. Research says S2 is data-limited → expect small single-frame gains; validate edge cost up front so a bigger/higher-res model doesn't surprise CoreML later.

**Tooling (landed).**
- Commented model toggles in `configs/stenosis_yolo.yaml` (`yolo11m` @768; `yolo11s` @1024).
- `src/export/yolo_to_coreml.py` gains `smoketest()` — exports + checks the `.mlpackage` exists / size, prints PASS/FAIL.

**Run.** Edit the `model:` line in the config → Run All. After downloading the resulting `best.pt`, on **Mac**:
```
python -m src.export.yolo_to_coreml --weights best.pt --smoketest
```

**CoreML notes.** YOLO11m/s export first-class (NMS baked in). 11m ≈ 2× latency/size (ANE-viable, edge ceiling); 11l/x = teacher-only. 1024px latency ~quadratic — bench on device. INT8 re-check mAP post-quant. RT-DETR = conversion risk (deferred).

**Effort** med · **Uplift** low–med · **Retrain** yes (per variant).

---

## 4. Test status (2026-07-18)

Full suite **275 passed** (+1 pre-existing unrelated `test_clgeodice` autograd warning). New: `test_annotation_qa.py` (17), `test_balance.py` (10), `test_yolo_to_coreml.py` (7). All new modules import torch/cv2/coremltools-free (heavy deps lazy).

---

## 5. Ranked (post-P1, tie to the §5b table)

| Lever | Effort | Uplift | Trigger |
|---|---|---|---|
| P2.1 annotation QA + harmonize | med | med–high | mAP50-95 stays low, or one source loose in §3c |
| P2.2 XCAD SSL (+ balance) | high | high | recall low uniformly; add effective patients |
| P2.3 11m / 1024 | med | low–med | after data levers; cheap to rule out |

**Gate reframe (carry-over from Phase 1 / Phase 3):** if per-video sensitivity clears the clinical screening bar, ship as a **deferred flag** under a per-video-sensitivity gate — don't wait on per-frame F1 0.57, which likely exceeds single-frame SOTA on this data. Needs Stage-5 intended-use sign-off.
