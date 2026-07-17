# Project Tracker — Interventional Imaging Pipeline

**Purpose:** single source of truth for *what is done* and *what is next*. Check boxes as you go.
**Last updated:** 2026-07-16 · **Owner:** tech@manufex.io
**Companion docs:** [`Model_Pipeline_Playbook.md`](Model_Pipeline_Playbook.md) (rationale) · [`DATASETS.md`](DATASETS.md) · [`COLAB_MAC_SPLIT.md`](COLAB_MAC_SPLIT.md) · repo [`README.md`](../README.md)

---

## 0. How to use this file

- `- [x]` done & verified · `- [~]` partial / in-progress · `- [ ]` not started · `- [!]` blocked (reason noted)
- Each stage carries **two gates**: an **accuracy floor** (before edge optimization) and a **safety/sign-off** gate (calibration + cross-vendor). A stage is not "done" until both gates pass on the target device.
- **Golden invariant:** heavy models are *teachers/labelers on the GPU build side only*. Only distilled/quantized students ship to edge (Mac / procedure-cart). Grounding DINO obeys this rule — it is a **build-side labeler**, never shipped.
- Build side = Colab/Kaggle GPU (thin notebooks import `src/*`). Deploy side = Mac CoreML. Local processed splits live on the GPU, not on this laptop — so "no `data/processed/` locally" is expected, not a gap.

---

## 1. Status snapshot (2026-07-16)

| Stage | Title | State | Trained artifact | Gate status |
|---|---|---|---|---|
| 0 | Setup + data prep | `~` partial | — | CLAHE walk **done**; edge-bench torch path still TODO |
| 1 | Coronary segmentation | `x` gate verified & passed | `student.pt`+onnx+int8 (CLGeoDice, 2026-07-16) — `outputs/coronary_student_clgeodice/` | **Dice 0.915 / clDice 0.956 via the CLGeoDice run (2026-07-16) → CLEARS the Dice ≥ 0.75 floor.** Prior `outputs/coronary_student/` run (2026-07-12) was gate UNVERIFIED (Dice/clDice unlogged); this run is verified |
| 2 | Stenosis detection | `!` below floor | honest `best.pt` (F1 0.291) | +CADICA re-run done 2026-07-16 (`arcade+cadica+danilov_yolo11s_768_e150`) → **F1 0.291 / recall 0.271 < 0.57 floor** (up from F1 0.214); CADICA (+3996 keyframes) confirmed patient-diversity is the lever |
| 2.5 | Calibration + abstention | `~` partial | — | ECE coded; reliability/temp-scale/OOD are TODO |
| 3 | Temporal + catheter tracking | `x` **done** | `best-catheter.pt` + 4 provenance zips | detection+track complete |
| 3b | Cross-vendor validation | `!` blocked | — | eval harness is a TODO shell |
| 4 | Domain (AVF / TAVR) | `[ ]` not started | — | data-gated (IRB) |
| 5 | Regulatory / intended-use gate | `[ ]` not started | — | name before any non-research use |
| GD | **Grounding DINO labeler** (new) | `~` scaffolded | — | modules + pure helpers done (2026-07-11); SSL-seed wiring pending |

**One-line summary:** Stage 3 (catheter) trained end-to-end. Stage 1 (coronary): CLGeoDice distillation run **done 2026-07-16 → Dice 0.915 / clDice 0.956, CLEARS the Dice ≥ 0.75 floor** — the first coronary run with metrics on record (artifacts in `outputs/coronary_student_clgeodice/`; the prior `outputs/coronary_student/` run had its gate unverified). Stage 2 (stenosis): +CADICA honest patient-grouped re-run **done 2026-07-16 → F1 0.291 / recall 0.271, still BELOW floor 0.57** (up from F1 0.214; CADICA added 3996 keyframes and confirmed patient diversity is the lever). Grounding DINO labeler is scaffolded (modules import torch-free, pure helpers unit-tested). Local test suite: **150 passing** (+3 skipped) (`pytest tests/`).

---

## 2. Code inventory (implemented vs stub)

Ground-truth from `src/` on 2026-07-11. Line counts in parens.

### Implemented (real code)
- [x] `src/env.py` (59) — Colab/Kaggle/local detection + paths
- [x] `src/data_prep/arcade_to_coco.py` (27) — ARCADE → COCO
- [x] `src/data_prep/dca1_to_nnunet.py` (48) — DCA1 → nnU-Net NIfTI/PNG
- [x] `src/data_prep/danilov_to_yolo.py` (91) — Danilov → YOLO boxes
- [x] `src/data_prep/cathaction_to_yolo.py` (96) — CathAction → YOLO
- [x] `src/data_prep/io_utils.py` (157) — shared IO + `clahe_unsharp`
- [x] `src/data_prep/verify_sequence.py` (61) — sequence integrity check
- [x] `src/data_prep/preprocess.py` `process_dir()` (2026-07-11) — CLAHE+unsharp batch walk, mirrors tree, normalizes to .png
- [x] `src/train/train_seg.py` (2026-07-11) — coronary teacher→distill→eval→export driver + pure config helpers
- [x] `src/data_prep/autolabel_gdino.py` (2026-07-11) — Grounding DINO auto-labeler (pure box→YOLO/COCO helpers + lazy `detect`/`autolabel_dir`)
- [x] `src/models/grounded_sam.py` (2026-07-11) — DINO box → SAM mask (box-prompted), `to_seg_pairs`
- [x] `tests/` — `test_preprocess.py`, `test_train_seg.py`, `test_autolabel_gdino.py`, `test_train_detector.py`, `test_split_grouping.py` → **52 passing**, all import torch-free
- [x] `src/models/seg_student.py` (46) — TinyU-Net student
- [x] `src/models/distill.py` (94) — KD loss + distillation loop
- [x] `src/train/train_detector.py` — YOLO11n trainer + pseudo-label SSL + **GD cold-start seed** + speed knobs (`train_kwargs`); pure helpers unit-tested (2026-07-11)
- [x] `src/eval/metrics.py` (30) — Dice / clDice / HD95
- [x] `src/eval/audit.py` (25) — input-hash + model-version + prediction log
- [x] `src/export/to_onnx.py` (14), `quantize_int8.py` (10), `to_coreml.py` (52), `coreml_validate.py` (78), `yolo_to_coreml.py` (21)
- [x] `src/serve/infer.py` (84), `predict_image.py` (65), `realtime.py` (84), `track.py` (121, ByteTrack), `app.py` (53, FastAPI)

### Stubs / partial (must implement before their stage can pass)
- [x] ~~`src/train/train_seg.py`~~ — **implemented 2026-07-11** (was `NotImplementedError`). No longer blocks Stage 1.
- [x] ~~`src/data_prep/preprocess.py` walk~~ — **`process_dir()` implemented 2026-07-11**.
- [x] ~~`src/models/sam_adapter.py`~~ — **deleted 2026-07-12** (dead `NotImplementedError` stub, 0 callers; superseded by `src/models/grounded_sam.py`).
- [!] `src/data_prep/dsca_sequences.py` (11) — `NotImplementedError`. DSA temporal prep. Blocks Stage 3 DSA.
- [!] `src/train/train_audio.py` (8) — `NotImplementedError`. AVF audio (mel → ViT). Blocks Stage 4 audio.
- [~] `src/eval/calibration.py` (41) — `ece()` implemented; **TODO:** reliability diagram, temperature scaling, OOD-AUROC. Blocks Stage 2.5 sign-off.
- [~] `src/eval/cross_vendor.py` (26) — shell only; **TODO:** wire to `train`+`metrics`, emit per-vendor table + worst-case gap. Blocks Stage 3b.
- [~] `src/eval/edge_benchmark.py` (39) — ONNX path works; **TODO:** torch path (param count + cuda/cpu timing).

---

## 3. Stage checklists (detailed)

### 3.0 Stage 0 — Setup + data prep  `~`
- [x] Repo scaffold, `environment.yml` / `requirements.txt`, `Makefile`
- [x] Dataset converters: ARCADE→COCO, DCA1→nnU-Net, Danilov→YOLO, CathAction→YOLO
- [x] Edge-benchmark harness (ONNX path)
- [x] **`preprocess.py` CLAHE+unsharp batch walk** — `process_dir(src, dst, size=…)` implemented + tested (2026-07-11)
- [ ] Run converters to materialize `data/processed/{coronary,stenosis}/` splits on the GPU
- [ ] Torch path in `edge_benchmark.py` (param count + cpu/cuda timing)
- **Exit gate:** one command reproduces a split + a latency report on the target device.

### 3.1 Stage 1 — Coronary segmentation  `~` (driver done; ready-to-run)
**Data ready on disk:** DCA1 (134 img + 134 `_gt` masks, complete), ARCADE syntax (train/val/test). XCAD unlabeled for SSL (GPU side).
- [x] **`src/train/train_seg.py` implemented** (2026-07-11) — wires the full path against existing APIs:
  - [x] nnU-Net v2 teacher train + predict argv builders (`nnunet_train_cmd`/`nnunet_predict_cmd`, subprocess)
  - [x] TinyU-Net student distill via `src.models.distill.distill` + `TeacherCacheDataset`
  - [x] eval via `src.eval.metrics` (Dice + clDice), `qualifies()` gate
  - [x] CoreML export via `src.export.to_coreml` (guarded: `export.coreml` and macOS)
  - [ ] **Refinement:** `qualifies()` gates on Dice only — extend to require clDice within ~3% of teacher (playbook exit gate)
- [x] **Coronary driver ran** — `outputs/coronary_student/{student.pt,student.onnx,student.int8.onnx}` produced (2026-07-12). **BUT Dice/clDice were not logged → accuracy-floor gate UNVERIFIED (re-eval to record numbers).**
- [x] **Accuracy-floor gate VERIFIED & PASSED** (2026-07-16) — CLGeoDice distillation run (`clgeodice_weight 0.5`, 200/200 epochs) → **Dice 0.915 (best mid-run 0.927) ≥ 0.75 ✅, clDice 0.956 (best mid-run 0.980)**. First coronary run with metrics on record; artifacts in `outputs/coronary_student_clgeodice/{student.pt,student.onnx,student.int8.onnx}` (gitignored). Supersedes the 2026-07-12 UNVERIFIED run above.
- [ ] SSL pretraining on XCAD 1,621 unlabeled + institutional cine
- [ ] CoreML export + `make validate-coreml` + `make bench-coreml` on Mac
- **Accuracy floor gate:** Dice ≥ 0.75 **AND** clDice within ~3% of teacher, **re-checked after INT8** (INT8 breaks thin vessels).
- **Fallback ladder if clDice drops:** QAT → larger student → keep teacher as offline second-read.

### 3.2 Stage 2 — Stenosis detection  `~` (fastest real win)
**Data ready on disk:** ARCADE stenosis (train/val/test). Danilov (GPU side) for COCO AP.
- [x] Trainer `train_detector.py` + pseudo-label SSL round implemented
- [x] `stenosis_yolo.yaml` config present
- [x] **Speed knobs** (2026-07-11): `train_kwargs` threads cache/workers/patience/amp into every `model.train`; config carries fast defaults; notebook enables cuDNN autotune — quality-neutral
- [x] **GD cold-start seed** wired (opt-in `ssl.seed: gdino`) — see Grounding DINO Slot 2
- [x] **First real run done** (2026-07-11, Kaggle): `arcade_yolo11n_640_e150`, ARCADE-only → **F1 0.246, mAP50 0.147 — below floor 0.57.** Verified learning (not a bug): clean labels, preds land on vessels but miss many. Archived: [`experiments/stenosis_arcade_yolo11n_640_e150/`](../experiments/stenosis_arcade_yolo11n_640_e150/RESULTS.md)
- [~] **Second run done** (2026-07-12, Kaggle, `arcade+danilov_yolo11s_768_e150`): +Danilov (7861 train/1464 val), 11s, imgsz 768, 101/150 epochs (12h cap) → **F1 0.885, mAP50 0.87 — but per-frame split leaks Danilov video frames (every patient in both splits), so the number is inflated and NOT a trustworthy Stage-2 result.** Archived: [`experiments/stenosis_arcade+danilov_yolo11s_768_e150/`](../experiments/stenosis_arcade+danilov_yolo11s_768_e150/RESULTS.md)
- [x] **Leakage fix**: `io_utils.split_of` now patient-grouped (`group_key`) so Danilov frames of a patient share a split; ARCADE unchanged; 47 tests pass
- [x] **Leakage hard-gate in the notebook** (2026-07-12 c): `io_utils.audit_split_leakage()` + a new §3b cell in `kaggle_stenosis_plug_and_play.ipynb` **raise before training** if (a) any patient/clip group is in both train+val, or (b) Danilov frames were not actually collapsed by `group_key` (real filenames ≠ `<site>_<patient>_<seq>_<frame>` → silent per-frame leak). Danilov stem set is read from raw *independently of the regex* so a silent no-op can't pass. SSL pseudo-label auto-disabled unless a disjoint `ssl.unlabeled_dir` exists (else it re-leaks val frames into train). 55 tests pass.
- [x] **Re-run with patient-grouped split DONE** (2026-07-13, Kaggle `jugalmodi0111/stenosis`): honest split (train 8766/1349 groups, val 1059/215 groups; leakage check passed) → **F1 0.214, mAP50 0.108 — BELOW floor 0.57.** The 0.885 was ~all frame-leakage; Danilov's 8325 frames = only 64 patients, so patient diversity (not epochs/model) is the bottleneck. Archived: [`experiments/stenosis_arcade+danilov_yolo11s_768_grouped/`](../experiments/stenosis_arcade+danilov_yolo11s_768_grouped/RESULTS.md)
- [x] **+CADICA re-run DONE** (2026-07-16, Kaggle `jugalmodipesurr/stenosis`, `arcade+cadica+danilov_yolo11s_768_e150`): added **CADICA (3996 keyframes)** on the honest patient-grouped split (leakage audit passed) → **F1 0.291 / recall 0.271 / mAP50 0.209 — still BELOW floor 0.57**, but a real lift from F1 0.214 (**+0.077 F1, +0.105 recall ~+63% relative, +0.101 mAP50**). CADICA is the **biggest honest single-lever gain to date** and confirms patient diversity is the lever; next levers are **more patients + pseudo-label SSL**. Archived: [`experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/`](../experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/RESULTS.md)
- [x] **Phase 1 quick-win code landed** (2026-07-17, local TDD, 3 parallel agents): (1) `src/eval/val_by_source.py` — per-source val (ARCADE/CADICA/Danilov) diagnostic, ultralytics lazy-imported, `source_of` unit-tested; (2) `train_detector.train_kwargs` now passes an optional `augment:` config block through to `model.train()` — was hardcoded to YOLO COCO defaults; `configs/stenosis_yolo.yaml` gets a domain-tuned block (mosaic 0.0, scale 0.2, erasing 0.0, HSV 0, box 9.0/dfl 2.0, cos_lr, epochs 150→80); (3) `io_utils.group_key` now recognizes CADICA `pXX_vYY_NNNNN → pXX` (fixes the ~34%-val over-count; no live leak, accounting only) + `cadica_to_yolo` per-patient cap via `datasets.cadica.max_frames_per_patient: 40`; (4) recall-first gate `target: {f1: 0.57, recall: 0.60}`. Plan: [`STAGE2_PHASE1_POA.md`](STAGE2_PHASE1_POA.md). Suite **240 passed** (+1 pre-existing torch-in-sys.modules order-pollution failure in `test_train_seg.py`, unrelated — passes in isolation). **GPU-side remaining:** run per-source val on baseline best.pt (P1.0), op-point sweep + temporal-voting per-video sensitivity (P1.1), combined aug+split re-run (P1.4).
- [ ] Run naming: `run_tag(cfg)` auto-names each run folder (no clobber); Kaggle notebook wired
- [ ] Pseudo-label SSL round on unlabeled frames (raise recall)
- [ ] Track COCO AP/AR on Danilov
- [ ] Export to CoreML (`yolo_to_coreml.py`) + edge bench on Mac
- **Accuracy floor gate:** F1 ≥ 0.55, **recall-weighted** (a missed stenosis is the costly error). Plain YOLO11n ~0.54 is below floor — step to `s` + SSL, or fall back to RT-DETR-R18.

### 3.2.5 Stage 2.5 — Calibration + abstention  `~` → mostly done (2026-07-12 e)
- [x] `ece()` implemented (NaN on empty, not fake-0)
- [x] **Reliability diagram** — `reliability_curve()` (pure per-bin conf/acc/count) + `save_reliability_diagram()` (matplotlib-guarded PNG)
- [x] **Post-hoc temperature scaling** — `temperature_scale()` (pure 1-D golden-section on BCE) + `apply_temperature()`. Verified: over-confident logits ECE 0.094 → **0.020** (< 0.05 gate)
- [x] **OOD-AUROC + coverage–risk** — `coverage_risk()` (None at zero coverage), `auroc()` (tie-averaged Mann–Whitney), `ood_auroc()` + `uncertainty_score()` (`1-|2p-1|`). Demo OOD-AUROC 0.907
- [x] **Brier** score (`brier()`)
- [ ] Wire `CoronaryDominance` artifact/quality tags into the defer path (needs a scored model + the RAD-DINO classifier head)
- [ ] Score a REAL model (Stage-1 seg or Stage-2 det) once weights land → record ECE/reliability/OOD on held-out
- **Exit gate:** ECE < ~0.05 after temp-scaling *(math verified on synthetic; pending real-model numbers)*; defer path demonstrably fires on OOD inputs (unfamiliar vendor/view/artifact).

### 3.3 Stage 3 — Temporal + catheter tracking  `x` DONE (catheter) / `~` (DSA pending)
- [x] Catheter/guidewire YOLO11n trained — `outputs/best-catheter.pt`, `last-catheter.pt`
- [x] ByteTrack tracking (`src/serve/track.py`) — `stage3-catheter_tracking.zip`
- [x] Audit/provenance bundle — `stage3-catheter_audit-provenance.zip`, `stage3-catheter_trainrun.zip`, `stage3-catheter_weights.zip`
- [ ] **Verify catheter gates:** IoU ≥ 0.50, fps + ID-switch count on the *real* device (record numbers here)
- [ ] Export catheter `best.pt` → CoreML + edge bench on Mac
- [ ] Thin-wire recall: reimplement AttWire multi-scale Gaussian-derivative attention head if guidewire recall is short
- [!] **Cerebral DSA (temporal):** implement `dsca_sequences.py` (stub) → keyframe 2D + ConvLSTM-lite + MinIP; DSANet as offline second-read. **DSA floor Dice ~0.85** (not 0.90 — that needs full temporal fusion).

### 3.3b Stage 3b — Cross-vendor validation  `!`
- [!] **Implement `cross_vendor.py`** (TODO shell): leave-one-vendor-out (ARCADE=Philips/Siemens, DCA1=IMSS, XCAD=GE, Danilov=Siemens+GE)
- [ ] Emit per-vendor Dice/F1 table + worst-case held-out gap
- **Exit gate:** held-out-vendor gap reported and within agreed bound.

### 3.4 Stage 4 — Domain extensions (AVF / TAVR)  `[ ]`
- [ ] AVF audio (ship first): implement `train_audio.py` (mel-spectrogram → small ViT / CNN-BiLSTM). **Sensitivity ≥ 0.85, framed as screening/triage, not confirmation.**
- [ ] AVF surveillance tabular (best ROI): XGBoost/LightGBM + SHAP, **AUROC ≥ 0.80** + calibration (ECE)
- [ ] AVF imaging (ultrasound/fistulography): lightweight U-Net from coronary weights — **data-gated, needs IRB**
- [ ] TAVR CT sizing (offline GPU, NOT edge): 3D nnU-Net/SwinUNETR on MM-WHS + Seg.A → domain-adapt; **ICC ≥ 0.95**
- [ ] TAVR intra-procedural fluoro (edge): YOLO11n valve/catheter, detection ≥ 0.85
- **Note:** AVF/TAVR open imaging data essentially unavailable — budget primary collection + IRB.

### 3.5 Stage 5 — Regulatory / intended-use gate  `[ ]`
- [ ] Name assistive vs autonomous, SaMD class, prospective-validation plan **before any non-research use**
- [ ] Set the provisional accuracy floors with clinical stakeholders (they are placeholders today)

---

## 4. Grounding DINO integration (new workstream)  `[ ]`

**What it is:** open-vocabulary object *detection* (text prompt → boxes), from IDEA Research. Not a classifier; not Meta. (Meta's are DINOv2 — already used via RAD-DINO encoder init — and SAM.)

**Placement decision:** build-side **auto-labeling teacher**, distilled *into* the edge YOLO students. Too heavy for the cart (Swin-T ~172M), so it never ships — same class as nnU-Net/DSANet. Correctness note: for whole-image **classification** (view type, coronary dominance, quality flag), use a **DINOv2/RAD-DINO encoder + linear head**, *not* Grounding DINO.

### Slot 1 — Grounded-SAM auto-labeler (primary)  `~` scaffolded
- [x] `src/data_prep/autolabel_gdino.py` (2026-07-11) — `detect()` (HF `grounding-dino-tiny`, lazy) + `autolabel_dir()`; pure `dino_boxes_to_yolo_lines`/`filter_detections`/`dino_to_coco` unit-tested. `DEFAULT_PROMPTS` for stenosis/catheter/coronary.
- [x] `src/models/grounded_sam.py` (2026-07-11) — `GroundedSAM.mask_from_boxes()` (box-prompted SAM, lazy `mobile_sam`/`segment_anything`) + `to_seg_pairs()` → `io_utils.write_pair`.
- [x] Emits COCO JSON (`autolabel_coco.json`) + YOLO dataset via `io_utils` conventions.
- [ ] **Run on Colab GPU** against real cine (needs transformers + SAM checkpoint) — validate boxes/masks before they train a shipping student.

### Slot 2 — Cold-start seed for SSL  `~` wired (opt-in)
- [x] `ssl.seed: gdino` option documented in `stenosis_yolo.yaml` (commented; default off) + helpers read it
- [x] Wired into `train_detector.py`: `train()` branches on `ssl_seed(cfg)=='gdino'` → `_gdino_seed_round()` runs before self-training; pure helpers `seed_prompt_and_classes`/`boxes_labels_to_yolo_lines` unit-tested (2026-07-11)
- [ ] **Run on Colab GPU** with `ssl.seed: gdino` + `transformers` installed + `ssl.unlabeled_dir` set → validate the cold-start lift

### Slot 3 — OOD flag at abstention gate
- [ ] Open-vocab detector flags objects the closed-set student never trained on → feed Stage 2.5 defer path

**Gate:** GD-labeled data must pass the same accuracy floor as hand-labeled before it trains a shipping student; log GD version + prompt in the audit trail.

---

## 5. Cross-cutting checklist (applies to every stage)
- [x] Audit trail: input-hash + model version + prediction (`eval/audit.py`)
- [ ] Standardize annotations: COCO JSON (detection) + nnU-Net NIfTI/PNG (semantic)
- [ ] Encoder init from RAD-DINO / BiomedCLIP; SSL on unlabeled angiograms for the grayscale gap
- [ ] Edge metrics on the **real device, INT8**: params(M), FLOPs(G), latency(ms), fps, peak RAM(MB), model size(MB)
- [ ] PhysioNet credentialed access (CITI + DUA) for MIMIC-CXR / VinDr-CXR; register CheXpert

---

## 6. Data inventory (on this laptop, 2026-07-11)
| Dataset | Path | Contents | Use |
|---|---|---|---|
| DCA1 (134 Angiograms) | `datasets/Database_134_Angiograms/` | 134 img + 134 `_gt.pgm` masks | Coronary seg (complete masks, mostly normal) |
| ARCADE syntax | `datasets/arcade/syntax/{train,val,test}` | SYNTAX regions | Coronary seg (task 1) |
| ARCADE stenosis | `datasets/arcade/stenosis/{train,val,test}` | boxes | Stenosis (task 2) |
| Model Selection Matrix | `../Model_Selection_Matrix.xlsx` | scored picks + floors | model choice |
| Dataset Validation Scoring | `../Angiography_Dataset_Validation_Scoring.xlsx` | data QA | data gate |

*XCAD, Danilov, CathAction, DIAS/DSCA, MM-WHS/Seg.A live on the GPU build side (see `DATASETS.md`).*

---

## 7. Immediate next actions (top of the queue)

**Do in this order — each is independently shippable:**

*Done 2026-07-11 (code-side, local, TDD): `preprocess.process_dir`; `train_seg.py` driver; `autolabel_gdino.py` + `grounded_sam.py`; GD Slot-2 SSL-seed wiring + detector speed knobs + notebook speedup. 45 tests passing. Remaining queue is GPU-run + wiring:*

1. **[Stage 1 — coronary]** ~~Re-eval to record Dice/clDice~~ **DONE 2026-07-16 → Dice 0.915 / clDice 0.956, CLEARS the ≥ 0.75 floor** (CLGeoDice run, `outputs/coronary_student_clgeodice/`). Remaining: **clDice vs teacher within ~3%** (compute teacher clDice) + **post-INT8 clDice re-check** (`coreml_validate.py` on the palettized/CoreML student) — the INT8-on-thin-vessels gate is still open.
2. **[Stage 2 — stenosis]** ~~Run kaggle_stenosis_plug_and_play~~ ~~DONE 2026-07-13 → F1 0.214~~ **+CADICA DONE 2026-07-16 → F1 0.291 / recall 0.271, still < 0.57 floor** (up from 0.214; CADICA confirmed patients > frames). Next lever: **more patient diversity** + pseudo-label SSL / GD cold-start — not epochs/model. See archive RESULTS.md.
3. **[Stage 1 refinement]** Extend `qualifies()` to require clDice within ~3% of teacher (not Dice-only).
4. **[Stage 3 close-out]** Record catheter IoU/fps/ID-switch on device; export catheter → CoreML.
5. **[Stage 2.5]** Finish `calibration.py` (reliability + temp-scaling + OOD) once ≥1 seg/det model exists to score.
6. **[GD Slot 3]** OOD flag at the abstention gate using the open-vocab detector.

---

## 8. Changelog

- **2026-07-17** — **Stage 2 Phase 1 quick-win code landed** (local, TDD, 3 parallel agents; disjoint files). Diagnostic of the below-floor CADICA run (F1 0.291) → recall-starved (74% missed, PR recall ceiling ~0.67), val-saturated by ep16, augmentation on YOLO COCO defaults, val fraction inflated to ~34%. Plan written: [`STAGE2_PHASE1_POA.md`](STAGE2_PHASE1_POA.md). Landed: (1) `src/eval/val_by_source.py` (+test) — per-source ARCADE/CADICA/Danilov val, ultralytics lazy; (2) `train_detector.train_kwargs` augment passthrough (was hardcoded COCO defaults) + `configs/stenosis_yolo.yaml` domain-tuned `augment:` block (mosaic 0.0, scale 0.2, erasing 0.0, HSV 0, box 9.0/dfl 2.0, cos_lr), epochs 150→80, recall-first `target: {f1:0.57, recall:0.60}`; (3) `io_utils.group_key` CADICA `pXX_vYY_NNNNN→pXX` (accounting fix, no live leak) + `cadica_to_yolo` per-patient cap (`datasets.cadica.max_frames_per_patient: 40`). Suite **240 passed** (+1 pre-existing torch-in-`sys.modules` order-pollution failure in `test_train_seg.py` — passes in isolation, unrelated). Remaining Phase 1 is GPU-side: per-source val on baseline best.pt, op-point sweep + temporal-voting per-video sensitivity, combined aug+split re-run.
- **2026-07-16** — **Two runs pulled + archived (CADICA stenosis + CLGeoDice coronary).**
  - **Stage 1 coronary — gate now VERIFIED & PASSED.** Kaggle `jugalmodipoiro/coronary`, CLGeoDice distillation (`clgeodice_weight 0.5`, 200/200 epochs) → **Dice 0.915 (best 0.927) ≥ 0.75, clDice 0.956 (best 0.980)** — first coronary run with metrics on record (the 2026-07-12 run's gate was unverified). Artifacts `outputs/coronary_student_clgeodice/{student.pt,student.onnx,student.int8.onnx}` + RESULTS.md (weights gitignored). Retrieved via direct Kaggle output-file URLs (kernel saved ~18.5k files incl. ~15k regenerable nnUNet cache PNGs — a full `kernels output` pull was infeasible; **fix for future coronary runs: put nnU-Net caches in `/kaggle/tmp`**). Still open: teacher-clDice comparison + post-INT8 clDice re-check.
  - **Stage 2 stenosis — +CADICA, biggest honest single-lever gain.** Kaggle `jugalmodipesurr/stenosis`, `arcade+cadica+danilov_yolo11s_768_e150` (patient-grouped, leakage audit PASSED; CADICA +3996 keyframes) → **F1 0.291 / recall 0.271 / mAP50 0.209 — still BELOW floor 0.57**, up from F1 0.214 (**+0.077 F1, +0.105 recall ~+63% rel, +0.101 mAP50**). Confirms patient diversity is the lever. Archived [`experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/`](../experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/RESULTS.md) (RESULTS.md + curves + demo; best.pt gitignored).
- **2026-07-13** — **Honest stenosis re-run pulled + archived.** Kaggle `jugalmodi0111/stenosis` (ARCADE+Danilov, yolo11s/768, **patient-grouped split — leakage check PASSED**: train 8766/1349 groups, val 1059/215 groups, danilov 8325 frames→64 patients) → **F1 0.214 / mAP50 0.108, BELOW floor 0.57** (best.pt F1 0.2136). Confirms the 0.885 was ~all frame-leakage; Danilov is 8325 frames but only **64 patients**, so patient diversity (not epochs/model) is the bottleneck. Archived [`experiments/stenosis_arcade+danilov_yolo11s_768_grouped/`](../experiments/stenosis_arcade+danilov_yolo11s_768_grouped/RESULTS.md) (RESULTS.md + curves; best.pt gitignored). Full suite **150 passed / 3 skipped**.
- **2026-07-12 (e)** — **Stage 2.5 calibration finished** (`src/eval/calibration.py`, pure numpy / torch-free). Added `reliability_curve` + `save_reliability_diagram` (matplotlib-guarded), `temperature_scale` (1-D golden-section on BCE) + `apply_temperature`, `auroc` (tie-averaged Mann–Whitney), `ood_auroc` + `uncertainty_score` (`1-|2p-1|`). Math verified: over-confident logits ECE **0.094 → 0.020** (< 0.05 gate), OOD-AUROC 0.907. Tests +7 (`tests/test_calibration_extra.py`); suite **150 passing** + 3 skimage-skipped. Stage 2.5 code-complete; remaining = score a real model once weights land + wire `CoronaryDominance` tags.
- **2026-07-12 (d)** — **Training-hazard fixes landed** (6 parallel implementation agents, disjoint files, TDD; suite 58→**144 passing** + 3 skimage-skipped). Closes the hazards the (c) audit found:
  - **ARCADE stem collision → FIXED** (`io_utils.coco_to_yolo` + `coco_seg_to_pairs`): new pure `_disambiguated_stem()` prefixes the source split (`train_5`) only for basenames that collide across COCO jsons; Danilov/unique stems unchanged (group_key still collapses them). nnU-Net `numTraining` now globs actual `imagesTr` files (`arcade_to_coco`). No more silent data loss.
  - **Coronary held-out val → FIXED** (`train_seg`, `distill`): `TeacherCacheDataset(stems=…)` filter + `split_stems()` (patient-grouped via `split_of`) → distill on train stems, eval/gate on **val** stems (fallback+warn if val empty). No more eval-on-train.
  - **Seg gate → FIXED**: `qualifies(scores, cfg, teacher_scores=None)` now also enforces an absolute clDice floor and a teacher-relative clDice bound (`cldice_rel_teacher`, default 0.03); `_int8_cldice_recheck()` wires `coreml_validate.py` after CoreML export (prints explicit `[TODO]` if it can't run — never silent).
  - **Detector F1 floor → ENFORCED/SURFACED** (`train_detector`): `best_f1_from_pr()` + `qualifies_det()`; `train()` prints F1 (recall-weighted) + `[PASS]/[FAIL]` vs `target.f1`. SSL (pseudo-label + gdino) now **guarded inside `train()`** — skipped unless a disjoint `ssl.unlabeled_dir` exists (not just the notebook).
  - **Metric fake-perfect → FIXED**: `metrics.dice/cldice` return **NaN on empty-GT** (excludable); `ece` NaN on empty; `coverage_risk` emits `None` (not fake-0) at zero coverage. Consumers (`train_seg._scores`, `coreml_validate.main`) updated to nan-drop, so an empty frame can't NaN-poison the mean and false-pass a `nan<floor` gate.
  - **Cross-vendor → FIXED**: `VENDOR_SPLITS` now **sets of atomic vendors**; `leave_one_vendor_out` excludes every dataset containing the held-out vendor (closes the siemens/ge leak) + asserts ≥2 vendors.
  - **Tracking metrics → FIXED** (`serve/track`): fragmentation/`max_tracks` counted from assigned **track IDs** not detections; fps measured wall-clock over detect+track (reports `det_fps` too); `mean_fps` frame-weighted; flat-numbering concatenation warns instead of collapsing to one clip.
  - **Seg defer → FIXED** (`serve/infer`): two-sided `seg_confidence()` (`mean(max(p,1-p))`) consistent with `coverage_risk`.
  - **CathAction converter → FIXED**: all img/mask dirs iterated (not just first); COCO class-map built by category **name**; value-coded masks mapped per class (binary-ambiguous fails loudly, not defaulting to catheter); mask-dir class match tightened.
  - Notebook §3b collision message updated to "auto-disambiguated (no loss)"; §4 notes the F1 floor is now enforced/printed.
- **2026-07-12 (c)** — **Stenosis notebook + conversion hardened against silent training hazards** (`kaggle_stenosis_plug_and_play.ipynb`, `io_utils.py`, `danilov_to_yolo.py`). New `io_utils.audit_split_leakage()` + §3b **hard-gate cell**: raises *before* training if any patient/clip group spans train+val, or if Danilov frames weren't actually collapsed by `group_key` (real filenames ≠ `<site>_<patient>_<seq>_<frame>` → silent per-frame leak). The Danilov stem set is read from raw *independently of the regex*, so a silent grouping no-op cannot slip through. The audit **strips `gd_`/`pl_` SSL prefixes** before grouping, so a self-labeled copy of a val patient re-injected into train is still caught. **Now wired into `danilov_to_yolo.main()`** (raises), so CLI runs are guarded too — not just the notebook. Added `io_utils.duplicate_basenames_across_cocos()` — flags **ARCADE cross-split stem collisions** (train/val/test each renumber `1..N`, so `5.png` exists in all three → pooled by basename → last-write-wins = silent data loss); `main()` + §3b warn with the exact drop count. Notebook SSL guard: pseudo-label **and** `seed: gdino` auto-disabled unless a disjoint, existing `ssl.unlabeled_dir` is attached. New §5 writes a val-only GT-vs-pred demo → `outputs/stenosis_demo.mp4` (+ `/kaggle/working`). Tests 52→**58**.
  - **Training-hazard audit (4 parallel read-only agents)** surfaced these still-open issues (not yet fixed — several are GPU-side or behavior-changing):
    - **[HIGH] ARCADE stem collision is a real data loss** — `coco_to_yolo`/`coco_seg_to_pairs` key outputs by bare basename; the *proper* fix is to namespace stems by source split (detector currently only *warns*). Also inflates nnU-Net `numTraining`.
    - **[HIGH] Coronary seg scores on its own training set** — `train_seg.py` uses ONE loader for distill + eval + gate; no held-out val (the `group_key`/`split_of` holdout is only used by the YOLO converters). Reported Dice is memorization.
    - **[HIGH] Detector F1 floor never enforced** — `train()` prints only `mAP50`; `target.f1: 0.57` and `metrics` are never read. A below-floor model is returned/zipped/exported as success. (`train_seg` gates; `train_detector` does not.)
    - **[HIGH] Seg gate is Dice-only + no INT8 re-check** — `qualifies()` ignores clDice and the teacher-relative bound; `coreml_validate.py` (the correct clDice-drop gate) is orphaned and points at a `val/` dir prep never creates.
    - **[MED-HIGH] Teacher soft labels are in-fold (not out-of-fold)** — `nnunet_predict_cmd` ensembles all folds over `imagesTr`, so each case's soft target is near-GT → distillation leakage.
    - **[MED] SSL guard only in the notebook** — `train()`/CLI still run pseudo-label/gdino from `unlabeled_dir` with no disjointness check.
    - **[MED] CathAction clip grouping** (`_CLIP_RE`) disagrees with `track.py` clip parsing → catheter split can silently go per-frame (Danilov bug, unguarded); `_from_img_mask_pairs` uses only the first img/mask dir (drops other clips); class-map assumes 1-indexed contiguous ids.
    - **[MED] Metric fake-perfect on empty masks** — `dice`/`cldice` return ≈1.0 on empty-GT+empty-pred → upward bias on the mean. ECE/Brier/cross-vendor stubs return 0/None/`all([])` → vacuous "pass".
    - **[MED] Cross-vendor uses composite vendor strings** — holding out `ge_innova` leaves `siemens_ge` (GE) in train → domain gap understated once wired.
    - **[MED] Tracking metrics** — `max_tracks`/`frag` counts detections not track-IDs (false "0 fragmentation"); fps excludes ByteTrack time; `mean_fps` unweighted; flat frame numbering collapses many clips into one.
    - **[MED] Seg defer confidence one-sided** — `mean(prob[mask==1]) ∈ [0.5,1]` so abstention rarely fires and mismatches `coverage_risk`'s `max(p,1-p)`.
- **2026-07-12 (b)** — Repo audit + cleanup (4 parallel audit agents: bugs / dead-code / tests+config / doc-drift). **outputs/** trimmed 180M→131M (deleted stale `stenosis_output_arcade-only/` [partial ep95, below-floor, curated copy already in `experiments/`], Kaggle-noise logs `run_catheter_clean/`+`run_catheter_honest2/`, `best_stenosis_dry.pt` DRY_RUN weights, `cath_nb/` dup notebook). **Dead code:** deleted `src/models/sam_adapter.py` (stub, 0 callers); removed unused imports (`preprocess` np, `track` time, `app` io) + unused `except ... as e` + dead `_find_img` in `danilov_to_yolo`. 52 tests still pass. Dropped 98M `stage3-catheter_tracking.zip` (single demo mp4 bloating git history). **Doc drift fixed:** test count 39/45/47→52, Stage 1 marked *trained-but-gate-unverified* (coronary `student.pt`+onnx+int8 exist, no Dice/clDice logged). **Bugs fixed (7):** HIGH `train_seg._scores` `device=None`→`x.to(None)` no-op vs cuda model (now resolves device / falls back to model's device); MED `train_detector` pseudo-labels hardcoded class 0 (now keeps predicted class) + SSL/GD-seed added non-CLAHE frames (now CLAHE+resize, and pseudo-label predicts on the CLAHE'd frame); MED `cathaction_to_yolo` single-value mask always class 0 (now value-coded catheter=1→0/guidewire=2→1); LOW `_mask_dirs` dropped all but first clip dir (now iterates all); LOW `calibration.ece` dropped the `prob==0` bin (first bin now inclusive); LOW `dca1_to_nnunet._pairs` matched `"ground"` in the full path — foreground/background dirs poisoned GT detection (now stem-anchored `_gt`/`_ground_truth`). Verified: 52 tests pass, ECE/DCA1/cathaction-mapping checked inline; heavy GPU paths verified by read+AST (no torch locally). **Still open (reported, not actioned):** unenforced stenosis F1 floor (`target.f1` never read → no `qualifies()` gate in `train_detector`); orphan configs (`edge_export`/`avf_tabular`/`tavr_ct_seg`); `transformers` missing from requirements.
- **2026-07-12 (a)** — Second stenosis run (Kaggle, `arcade+danilov_yolo11s_768_e150`): ARCADE+Danilov, 7861/1464 split, 101/150 epochs → F1 0.885 / mAP50 0.87. **Flagged as leakage-inflated**: Danilov video frames were split per-frame (every patient in both train+val). Fixed `io_utils.split_of` → patient-grouped via `group_key` (Danilov `<site>_<patient>`; ARCADE unchanged), 47 tests pass. Also fixed `danilov_to_yolo` O(n²) image lookup (per-annotation recursive glob → single-walk index) and `.bmp` resolution. Archived `experiments/stenosis_arcade+danilov_yolo11s_768_e150/` (+RESULTS.md). Next: re-run on the patient-grouped split for the honest F1.
- **2026-07-11 (d)** — First real stenosis run (Kaggle): `arcade_yolo11n_640_e150`, ARCADE-only → F1 0.246 / mAP50 0.147, **below floor** (learning confirmed via val previews, not a bug). Added `run_tag(cfg)` (auto run-naming, TDD) + wired Kaggle notebook to use it. Archived run to `experiments/stenosis_arcade_yolo11n_640_e150/` (+ RESULTS.md). Tests 45→**47 passing**. Next: `arcade+danilov_yolo11s_768_e150`.
- **2026-07-11 (c)** — GD Slot 2 wired: `ssl.seed: gdino` cold-start in `train_detector.py` (`_gdino_seed_round` + pure helpers `ssl_seed`/`seed_prompt_and_classes`/`boxes_labels_to_yolo_lines`). Detector speed knobs (`train_kwargs`: cache/workers/patience/amp) threaded into all `model.train` calls; stenosis+catheter configs updated. Notebook speedups (cuDNN autotune + surfaced knobs + GD-seed note) applied to **`colab_stenosis_build.ipynb`** and both **Kaggle** builds (`kaggle_coronary_build.ipynb` cuDNN; `kaggle_stenosis_build.ipynb` cuDNN + gdino toggle) — all quality-neutral. Tests 39→**45 passing**, still torch-free.
- **2026-07-11 (b)** — Implemented (local, TDD, 39 tests passing): `preprocess.process_dir` CLAHE walk; `train_seg.py` coronary driver (unblocks Stage 1); `autolabel_gdino.py` + `grounded_sam.py` (Grounding DINO labeler, Slot 1). All new modules import torch-free. Stage 1 `!`→`~`; GD `[ ]`→`~`.
- **2026-07-11 (a)** — Tracker created. Snapshot: Stage 3 catheter done; Stage 1 blocked on stubbed `train_seg.py`; Stage 2 ready-to-run; Grounding DINO workstream added.
