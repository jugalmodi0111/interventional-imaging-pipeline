# Project Tracker — Interventional Imaging Pipeline

**Purpose:** single source of truth for *what is done* and *what is next*. Check boxes as you go.
**Last updated:** 2026-07-11 · **Owner:** tech@manufex.io
**Companion docs:** [`Model_Pipeline_Playbook.md`](Model_Pipeline_Playbook.md) (rationale) · [`DATASETS.md`](DATASETS.md) · [`COLAB_MAC_SPLIT.md`](COLAB_MAC_SPLIT.md) · repo [`README.md`](../README.md)

---

## 0. How to use this file

- `- [x]` done & verified · `- [~]` partial / in-progress · `- [ ]` not started · `- [!]` blocked (reason noted)
- Each stage carries **two gates**: an **accuracy floor** (before edge optimization) and a **safety/sign-off** gate (calibration + cross-vendor). A stage is not "done" until both gates pass on the target device.
- **Golden invariant:** heavy models are *teachers/labelers on the GPU build side only*. Only distilled/quantized students ship to edge (Mac / procedure-cart). Grounding DINO obeys this rule — it is a **build-side labeler**, never shipped.
- Build side = Colab/Kaggle GPU (thin notebooks import `src/*`). Deploy side = Mac CoreML. Local processed splits live on the GPU, not on this laptop — so "no `data/processed/` locally" is expected, not a gap.

---

## 1. Status snapshot (2026-07-11)

| Stage | Title | State | Trained artifact | Gate status |
|---|---|---|---|---|
| 0 | Setup + data prep | `~` partial | — | CLAHE walk **done**; edge-bench torch path still TODO |
| 1 | Coronary segmentation | `~` ready-to-run | none | **train driver implemented** (2026-07-11); needs Colab GPU run |
| 2 | Stenosis detection | `~` ready-to-run | none | data on disk, trainer coded, not run |
| 2.5 | Calibration + abstention | `~` partial | — | ECE coded; reliability/temp-scale/OOD are TODO |
| 3 | Temporal + catheter tracking | `x` **done** | `best-catheter.pt` + 4 provenance zips | detection+track complete |
| 3b | Cross-vendor validation | `!` blocked | — | eval harness is a TODO shell |
| 4 | Domain (AVF / TAVR) | `[ ]` not started | — | data-gated (IRB) |
| 5 | Regulatory / intended-use gate | `[ ]` not started | — | name before any non-research use |
| GD | **Grounding DINO labeler** (new) | `~` scaffolded | — | modules + pure helpers done (2026-07-11); SSL-seed wiring pending |

**One-line summary:** Stage 3 (catheter) trained end-to-end. Stage 1 (coronary) driver is now **implemented** (was the blocker) — ready for its Colab GPU run. Stage 2 (stenosis) has data + code ready. Grounding DINO labeler is scaffolded (modules import torch-free, pure helpers unit-tested). Local test suite: **39 passing** (`pytest tests/`).

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
- [x] `tests/` — `test_preprocess.py`, `test_train_seg.py`, `test_autolabel_gdino.py`, `test_train_detector.py` → **45 passing**, all import torch-free
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
- [~] `src/models/sam_adapter.py` (5) — still `NotImplementedError`, but **superseded** by `src/models/grounded_sam.py` (Grounded-SAM path, 2026-07-11). Delete or leave as legacy.
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
- [ ] **Run `notebooks/colab_coronary_build.ipynb` on Colab GPU** (materialize splits → teacher → distill → student.pt) ← next real action
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
- [ ] **Cross the floor** — next run `arcade+danilov_yolo11s_768_e150`: +Danilov (~9.8k imgs), YOLO11n→11s, imgsz 640→768, enable SSL
- [ ] Run naming: `run_tag(cfg)` auto-names each run folder (no clobber); Kaggle notebook wired
- [ ] Pseudo-label SSL round on unlabeled frames (raise recall)
- [ ] Track COCO AP/AR on Danilov
- [ ] Export to CoreML (`yolo_to_coreml.py`) + edge bench on Mac
- **Accuracy floor gate:** F1 ≥ 0.55, **recall-weighted** (a missed stenosis is the costly error). Plain YOLO11n ~0.54 is below floor — step to `s` + SSL, or fall back to RT-DETR-R18.

### 3.2.5 Stage 2.5 — Calibration + abstention  `~`
- [x] `ece()` implemented
- [ ] Reliability diagram plot
- [ ] Post-hoc temperature scaling
- [ ] OOD detector + coverage–risk (defer-to-human) curve; wire `CoronaryDominance` artifact/quality tags
- [ ] Brier score reporting hookup
- **Exit gate:** ECE < ~0.05 after temp-scaling; defer path demonstrably fires on OOD inputs (unfamiliar vendor/view/artifact).

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

1. **[Stage 1 — coronary]** Run `colab_coronary_build.ipynb` on Colab GPU — driver is ready; this is the first real training run. → Dice ≥ 0.75 + clDice.
2. **[Stage 2 — stenosis]** Run the (now faster) `colab_stenosis_build.ipynb` on GPU. → F1 ≥ 0.55 recall-weighted. Optionally flip `ssl.seed: gdino` for the open-vocab cold start.
3. **[Stage 1 refinement]** Extend `qualifies()` to require clDice within ~3% of teacher (not Dice-only).
4. **[Stage 3 close-out]** Record catheter IoU/fps/ID-switch on device; export catheter → CoreML.
5. **[Stage 2.5]** Finish `calibration.py` (reliability + temp-scaling + OOD) once ≥1 seg/det model exists to score.
6. **[GD Slot 3]** OOD flag at the abstention gate using the open-vocab detector.

---

## 8. Changelog
- **2026-07-11 (d)** — First real stenosis run (Kaggle): `arcade_yolo11n_640_e150`, ARCADE-only → F1 0.246 / mAP50 0.147, **below floor** (learning confirmed via val previews, not a bug). Added `run_tag(cfg)` (auto run-naming, TDD) + wired Kaggle notebook to use it. Archived run to `experiments/stenosis_arcade_yolo11n_640_e150/` (+ RESULTS.md). Tests 45→**47 passing**. Next: `arcade+danilov_yolo11s_768_e150`.
- **2026-07-11 (c)** — GD Slot 2 wired: `ssl.seed: gdino` cold-start in `train_detector.py` (`_gdino_seed_round` + pure helpers `ssl_seed`/`seed_prompt_and_classes`/`boxes_labels_to_yolo_lines`). Detector speed knobs (`train_kwargs`: cache/workers/patience/amp) threaded into all `model.train` calls; stenosis+catheter configs updated. Notebook speedups (cuDNN autotune + surfaced knobs + GD-seed note) applied to **`colab_stenosis_build.ipynb`** and both **Kaggle** builds (`kaggle_coronary_build.ipynb` cuDNN; `kaggle_stenosis_build.ipynb` cuDNN + gdino toggle) — all quality-neutral. Tests 39→**45 passing**, still torch-free.
- **2026-07-11 (b)** — Implemented (local, TDD, 39 tests passing): `preprocess.process_dir` CLAHE walk; `train_seg.py` coronary driver (unblocks Stage 1); `autolabel_gdino.py` + `grounded_sam.py` (Grounding DINO labeler, Slot 1). All new modules import torch-free. Stage 1 `!`→`~`; GD `[ ]`→`~`.
- **2026-07-11 (a)** — Tracker created. Snapshot: Stage 3 catheter done; Stage 1 blocked on stubbed `train_seg.py`; Stage 2 ready-to-run; Grounding DINO workstream added.
