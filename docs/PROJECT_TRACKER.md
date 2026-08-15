# Project Tracker - Interventional Imaging Pipeline

**Purpose:** single source of truth for *what is done* and *what is next*. Check boxes as you go.
**Last updated:** 2026-08-15 · **Owner:** jugalmodi0111 · **HEAD at update:** `e41108a` (main, clean tree)
**Verified suite at update:** **621 passing** across 42 test files (`python -m pytest tests/ -q`)
**Companion docs:** [`Model_Pipeline_Playbook.md`](Model_Pipeline_Playbook.md) (rationale) · [`DATASETS.md`](DATASETS.md) · [`INGEST_HDD_RUNBOOK.md`](INGEST_HDD_RUNBOOK.md) · [`Dialygo_Orientation_and_Requirements.md`](Dialygo_Orientation_and_Requirements.md) (B1–B9, binding) · [`INTENDED_USE.md`](INTENDED_USE.md)

---

## 0. How to use this file

- `- [x]` done & verified · `- [~]` partial / in-progress · `- [ ]` not started · `- [!]` blocked (reason noted)
- Each stage carries **two gates**: an **accuracy floor** and a **safety/sign-off** gate (calibration + cross-vendor). A stage is not "done" until both pass.
- **Every number in this file was re-verified against the working tree on 2026-08-15.** Claims that could not be verified are marked so explicitly rather than carried forward.
- **Project posture changed in Aug 2026.** This is now the **Dialygo** engagement: AVF (dialysis vascular access) is the product; coronary/catheter work is a parallel research track that funds the technique, not the deliverable. The B-requirements in `Dialygo_Orientation_and_Requirements.md` override any older statement in this repo.
- **Deployment posture changed too.** B8 mandates **hosted/central** serving. The older "golden invariant" (edge/CoreML/procedure-cart only) is retained below as *research-track* guidance; it does not govern Model One.

---

## 1. Status snapshot (verified 2026-08-15)

| # | Workstream | State | Artifact on disk | Gate status |
|---|---|---|---|---|
| A | **DICOM ingest pipeline (T1.7)** | `x` **code-complete, 16/16 tasks** | `src/ingest/` (10 modules, 2,188 LOC) + `scripts/ingest_hdd.py` | Verified end-to-end on **synthetic DICOM only**. Real-drive run **blocked on B5/B9** (both flags `false`). |
| B | **Leakage guard (P0.2)** | `x` **fixed & tested** | `src/data_prep/io_utils.py` `_AVF_RE` | 200 frames/1 patient → 1 group, verified. `audit_split_leakage(avf_stems=…)` tripwire added. |
| C | **Serve layer** | `~` **hardened; no model behind it** | `src/serve/` (9 modules, 1,090 LOC) | 3 audit criticals **closed**. Every real `/analyze` returns `router-unavailable` — router has no weights. |
| D | **Event bus / observability** | `x` **new, done** | `src/serve/events.py` (94) + `runs/events.jsonl` + `GET /events` | Observe-only; 15 tests. |
| E | **Coronary segmentation** | `x` **gate PASSED** | `outputs/coronary_student_clgeodice/` | Dice **0.915** / clDice **0.956** ≥ 0.75 floor. CoreML 6-bit gate passed. |
| F | **Stenosis detection** | `!` **BELOW floor** | `experiments/…cadica+danilov…/run/weights/best.pt` | F1 **0.291** / recall 0.271 vs floor F1 0.57 / recall 0.60. |
| G | **Catheter tracking** | `~` **trained, never gated** | `outputs/best-catheter.pt` | IoU / fps / ID-switch **never measured**. Tracking code was **deleted** with the video path (see §4). |
| H | **Edge export / INT8** | `x` **static PTQ landed** | `outputs/coronary_student_clgeodice/student.int8.static.onnx` | INT8 Dice **0.9157** vs fp32 0.9156 (zero drop), 1.94 MB → 0.52 MB. |
| I | **Model One (AVF classifier)** | `[ ]` **not started — plan written** | none | Plan: [`plans/2026-08-13-model-one-classifier-scaffold.md`](superpowers/plans/2026-08-13-model-one-classifier-scaffold.md). Awaiting approval. |
| J | **AVF data acquisition** | `!` **PENDING — no public data exists** | none | See §7. Confirmed: zero public AVF fistulography datasets worldwide. |
| K | **Cerebral DSA / TAVR / AVF audio / AVF tabular** | `[ ]` not started | none | Stubs or orphan configs only. |
| L | **Regulatory (Stage 5)** | `[ ]` not started | `INTENDED_USE.md` drafted, unsigned | SaMD class undetermined; `HOSTING_QUESTIONNAIRE.md` is 1 byte. |

**One-line summary:** The ingest pipeline that turns the institutional HDD into a de-identified, patient-grouped PNG frame store is **finished and verified on synthetic data**; the leakage bug that would have silently invalidated every future split is **fixed**; the serve layer's three criticals are **closed** and the video path **deleted**. Nothing trains on real AVF data yet: **B5/B9 are unexecuted**, and a world-wide dataset survey (2026-08-13) confirmed **no public AVF fistulography data exists at all**. The next buildable thing is the Model One classifier scaffold (plan written, awaiting approval); the next *unblockable* thing is legal sign-off.

---

## 2. Code inventory — verified line-by-line 2026-08-15

Total `src/`: **6,907 LOC** across 58 Python files. Counts are exact (`wc -l`).

### 2.1 `src/ingest/` — Dialygo institutional ingest (2,188 LOC, 10 modules) `x`

Built 2026-08-09 → 2026-08-13 across 16 planned tasks. **All modules import torch- and cv2-free** (lazy imports inside functions), expose `main()`, run as `python -m src.ingest.<module>`.

| Module | LOC | Responsibility | Notes verified this pass |
|---|---|---|---|
| `clearance.py` | 138 | B5/B9 legal gate | `VALID_MODES = ("synthetic","real")` — **there is no "cleared" mode**. `refuse_synthetic_against_mounted_drive()` added (P0.1). |
| `manifest.py` | 197 | JSONL append, atomic JSON, resume state, sha256, provenance | `read_jsonl` opens `errors="replace"` (torn-UTF8 tolerant); `fsync_file()` helper added. |
| `scan.py` | 282 | Phase 1 read-only drive walk, magic-byte typing | `onerror=` logs `kind:"unreadable_dir"`; missing roots raise `ValueError`; fsync before each checkpoint. |
| `index_dicom.py` | 301 | Phase 2 header-only index + SOP dedupe | Writes `index_errors.jsonl` (`{path, reason[, kept_copy]}`); counts include `n_dicom_rows_seen`/`n_unparsed`. |
| `deid.py` | 350 | HMAC pseudonyms, PS3.15 scrub, date shift, UID remap, crosswalk | 31 `REMOVE_TAGS`; `DEIDENTIFICATION_METHOD` is a **tuple** (multi-valued LO, 64-char VR limit). **Its CLI provisions the salt only — takes no `--mode`.** |
| `pixel_deid.py` | 178 | OCR-free burned-in overlay detect + mask | Boxes are `(x,y,w,h)`; negative-origin clip **fixed** (plan code widened boxes). |
| `extract.py` | 357 | VOI-LUT → 8-bit PNG frames + sidecars | Imports `sha256_file` from `manifest` (no duplicate). One source file per CLI invocation. |
| `labels.py` | 249 | CSV/COCO/mask-dir adapters + index↔label join | **PHI quarantine:** narrative column *names* recorded, values never parsed. |
| `link.py` | 132 | Symlink clean frames → `data/raw/avf_fistulography` | Refuses to clobber real files/dirs; idempotent symlink replace. |
| `doctor.py` | 204 | Health check | 4 checks: mounted / links / manifest / **no-PHI-in-repo**. Read-only, **no clearance gate by design**. Runs `[ok]` on this repo today. |

**Driver:** `scripts/ingest_hdd.py` — all five phases end-to-end (scan → index → PHI-audit checkpoint → deid → extract), resumable, per-file failures logged not fatal. Includes the **T17/T18/T19 gaps the plan left open**: per-instance series allocator (no stem collisions), batch deid with `residual_phi` quarantine gate, and crosswalk writing.

**Task ledger (plan `2026-08-02-ingest-dicom-pipeline.md`):**
- Tasks 1–5 (skeleton, clearance, manifest, scan, index) — done 2026-08-02..09
- Tasks 7–11 (deid ×2, pixel_deid, extract ×2) — done 2026-08-09
- Task 12 (leakage guard) — done 2026-08-10
- Tasks 13–15 (labels, link, doctor) — done 2026-08-13
- Task 16 (wiring: Makefile, `configs/ingest_sites.yaml`, docs) — done 2026-08-13
- **Task 6 (standalone PHI-audit CLI) — deliberately SKIPPED**: covered inline by `write_phi_audit()` in the driver, which **stops the run** until `--ack-phi-audit`.

### 2.2 `src/serve/` — decision layer (1,090 LOC, 9 modules) `~`

| Module | LOC | State |
|---|---|---|
| `orchestrator.py` | 287 | `x` route → resolve → infer → typed findings → `StudyReport`. C2/C3/C4 fail-safes present. `RouterUnavailable` added. Video path **removed**. |
| `app.py` | 193 | `x` FastAPI: `/health`, `/infer`, `/analyze` (400 on `kind=video`), **`/events`** (SSE). |
| `stenosis_triage.py` | 105 | `x` pure, tested |
| `registry.py` | 95 | `x` fail-safe `floor_ok` parse; file-handle leak fixed |
| `router.py` | 90 | `~` logic fine; **no weights exist**, `timm` in requirements but not installed |
| `events.py` | 94 | `x` **NEW** — `EventBus`, `JsonlSink`, `RingBuffer` |
| `report.py` | 64 | `x` `to_dict` now sanitizes numpy scalars (float32 500 fixed) |
| `diagnosis.py` | 62 | `x` `det_to_findings` / `seg_to_finding` / `study_defer` |
| `infer.py` | 100 | `~` CoreML det/seg wrappers; untested against real weights |

**Deleted 2026-08-13** (video path, per audit P3 + user decision): `temporal_vote.py`, `track.py`, `realtime.py`, `stenosis_infer.py`, `predict_image.py`, plus their tests (49 tests removed) and the `track`/`track-eval`/`realtime` Makefile targets.

### 2.3 Other implemented modules

- `src/data_prep/` (1,626 LOC): `io_utils.py` (431 — grouping, splits, `audit_split_leakage`), `cathaction_to_yolo` (210), `cadica_to_yolo` (203), `autolabel_gdino` (148), `danilov_to_yolo` (140), `harmonize` (108), `balance` (94), `build_router_manifest` (91), `verify_sequence` (61), `preprocess` (50), `dca1_to_nnunet` (48), `arcade_to_coco` (31)
- `src/eval/` (553 LOC): `calibration.py` (167 — ECE, Brier, reliability, temperature scaling, AUROC, OOD), `annotation_qa` (148), `metrics` (77 — Dice/clDice/CLGeoDice/HD95), `val_by_source` (56), `audit` (25)
- `src/export/` (332 LOC): `quantize_int8.py` (**94 — rewritten to real static PTQ 2026-08-13**, was a 10-line dynamic stub), `yolo_to_coreml` (89), `coreml_validate` (83), `to_coreml` (52), `to_onnx` (14)
- `src/train/` (575 LOC): `train_detector.py` (334), `train_seg.py` (233), `train_audio.py` (**8 — stub**)
- `src/models/` (275 LOC): `distill` (108), `clgeodice` (59), `grounded_sam` (62), `seg_student` (46)

### 2.4 Stubs and TODO shells — the honest list (4 remaining)

- [!] `src/data_prep/dsca_sequences.py` (11) — `NotImplementedError`. Blocks cerebral DSA entirely.
- [!] `src/train/train_audio.py` (8) — `NotImplementedError`. Blocks AVF audio. **Now unblockable on public data** (see §7.3).
- [~] `src/eval/cross_vendor.py` (41) — TODO shell. Blocks Stage 3b.
- [~] `src/eval/edge_benchmark.py` (39) — ONNX path works; torch path prints a TODO.

### 2.5 Missing modules Model One needs (none exist yet)

`src/models/frozen_backbone.py` · `src/train/train_classifier.py` · `src/eval/cls_metrics.py` · `src/serve/infer_cls.py` · `cls_to_finding` in `diagnosis.py`. All specified in the Model One plan; none written.

---

## 3. Safety and legal state — the part that governs everything

### 3.1 B5 / B9 gates — **BOTH CLOSED**

`configs/ingest_clearance.yaml` ships `data_agreement_executed: false`, `ip_agreement_executed: false`. No real patient data has been processed at any point in this project.

**Gate hardening completed 2026-08-13 (audit P0.1), verified:**
- `--mode` is **required, no default** on `scan`, `index_dicom`, `extract`, `pixel_deid` CLIs
- `--clearance` renamed `--clearance-override-for-tests`; `allow_abbrev=False` added after a live test proved the old bare `--clearance` still worked as an abbreviation prefix and defeated the rename entirely
- Marker path resolves from repo root, not cwd (a permissive YAML in the working directory no longer opens the gate)
- `scan_tree`/`build_index` refuse `mode="synthetic"` when any source path is under `/Volumes/`
- Driver adds: real-mode refuses `--work`/`--clean-root` inside the repo; PHI-audit checkpoint blocks phases 4–5 until `--ack-phi-audit`
- **Deferred:** function-level gates inside `extract_series`/`extract_video` (audit P0.1 item 4) — needs a design decision on composition with the existing three gate layers.

### 3.2 PHI containment — verified

- `.gitignore` carries `.ingest/`, `*.crosswalk.csv`, `*crosswalk*.csv`, `crosswalk.csv`, `*.salt`, `salt.bin`, `_keys/`, `data/interim/` — **on main** (P0.3/P0.4 closed)
- `doctor.py` sweeps the repo for `.dcm`/crosswalk/salt files and verifies `data/raw` is gitignored — reports `[ok]` today
- Crosswalk (`_keys/crosswalk.csv`, 0600) is the re-identification key: **never in git, never off the drive**
- Event payloads carry hashes/ids/reasons only — a test asserts no ndarray or bytes ever enters an event

### 3.3 Audit findings ledger (`2026-08-03-audit-remediation-plan.md`)

| ID | Finding | State |
|---|---|---|
| P0.1 | Clearance gate did not control data access | **CLOSED** 2026-08-13 (+ abbreviation-bypass bug found and fixed beyond spec) |
| P0.2 | AVF frames split per-frame; auditor certifies the leak | **CLOSED** 2026-08-10 (Task 12 `_AVF_RE` + `avf_stems=` tripwire) |
| P0.3 | Crosswalk not gitignored under canonical name | **CLOSED** |
| P0.4 | Ingest ignore patterns absent from main | **CLOSED** (merged `14679c1`) |
| P0.5 | `index_dicom --out` writes PHI to unvalidated path | **PARTIAL** — driver refuses in-repo output in real mode; module-level `chmod 0600` on the index not implemented |
| P0.6 | Corrupt DICOM dropped with no record | **CLOSED** — `index_errors.jsonl` + counts |
| P0.7 | Unreadable directory silently omitted | **CLOSED** — `onerror` callback |
| P0.8 | Roots unvalidated; durability inversion; torn UTF-8; missing PHI tags; SOP dedupe discards evidence | **MOSTLY CLOSED** — first four done; SOP-dedupe now *logs* the losing copy (`kept_copy`) but still picks by path order |
| P3.1 | False "normal" on confident evidence (`analyze_video`) | **CLOSED by deletion** |
| P3.2 | Multi-lesion deletion in `_flatten_voted` | **CLOSED by deletion** |
| P3.3 | Any positive finding → HTTP 500 (numpy.float32) | **CLOSED** — `report.to_dict` sanitizes |
| P3.4 | Router unguarded → generic `analysis-error` | **CLOSED** — `RouterUnavailable` + distinct defer reason |

---

## 4. Workstream detail

### 4.1 Ingest pipeline `x` — see §2.1. Remaining: the real run.

**The run itself is one command, documented in [`INGEST_HDD_RUNBOOK.md`](INGEST_HDD_RUNBOOK.md).** Preconditions: both clearance flags flipped (a legal act), drive at `/Volumes/INU` (~246 GB, ~754 GB free), pyenv 3.12.9. Expect several hours; interrupt/resume safe.

- [ ] **Flip B5/B9 flags** — owner: legal/institution, not engineering
- [ ] Pilot run `--limit 20` on the real drive
- [ ] Full run 1 (stops at PHI audit) → human reads `phi_audit.md`
- [ ] Full run 2 `--ack-phi-audit` → clean tree + frames + crosswalk
- [ ] Post-run: `make ingest-link`, `make ingest-doctor`, review `qa_review.jsonl` and `deid_quarantine.jsonl` (must be empty)

### 4.2 Serve layer `~`

**What `/analyze` does today, verified live:** builds fine, but the router has no weights and `timm` is not installed → returns HTTP 200 with `deferred: true, defer_reason: "router-unavailable"`. That is the designed fail-safe, not a bug — but it means the endpoint has never served a real prediction.

- [x] Video path deleted; `kind=video` → 400 with a clear message
- [x] float32 JSON serialization fixed (was 500 on every positive finding)
- [x] `RouterUnavailable` fail-safe with distinguishable defer reason
- [x] `configs/orchestrator.yaml` stenosis path repointed to the real weights (was a nonexistent `runs/…` path)
- [x] `/infer` default repointed to the gate-passed CoreML package
- [x] Event bus mirrors every step (§4.3)
- [ ] Install `timm`; train or obtain router weights (no router trainer exists — `build_router_manifest.py` has no consumer)
- [ ] Decide: does Model One even need the modality router, or does the validity gate replace it? (B3 describes a validity gate, not a modality router)

### 4.3 Event bus `x` — new 2026-08-13

Observe-only pub/sub mirroring the pipeline. **Never carries control flow**; a crashing subscriber is counted (`bus.errors`) and skipped, proven by test — the bus cannot alter a clinical verdict by construction.

- Topics: `bus.subscribed`, `registry.loaded` (per modality, with `floor_ok`), `frame.received`, `router.decided`, `router.unavailable`, `model.inferred`, `model.unavailable`, `verdict.emitted`
- Sinks: `runs/events.jsonl` (one JSON line each) + in-memory ring (1,000) for replay
- Live: `curl -N 'localhost:8000/events'` — replays recent, then streams; bounded variants for scripts
- Frame digest is `audit.input_hash` (16-hex), **the same value written to `runs/audit.jsonl`** — the two logs cross-correlate row-for-row
- 15 tests

### 4.4 Coronary segmentation `x` — gate passed, three items open

Dice **0.915** (best mid-run 0.927), clDice **0.956** (best 0.980) ≥ 0.75 floor — CLGeoDice run 2026-07-16, `outputs/coronary_student_clgeodice/`.

- [ ] Teacher-clDice comparison (`qualifies()` gates Dice only; playbook wants clDice within ~3% of teacher)
- [ ] Post-INT8 clDice re-check on the new **static** INT8 artifact (Dice-level evidence suggests it passes easily)
- [ ] SSL pretraining on XCAD unlabeled

### 4.5 Stenosis detection `!` — below floor, parked by design

**F1 0.291 / recall 0.271 / mAP50 0.209** vs floor F1 0.57 / recall 0.60 (`arcade+cadica+danilov_yolo11s_768_e150`, honest patient-grouped split, 2026-07-16). Progression: 0.246 (ARCADE only) → 0.885 (**leakage-inflated, discarded**) → 0.214 (honest) → 0.291 (+CADICA). CADICA remains the biggest honest single-lever gain; patient diversity is the lever, not epochs or model size.

- [ ] **P1.0 per-source val table** (GPU, ~1 hr) — the gate on every later lever; code landed 2026-07-17, never run
- [ ] P1.1 op-point sweep + per-video sensitivity; P1.4 combined aug+split re-run
- [ ] Phase 2 levers (harmonize, balance, SSL) — **hard-ordered behind P1.0**
- [ ] **Open clinical question:** is per-frame F1 even the right gate? The proposed reframe (per-video sensitivity + abstention) needs clinical sign-off and has been pending since 2026-07-17.

### 4.6 Catheter tracking `~` — trained, never gated, tracking code now deleted

`outputs/best-catheter.pt` exists. IoU / fps / ID-switch were **never recorded** — the tracker line has been unchecked since 2026-07-11. The ByteTrack implementation (`src/serve/track.py`) was deleted with the video path on 2026-08-13, so measuring these now requires either restoring it from git history or re-implementing. CathAction is also **not on disk** (`data/` holds only processed coronary val).

- [ ] Decide: close this out (restore tracker + re-download CathAction + measure) or formally park it as research-track debt

### 4.7 Model One — AVF classifier `[ ]` — plan written, not started

Plan: [`2026-08-13-model-one-classifier-scaffold.md`](superpowers/plans/2026-08-13-model-one-classifier-scaffold.md) — 7 TDD tasks, everything testable on synthetic data before real frames exist.

1. `cls_metrics.py` — sensitivity/specificity/confusion, threshold-at-target-sensitivity, bootstrap CIs
2. `frozen_backbone.py` — timm-lazy factory + frozen backbone + linear head (B4); offline `test-tiny` backbone
3. Trainer part 1 — dataset over the ingest frame store; patient-grouped split where overlap is an **assertion failure**
4. Trainer part 2 — head-only training, temperature calibration, threshold from a sensitivity target, `head.pt` + `metrics.json`
5. `infer_cls.py` — hosted torch `ClsModel`, defer band enforced at the model boundary
6. Wiring — `cls_to_finding`, orchestrator `cls` branch, registry entry with `floor_ok: false`
7. End-to-end synthetic proof + tracker update

**Proposed Task 8 (added after the dataset survey, not yet approved):** `angiocad_to_cls.py` + `cadica_to_cls.py` adapters so the proxy path trains through the identical code the real data will use.

**Blocking facts:** floors are `null` in `configs/avf_fistulography.yaml` (B7 sign-off required); backbone bake-off undecided (DINOv2 vs RAD-DINO vs BiomedCLIP — `STAGE_ACCURACY_RESEARCH.md` recorded "REFUTED: DINOv2 wins" so this must be measured, not assumed); `timm` not installed.

---

## 5. Configs — 12 files, state of each

| Config | Floor declared | State |
|---|---|---|
| `coronary_seg.yaml` | `dice: 0.75` | **met** (0.915) |
| `stenosis_yolo.yaml` | `f1: 0.57, recall: 0.60` | **not met** (0.291/0.271); floor itself contested |
| `edge_export.yaml` | `cldice_drop_max: 0.03` | **met** for CoreML; static PTQ now matches the declared `method: static_ptq` |
| `avf_fistulography.yaml` | `sensitivity: null, specificity: null` | **floors unsigned** — code treats null as "not signed off" |
| `orchestrator.yaml` | `floor_ok: false` (stenosis) | truthful; weights path fixed 2026-08-13 |
| `ingest_clearance.yaml` | — | **both flags false** (the legal gate) |
| `ingest_sites.yaml` | — | `drive_roots: []`; a test asserts it stays B5-safe |
| `catheter_track.yaml` | **none** | no `target:` block at all |
| `cerebral_dsa_temporal.yaml` | **none** | orphan — no trainer exists |
| `tavr_ct_seg.yaml` | **none** | orphan — zero code references |
| `avf_audio.yaml` | **none** | stub trainer only |
| `avf_tabular.yaml` | **none** | orphan — lightgbm/xgboost installed, never imported |
| *missing* `router.yaml` | — | referenced by `build_router_manifest.py` docstring; **does not exist** |

---

## 6. Test suite — 621 passing, 42 files (verified 2026-08-15)

Trajectory this month: 374 (main, pre-ingest) → 470 (ingest tasks 1–5) → 572 (tasks 7–12 + robustness) → 616 (tasks 13–15) → 641 (P0.1) → 642 (task 16) → **606** (video path deleted, −49 +7) → 621 (static PTQ +6, events +15, minus concurrent churn).

Ingest coverage: clearance, manifest, scan, index, fixture, deid (53), pixel_deid (11), extract (16), labels (15), link (11), doctor (18), group_key (13). Serve coverage: orchestrator, analyze endpoint, diagnosis, registry, report, router, triage, events (15). 52 warnings, all pre-existing (pydicom VR format, unclosed-file ResourceWarnings in `test_registry.py`).

---

## 7. AVF data acquisition — **PENDING** (survey completed 2026-08-13)

Three parallel verified web sweeps (~110 fetches across Zenodo, Mendeley, Figshare, Dryad, IEEE DataPort, PhysioNet, Kaggle, HuggingFace, Grand-Challenge, DataCite, PubMed). Full report: published artifact + the survey findings summarized here.

### 7.1 The finding: no public AVF fistulography dataset exists anywhere

DataCite query "fistulography" → **0 results**. Every AVF-imaging AI paper keeps its data private or "on reasonable request". This makes the institutional HDD not merely useful but **irreplaceable** — and it means no public benchmark will exist to compare Model One against.

**Two credible holders of real AVF DSA, both request-only — approach as B6 external-validation partners:**
- **UCLA** — Aichi Chien, `aichi@ucla.edu` — 28 patients, pre/post-PTA DSA cine, 3-month failure labels (EngMedicine 2024)
- **Yonsei Severance** — Kichang Han, `wowsaycheese@yuhs.ac` — 40 patients (Clin Kidney J 2023 / KJR 2022)
- [ ] Send both emails (drafts prepared)

### 7.2 Open-access figure harvest — the only permissionless AVF imagery

Verified manifest: **23 CC-BY articles ≈ 114 angiographic frames** (~55–65 pathology-positive, juxta-anastomotic dominant) + 5 NC/ND articles ≈ 66 frames (including the only good "normal fistulogram" examples). Reproducible recipe, all steps verified: Europe PMC REST with `LICENSE:"cc by"` filter (105 hits, ~40 unmined) → `oa.fcgi` per-PMCID license gate + figure tarball → extract.

**Honest ceiling:** enough for a **validity-gate positive set** and **pipeline smoke tests**; **not** classifier training (~180 lossy composite frames, caption-level labels, near-zero normals, no DICOM, no patient structure).
- [ ] Optional: `scripts/harvest_pmc_figures.py` to automate the recipe

### 7.3 Adjacent AVF modalities — both tracks have real public starting points

- **Audio** (`avf_audio.yaml`, stub trainer): **Blood Flow Sound** (figshare, CC-BY, 111 recordings / 45 AV-graft patients) is open **today** — enough to replace the `NotImplementedError`. Request-only prizes: Weill Cornell (433 patients, 2,565 duplex-validated recordings — the field benchmark), Mario Negri VAsound. PhysioNet has nothing.
- **Tabular** (`avf_tabular.yaml`, orphan): **NIDDK HFM** (602 patients, serial ultrasound measurements + maturation outcomes) via proposal+DUA+IRB is the best effort-to-value asset found. NIDDK DAC trials come with the same application. **USRDS is US-located researchers only** — blocked for this team without a US collaborator.
- **Ultrasound imaging:** no public AVF duplex dataset exists (confirmed via the UltraSam US-43d aggregate: 43 open ultrasound datasets, zero dialysis access).
- [ ] Start the NIDDK-CR application (weeks–months latency)
- [ ] Send the two audio data requests

### 7.4 Proxy angiography — the stack improved in 2025/26

For validating code, the backbone bake-off, and GPU workflow on real X-ray angiography (**never for clinical claims** — coronary ≠ AVF anatomy):

| Dataset | Role | License |
|---|---|---|
| **AngioCAD** (413 pts, 7-grade severity) | new **primary** proxy trainer — CADICA at 10× scale | CC-BY |
| **CARDIAG** (5 centers, anti-leakage metadata) | held-out **external test** — proxy B6 rehearsal | CC-BY |
| **CardioSyntax** (1,844 studies, 144 GB video) | backbone bake-off / SSL corpus | CC-BY |
| CADICA / Danilov / ARCADE | secondary trainers; converters already in repo | mixed |
| Dr-SAM, DiGDA, CoronaryDominance | **license-blocked or parked** (NC / NC-SA / ND) | — |

---

## 8. Immediate queue — ordered, with owners

**Blocked on the user / institution (engineering cannot proceed):**
1. **B5/B9 clearance flags** — the single highest-leverage unblock in the project. Gates the HDD run, all real AVF data, external validation, Track 3 entirely. *Asked 2026-08-01; still unanswered.*
2. **Clinical sign-off on floors** — AVF sensitivity/specificity (`null` today), B7 ground-truth protocol, and the per-video-metric reframe for stenosis. Owner: Dr. Reddy.
3. **Realignment plan acceptance** — `2026-08-01-dialygo-realignment.md` is still stamped "proposed" though the repo has been executing it for two weeks. Its three decisions (AVF priority flip, agreement status, hosted-serving posture) were never formally answered.
4. **Approve the Model One plan** (± Task 8 adapters) and pick an execution style.

**Buildable now, no blockers:**
5. Model One scaffold, tasks 1–7 (synthetic-data-testable end to end)
6. Proxy training path: AngioCAD adapter → train → CARDIAG external test (needs Kaggle/Colab GPU)
7. `pip install timm`; decide router-vs-validity-gate
8. AVF audio proof-of-concept on the open figshare set (replaces the `train_audio.py` stub)
9. Doc reconciliation (§9)

**GPU queue (Kaggle/Colab), in strict order:**
10. **P1.0 per-source val table** — hard prerequisite for every other stenosis lever
11. Grounding DINO labeler Slots 1+2 (code landed 2026-07-11, never run on GPU)
12. Phase-1/2 stenosis retrains — only after P1.0 says which lever to pull
13. Backbone bake-off on CardioSyntax (after the Model One scaffold exists)

**Paperwork with long lead times — start now, finish later:**
14. Email UCLA + Yonsei (external validation + data)
15. Email Mario Negri + Weill Cornell (audio)
16. NIDDK-CR application (HFM + DAC)

---

## 9. Known documentation contradictions — to reconcile

Catalogued 2026-08-09, several still open as of this update:

- [x] Test count — **fixed by this update** (was showing both "150 passing" and "642 passing" in one file)
- [x] Stage-4 AVF described as "lightweight U-Net from coronary weights" — **corrected**: Model One is classification, explicitly NOT Dice/segmentation
- [x] Golden invariant (edge/cart) vs B8 (hosted) — **now scoped** as research-track guidance
- [ ] `Model_Pipeline_Playbook.md:68` still mandates **Dice ≥ 0.75 for AVF imaging** — contradicts `avf_fistulography.yaml` ("NOT Dice")
- [ ] **Danilov patient count: 100 (`DATASETS.md`) vs 64 (this tracker, `STAGE2_SETUP.md`)** — all split/leakage reasoning depends on this number; unresolved
- [ ] DSA floor: Dice ~0.85 (tracker) vs ≥ 0.80 (playbook)
- [ ] `src/serve/app.py:1-9` still documents an air-gapped localhost service; B8 mandates hosted
- [ ] `docs/HOSTING_QUESTIONNAIRE.md` is **1 byte** — needs 8 answers (jurisdiction, whether inference leaves the Institute's network and what exactly leaves — this determines whether hosted serving is even legal under B5, weights custody, PHI-in-transit, retention, auth, unreachable behavior, DINOv3 licence)
- [ ] Realignment plan checkboxes T1.1/T1.3/T1.5/T1.6/T1.7 show `[ ]` though all are built
- [ ] Orchestrator plan shows 80/85 steps unchecked though phases B0/C/D/E landed
- [ ] `DATASETS.md` predates the 2026-08-13 survey — needs AngioCAD/CARDIAG/CardioSyntax + the AVF findings
- [ ] No `pipelines/stage5_*.md` though Stage 5 is in the roadmap

---

## 10. Changelog

- **2026-08-15** — **Tracker rebuilt against ground truth.** Every claim re-verified: 6,907 LOC across 58 files (exact `wc -l`), 621 tests / 42 files, 12 configs, HEAD `e41108a`, clean tree, 4 remaining stubs. Restructured from the old stage-numbered layout (which predated the Dialygo pivot) to workstream-based. Added: §3 safety/legal state with the full audit ledger (12 findings, 9 closed), §4.3 event bus, §7 the complete AVF data-acquisition survey (marked PENDING), §9 documentation contradictions. Corrected the self-contradicting test count, the AVF-as-segmentation description, and the edge-only deployment invariant. **No code changed in this update.**

- **2026-08-13 (e)** — **Observe-only pub/sub event layer** (`src/serve/events.py`, +15 tests). `EventBus` with fnmatch topic patterns, monotonic seq, ISO timestamps; subscriber exceptions counted (`bus.errors`) and swallowed so an observer can never affect a verdict; `subscribe()` itself publishes `bus.subscribed` (announced *before* attaching, so a subscriber sees every registration but its own). `JsonlSink` → `runs/events.jsonl`; `RingBuffer(1000)` backs replay. Orchestrator publishes at every existing seam — `frame.received` / `router.decided` / `router.unavailable` / `model.inferred` / `model.unavailable` / `verdict.emitted`, plus `registry.loaded` per modality at build. New `GET /events` streams SSE (replay + live, bounded by `max_events`/`timeout`). Frame digest reuses `audit.input_hash` so events and the audit trail cross-correlate. Suite 606 → **621**.

- **2026-08-13 (d)** — **Serve layer: video path deleted, three criticals closed.** Per the 2026-08-03 audit §P3 and an explicit user decision (Model One per B3 is single-still-frame). Deleted `temporal_vote.py`, `track.py`, `realtime.py`, `stenosis_infer.py`, `predict_image.py`, `analyze_video` and its helpers, and 49 tests; `/analyze?kind=video` now returns a 400 contract refusal. This closes criticals 1 (false "normal" on confident evidence) and 2 (multi-lesion deletion via float-equality dedupe) **by removal** rather than by patch. Critical 3 fixed directly: `report.to_dict` recursively coerces numpy scalars, so a positive finding no longer 500s (regression test reproduces the exact old failure). Added `RouterUnavailable` — router load/classify failures now yield a deferred report with `defer_reason: "router-unavailable"` instead of escaping as `ModuleNotFoundError` into a generic `analysis-error`, so operators can distinguish "weights missing" from "bug". `configs/orchestrator.yaml` stenosis path repointed to `experiments/…/run/weights/best.pt` (the old `runs/…` path did not exist); `/infer` default repointed to `outputs/coronary_student_clgeodice/student.mlpackage`; `timm>=1.0` added to requirements; `registry.py` file-handle leak fixed. `floor_ok: false` deliberately left alone — stenosis genuinely is below floor. Suite 642 → **606** (−49 +7).

- **2026-08-13 (c)** — **Real static PTQ INT8 export** (`src/export/quantize_int8.py`, 10 → 94 lines, +6 tests). `configs/edge_export.yaml` declared `method: static_ptq, calib_images: 200` while the code ran `quantize_dynamic` — config is the contract, so the code now implements it. `PngCalibrationReader` streams grayscale PNGs matched to the model's own input name and spatial size, caps at what exists on disk and logs "using N of M requested"; `quant_pre_process` runs first with a loud fallback. Verified on the real artifact with the 50 coronary val images: **INT8 Dice vs GT 0.9157 vs fp32 0.9156** (no drop), mean mask agreement 0.9935, size 1.94 MB → 0.52 MB. New artifact `student.int8.static.onnx` alongside the existing dynamic one. `--dynamic` kept as an explicit fallback.

- **2026-08-13 (b)** — **Audit P0.1 clearance-gate hardening** (+25 tests). `--mode` made **required** (no default) on `scan`/`index_dicom`/`extract`/`pixel_deid` CLIs; `deid.py` verified exempt (its CLI only provisions the salt). `--clearance` → `--clearance-override-for-tests` with `dest="clearance"` preserved. **Bug found beyond spec:** argparse's `allow_abbrev=True` meant the *old* bare `--clearance` still worked as an unambiguous prefix, completely defeating the rename — verified live that `--mode real --clearance /tmp/fake_true.yaml` opened the gate. Fixed with `allow_abbrev=False` on all four parsers. `DEFAULT_CLEARANCE_PATH` now resolves from the repo root, not cwd. New `clearance.refuse_synthetic_against_mounted_drive()` wired into `scan_tree` and `build_index`. Item 4 (function-level gates on `extract_series`/`extract_video`) deliberately deferred.

- **2026-08-13 (a)** — **Ingest tasks 13–16 + merge to main.** Task 13 `labels.py` (15 tests) — CSV/COCO/mask-dir adapters and the index↔label join, with PHI quarantine: narrative column *names* are recorded, values never parsed. Task 14 `link.py` (11 tests) — symlinks `data/raw/avf_fistulography` at the clean frame tree; refuses to clobber a real file or directory, idempotent for existing symlinks, `verify_link` never raises. Task 15 `doctor.py` (18 tests) — four crash-safe checks (mounted / links / manifest / no-PHI-in-repo), no clearance gate by design since it must run in any legal state; verified `[ok]` against the real repo. Task 16 wiring — 8 `make ingest-*` targets written against the CLIs **as actually built** (the plan's `MODE=cleared` is not a valid mode; `deid` takes no `--mode`; `extract` is one-file-per-invocation), `configs/ingest_sites.yaml` shipping `drive_roots: []` with a test asserting it stays B5-safe. Branch `feat/ingest-dicom-pipeline` merged to main (`14679c1`) after resolving a `.gitignore` conflict as a union of both sides.

- **2026-08-10** — **Task 12 leakage guard + scan/index robustness.** `_AVF_RE` added to `io_utils.group_key`: `avf_inu_<pid>_s01_00012` → `avf_inu_<pid>` (verified: 200 frames of one patient collapse to 1 group; without it they split 165/35 across train/val and `audit_split_leakage` reported the split clean — the exact F1 0.885 → 0.214 mechanism). Added `avf_stems=` to `audit_split_leakage` as a tripwire so a future regex no-op raises instead of passing. Danilov/CADICA/CathAction/ARCADE grouping byte-for-byte unchanged (34 tests re-run in isolation). Robustness (audit P0.6/P0.7/P0.8): `build_index` writes `index_errors.jsonl` per dropped file plus `n_dicom_rows_seen`/`n_unparsed` counts (closing Task 5's open finding); `os.walk(onerror=…)` logs unreadable directories instead of omitting them; `scan_tree` raises on missing roots (a typo'd drive path used to yield a confident empty inventory); `files.jsonl` fsynced before each checkpoint so resume state cannot vouch for lost rows; `read_jsonl` survives torn multi-byte UTF-8 tails.

- **2026-08-09** — **Ingest tasks 7–11 + HDD driver.** `deid.py` (PS3.15 Annex E scrub, HMAC-SHA256 pseudonyms, per-patient date shift, UID remap under `2.25.`, crosswalk writer; `DEIDENTIFICATION_METHOD` stored as multi-valued LO after a single 248-char string broke the 64-char VR limit; `AdditionalPatientHistory` added to `REMOVE_TAGS` per audit P0.8). `pixel_deid.py` (OCR-free burned-in text detection + masking; **fixed a real negative-origin bug in the plan's reference code** that silently widened boxes with negative origins). `extract.py` (VOI-LUT windowing → 8-bit PNG frames + sidecars; imports `sha256_file` from `manifest` rather than duplicating). `scripts/ingest_hdd.py` — the end-to-end driver, including the three gaps the plan itself left open: per-instance series allocator (P2/T17 — without it every series of a patient collided on `_s01`), batch deid with a `residual_phi` quarantine gate (T18), and crosswalk writing (T19 — `write_crosswalk` had been built in Task 7 and called nowhere). 53 + 11 + 16 tests transcribed, all passing first run against the modules.

- **2026-08-09 (audit)** — **Five-lens read-only audit** → `2026-08-03-audit-remediation-plan.md`. Every finding demonstrated by executing code, not by reading it. Headline findings: the B5/B9 gate did not actually control data access (`--mode` defaulted to synthetic, `--clearance` accepted any path); AVF frames would split per-frame and the leakage auditor would certify the leak clean; the crosswalk was not gitignored under its canonical name; three serve-layer criticals masked by `floor_ok: false`. All P0 items are now closed or partial — see §3.3.

- **2026-08-02** — Documentation consistency pass (stenosis floor 0.55 → 0.57 corrected across playbook and tracker; calibration status reconciled; `transformers`-missing claim corrected — `timm` is the genuinely missing package; `STAGE2_SETUP.md` rebaselined to F1 0.291; orchestrator plan Task A1 ticked with the CADICA result formally **rejected** against its gate).

- **2026-07-18 (b)** — Dry-run wiring verified on Kaggle v7; CADICA grouping+cap fix confirmed (val 34% → ~14% by group). §3c annotation QA confirmed the box-convention mismatch: median box area ARCADE 0.0108 / CADICA 0.0058 / **Danilov 0.0029** (`tiny_frac` 0.36) — Danilov is the outlier, matching the ARCADE-only 0.246 > +Danilov 0.214 ablation. `harmonize.py` (+9 tests) clamps tiny boxes train-only. Suite 284. **Real 80-epoch run blocked: Kaggle 30h/week GPU cap hit.**

- **2026-07-18** — Stage 2 Phase 2 tooling: `annotation_qa.py` (+17), `balance.py` (+10), `yolo_to_coreml.smoketest()` (+7). Which lever to run is decided by Phase-1 §5b outputs — GPU runs pending those numbers.

- **2026-07-17** — Stage 2 Phase 1 quick-win code: `val_by_source.py`, augment passthrough (was hardcoded to YOLO COCO defaults) + domain-tuned block, CADICA `group_key` + per-patient cap, recall-first `target: {f1:0.57, recall:0.60}`. Suite 240.

- **2026-07-16** — **Two runs archived.** Coronary CLGeoDice → Dice 0.915 / clDice 0.956, first coronary run with metrics on record, **gate PASSED**. Stenosis +CADICA → F1 0.291 / recall 0.271, still below floor but the biggest honest single-lever gain (+0.077 F1, +63% relative recall); confirms patient diversity is the lever.

- **2026-07-13** — Honest patient-grouped stenosis re-run → F1 0.214. Confirms the 0.885 was ~all frame leakage; Danilov's 8,325 frames are only 64 patients.

- **2026-07-12 (a–e)** — Leakage discovery and the hardening that followed: `split_of` made patient-grouped; `audit_split_leakage()` + notebook hard-gate; ARCADE stem-collision fix; coronary held-out val fix (was scoring on its own training set); seg gate extended beyond Dice-only; detector F1 floor enforced; metrics return NaN on empty instead of fake-perfect; cross-vendor atomic vendor sets; tracking metrics counted from track IDs; two-sided seg defer confidence. Stage 2.5 calibration completed (ECE 0.094 → 0.020, OOD-AUROC 0.907).

- **2026-07-11** — Tracker created; `preprocess.process_dir`, `train_seg.py` driver, Grounding DINO labeler (Slots 1–2), first stenosis run (F1 0.246).
