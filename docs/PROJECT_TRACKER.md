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

## 10. Changelog — entries since the 2026-08-13 rebuild

> Earlier entries (2026-07-11 → 2026-08-13) are preserved in full in **Part II §8** below.

- **2026-08-15** — **Tracker rebuilt against ground truth.** Every claim re-verified: 6,907 LOC across 58 files (exact `wc -l`), 621 tests / 42 files, 12 configs, HEAD `e41108a`, clean tree, 4 remaining stubs. Restructured from the old stage-numbered layout (which predated the Dialygo pivot) to workstream-based. Added: §3 safety/legal state with the full audit ledger (12 findings, 9 closed), §4.3 event bus, §7 the complete AVF data-acquisition survey (marked PENDING), §9 documentation contradictions. Corrected the self-contradicting test count, the AVF-as-segmentation description, and the edge-only deployment invariant. **No code changed in this update.**

- **2026-08-13 (e)** — **Observe-only pub/sub event layer** (`src/serve/events.py`, +15 tests). `EventBus` with fnmatch topic patterns, monotonic seq, ISO timestamps; subscriber exceptions counted (`bus.errors`) and swallowed so an observer can never affect a verdict; `subscribe()` itself publishes `bus.subscribed` (announced *before* attaching, so a subscriber sees every registration but its own). `JsonlSink` → `runs/events.jsonl`; `RingBuffer(1000)` backs replay. Orchestrator publishes at every existing seam — `frame.received` / `router.decided` / `router.unavailable` / `model.inferred` / `model.unavailable` / `verdict.emitted`, plus `registry.loaded` per modality at build. New `GET /events` streams SSE (replay + live, bounded by `max_events`/`timeout`). Frame digest reuses `audit.input_hash` so events and the audit trail cross-correlate. Suite 606 → **621**.

- **2026-08-13 (d)** — **Serve layer: video path deleted, three criticals closed.** Per the 2026-08-03 audit §P3 and an explicit user decision (Model One per B3 is single-still-frame). Deleted `temporal_vote.py`, `track.py`, `realtime.py`, `stenosis_infer.py`, `predict_image.py`, `analyze_video` and its helpers, and 49 tests; `/analyze?kind=video` now returns a 400 contract refusal. This closes criticals 1 (false "normal" on confident evidence) and 2 (multi-lesion deletion via float-equality dedupe) **by removal** rather than by patch. Critical 3 fixed directly: `report.to_dict` recursively coerces numpy scalars, so a positive finding no longer 500s (regression test reproduces the exact old failure). Added `RouterUnavailable` — router load/classify failures now yield a deferred report with `defer_reason: "router-unavailable"` instead of escaping as `ModuleNotFoundError` into a generic `analysis-error`, so operators can distinguish "weights missing" from "bug". `configs/orchestrator.yaml` stenosis path repointed to `experiments/…/run/weights/best.pt` (the old `runs/…` path did not exist); `/infer` default repointed to `outputs/coronary_student_clgeodice/student.mlpackage`; `timm>=1.0` added to requirements; `registry.py` file-handle leak fixed. `floor_ok: false` deliberately left alone — stenosis genuinely is below floor. Suite 642 → **606** (−49 +7).

- **2026-08-13 (c)** — **Real static PTQ INT8 export** (`src/export/quantize_int8.py`, 10 → 94 lines, +6 tests). `configs/edge_export.yaml` declared `method: static_ptq, calib_images: 200` while the code ran `quantize_dynamic` — config is the contract, so the code now implements it. `PngCalibrationReader` streams grayscale PNGs matched to the model's own input name and spatial size, caps at what exists on disk and logs "using N of M requested"; `quant_pre_process` runs first with a loud fallback. Verified on the real artifact with the 50 coronary val images: **INT8 Dice vs GT 0.9157 vs fp32 0.9156** (no drop), mean mask agreement 0.9935, size 1.94 MB → 0.52 MB. New artifact `student.int8.static.onnx` alongside the existing dynamic one. `--dynamic` kept as an explicit fallback.

- **2026-08-13 (b)** — **Audit P0.1 clearance-gate hardening** (+25 tests). `--mode` made **required** (no default) on `scan`/`index_dicom`/`extract`/`pixel_deid` CLIs; `deid.py` verified exempt (its CLI only provisions the salt). `--clearance` → `--clearance-override-for-tests` with `dest="clearance"` preserved. **Bug found beyond spec:** argparse's `allow_abbrev=True` meant the *old* bare `--clearance` still worked as an unambiguous prefix, completely defeating the rename — verified live that `--mode real --clearance /tmp/fake_true.yaml` opened the gate. Fixed with `allow_abbrev=False` on all four parsers. `DEFAULT_CLEARANCE_PATH` now resolves from the repo root, not cwd. New `clearance.refuse_synthetic_against_mounted_drive()` wired into `scan_tree` and `build_index`. Item 4 (function-level gates on `extract_series`/`extract_video`) deliberately deferred.

- **2026-08-13 (a)** — **Ingest tasks 13–16 + merge to main.** Task 13 `labels.py` (15 tests) — CSV/COCO/mask-dir adapters and the index↔label join, with PHI quarantine: narrative column *names* are recorded, values never parsed. Task 14 `link.py` (11 tests) — symlinks `data/raw/avf_fistulography` at the clean frame tree; refuses to clobber a real file or directory, idempotent for existing symlinks, `verify_link` never raises. Task 15 `doctor.py` (18 tests) — four crash-safe checks (mounted / links / manifest / no-PHI-in-repo), no clearance gate by design since it must run in any legal state; verified `[ok]` against the real repo. Task 16 wiring — 8 `make ingest-*` targets written against the CLIs **as actually built** (the plan's `MODE=cleared` is not a valid mode; `deid` takes no `--mode`; `extract` is one-file-per-invocation), `configs/ingest_sites.yaml` shipping `drive_roots: []` with a test asserting it stays B5-safe. Branch `feat/ingest-dicom-pipeline` merged to main (`14679c1`) after resolving a `.gitignore` conflict as a union of both sides.

- **2026-08-10** — **Task 12 leakage guard + scan/index robustness.** `_AVF_RE` added to `io_utils.group_key`: `avf_inu_<pid>_s01_00012` → `avf_inu_<pid>` (verified: 200 frames of one patient collapse to 1 group; without it they split 165/35 across train/val and `audit_split_leakage` reported the split clean — the exact F1 0.885 → 0.214 mechanism). Added `avf_stems=` to `audit_split_leakage` as a tripwire so a future regex no-op raises instead of passing. Danilov/CADICA/CathAction/ARCADE grouping byte-for-byte unchanged (34 tests re-run in isolation). Robustness (audit P0.6/P0.7/P0.8): `build_index` writes `index_errors.jsonl` per dropped file plus `n_dicom_rows_seen`/`n_unparsed` counts (closing Task 5's open finding); `os.walk(onerror=…)` logs unreadable directories instead of omitting them; `scan_tree` raises on missing roots (a typo'd drive path used to yield a confident empty inventory); `files.jsonl` fsynced before each checkpoint so resume state cannot vouch for lost rows; `read_jsonl` survives torn multi-byte UTF-8 tails.

- **2026-08-09** — **Ingest tasks 7–11 + HDD driver.** `deid.py` (PS3.15 Annex E scrub, HMAC-SHA256 pseudonyms, per-patient date shift, UID remap under `2.25.`, crosswalk writer; `DEIDENTIFICATION_METHOD` stored as multi-valued LO after a single 248-char string broke the 64-char VR limit; `AdditionalPatientHistory` added to `REMOVE_TAGS` per audit P0.8). `pixel_deid.py` (OCR-free burned-in text detection + masking; **fixed a real negative-origin bug in the plan's reference code** that silently widened boxes with negative origins). `extract.py` (VOI-LUT windowing → 8-bit PNG frames + sidecars; imports `sha256_file` from `manifest` rather than duplicating). `scripts/ingest_hdd.py` — the end-to-end driver, including the three gaps the plan itself left open: per-instance series allocator (P2/T17 — without it every series of a patient collided on `_s01`), batch deid with a `residual_phi` quarantine gate (T18), and crosswalk writing (T19 — `write_crosswalk` had been built in Task 7 and called nowhere). 53 + 11 + 16 tests transcribed, all passing first run against the modules.

- **2026-08-09 (audit)** — **Five-lens read-only audit** → `2026-08-03-audit-remediation-plan.md`. Every finding demonstrated by executing code, not by reading it. Headline findings: the B5/B9 gate did not actually control data access (`--mode` defaulted to synthetic, `--clearance` accepted any path); AVF frames would split per-frame and the leakage auditor would certify the leak clean; the crosswalk was not gitignored under its canonical name; three serve-layer criticals masked by `floor_ok: false`. All P0 items are now closed or partial — see §3.3.

---

# Part II — Tracker as of 2026-08-13 (preserved verbatim)

> Everything below is the previous tracker exactly as it stood at commit `123ddf5`, before the
> 2026-08-15 verification pass. It is kept in full: the stage-by-stage checklists, the per-run
> detail, and the complete changelog back to 2026-07-11 are the project's working history and
> several entries are cited by other docs. Where Part I contradicts Part II, **Part I is current**
> — it was re-verified against the working tree; Part II reflects what was believed on 2026-08-13.
> Known supersessions: test count (150/642 → **621 verified**), AVF described as segmentation
> (→ **classification**, Model One), edge-only deployment invariant (→ **hosted per B8**), serve
> video path (→ **deleted**), and the Stage-numbered scheme (→ **workstreams A–L**).

*Its original header, verbatim:*
> **Last updated:** 2026-08-13 · **Owner:** jugalmodi0111 · **Contact:**
> **Companion docs:** [`Model_Pipeline_Playbook.md`](Model_Pipeline_Playbook.md) (rationale) · [`DATASETS.md`](DATASETS.md) · [`COLAB_MAC_SPLIT.md`](COLAB_MAC_SPLIT.md) · repo [`README.md`](../README.md)

### 0. How to use this file

- `- [x]` done & verified · `- [~]` partial / in-progress · `- [ ]` not started · `- [!]` blocked (reason noted)
- Each stage carries **two gates**: an **accuracy floor** (before edge optimization) and a **safety/sign-off** gate (calibration + cross-vendor). A stage is not "done" until both gates pass on the target device.
- **Golden invariant:** heavy models are *teachers/labelers on the GPU build side only*. Only distilled/quantized students ship to edge (Mac / procedure-cart). Grounding DINO obeys this rule — it is a **build-side labeler**, never shipped.
- Build side = Colab/Kaggle GPU (thin notebooks import `src/*`). Deploy side = Mac CoreML. Local processed splits live on the GPU, not on this laptop — so "no `data/processed/` locally" is expected, not a gap.

---

### 1. Status snapshot (2026-07-16)

| Stage | Title | State | Trained artifact | Gate status |
|---|---|---|---|---|
| 0 | Setup + data prep | `~` partial | — | CLAHE walk **done**; edge-bench torch path still TODO |
| 1 | Coronary segmentation | `x` gate verified & passed | `student.pt`+onnx+int8 (CLGeoDice, 2026-07-16) — `outputs/coronary_student_clgeodice/` | **Dice 0.915 / clDice 0.956 via the CLGeoDice run (2026-07-16) → CLEARS the Dice ≥ 0.75 floor.** Prior `outputs/coronary_student/` run (2026-07-12) was gate UNVERIFIED (Dice/clDice unlogged); this run is verified |
| 2 | Stenosis detection | `!` below floor | honest `best.pt` (F1 0.291) | +CADICA re-run done 2026-07-16 (`arcade+cadica+danilov_yolo11s_768_e150`) → **F1 0.291 / recall 0.271 < 0.57 floor** (up from F1 0.214); CADICA (+3996 keyframes) confirmed patient-diversity is the lever |
| 2.5 | Calibration + abstention | `~` partial | — | ECE, reliability diagram, temperature scaling, OOD-AUROC all implemented (ECE 0.094→0.020, OOD-AUROC 0.907 on synthetic — see §3.2.5); still open: wire `CoronaryDominance` tags into the defer path, score a real model |
| 3 | Temporal + catheter tracking | `x` **done** | `best-catheter.pt` + 4 provenance zips | detection+track complete |
| 3b | Cross-vendor validation | `!` blocked | — | eval harness is a TODO shell |
| 4 | Domain (AVF / TAVR) | `[ ]` not started | — | data-gated (IRB) |
| 5 | Regulatory / intended-use gate | `[ ]` not started | — | name before any non-research use |
| GD | **Grounding DINO labeler** (new) | `~` scaffolded | — | modules + pure helpers done (2026-07-11); SSL-seed wiring pending |

**One-line summary:** Stage 3 (catheter) trained end-to-end. Stage 1 (coronary): CLGeoDice distillation run **done 2026-07-16 → Dice 0.915 / clDice 0.956, CLEARS the Dice ≥ 0.75 floor** — the first coronary run with metrics on record (artifacts in `outputs/coronary_student_clgeodice/`; the prior `outputs/coronary_student/` run had its gate unverified). Stage 2 (stenosis): +CADICA honest patient-grouped re-run **done 2026-07-16 → F1 0.291 / recall 0.271, still BELOW floor 0.57** (up from F1 0.214; CADICA added 3996 keyframes and confirmed patient diversity is the lever). Grounding DINO labeler is scaffolded (modules import torch-free, pure helpers unit-tested). Local test suite: **150 passing** (+3 skipped) (`pytest tests/`).

---

### 2. Code inventory (implemented vs stub)

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
- [x] `src/ingest/` (2026-08-13) — Dialygo institutional fistulography ingest, **synthetic DICOM
  only**: `clearance.py` (B5/B9 mode gate — `VALID_MODES = ("synthetic", "real")`, there is no
  "cleared" mode), `manifest.py` (jsonl / atomic-json / provenance / sha256), `scan.py`
  (read-only drive inventory), `index_dicom.py` (header-only index + SOP dedupe +
  `index_errors.jsonl` drop accounting), `deid.py` (HMAC pseudonyms + PS3.15 tag scrub +
  residual-PHI gate + crosswalk writer — **its CLI only provisions the salt and takes no
  `--mode`**, unlike the other phase CLIs), `extract.py` (VOI-LUT → 8-bit PNG frames at
  `<clean_root>/<site>/frames/<stem_prefix>/fNNNNN.png` + sidecars, one source file per CLI
  invocation), `pixel_deid.py` (OCR-free burned-in-overlay screen + mask, wired into
  `extract_series`), `labels.py` (CSV/COCO/mask-dir adapters + reporting join, B7 verbatim
  passthrough), `link.py` (`data/raw/avf_fistulography` → clean tree, symlink never a copy),
  `doctor.py` (mounted / links / manifest / **no-PHI-in-repo** health check — read-only, no
  clearance gate by design). Driver: `scripts/ingest_hdd.py` runs all five phases end-to-end
  (scan → index → PHI-audit checkpoint → deid → extract) against a mounted drive; resumable,
  per-file failures logged to `*_errors.jsonl` rather than aborting the run. All `src/ingest/`
  modules import torch- and cv2-free (lazy imports inside functions), expose `main()`, run as
  `python -m src.ingest.<module>`. Wiring (Task 16, see §8 changelog): `configs/ingest_sites.yaml`
  (ships with `drive_roots: []`) + `make ingest-scan|index|deid|extract|labels|link|doctor|hdd`.
  Realignment plan T1.7 (`docs/superpowers/plans/2026-08-01-dialygo-realignment.md`): code-complete
  against synthetic DICOM, wiring landed, real-drive run pending B5/B9.

### Stubs / partial (must implement before their stage can pass)
- [x] ~~`src/train/train_seg.py`~~ — **implemented 2026-07-11** (was `NotImplementedError`). No longer blocks Stage 1.
- [x] ~~`src/data_prep/preprocess.py` walk~~ — **`process_dir()` implemented 2026-07-11**.
- [x] ~~`src/models/sam_adapter.py`~~ — **deleted 2026-07-12** (dead `NotImplementedError` stub, 0 callers; superseded by `src/models/grounded_sam.py`).
- [!] `src/data_prep/dsca_sequences.py` (11) — `NotImplementedError`. DSA temporal prep. Blocks Stage 3 DSA.
- [!] `src/train/train_audio.py` (8) — `NotImplementedError`. AVF audio (mel → ViT). Blocks Stage 4 audio.
- [~] `src/eval/calibration.py` (41) — `ece()`, reliability diagram, temperature scaling, and OOD-AUROC all implemented (2026-07-12 e; math verified: ECE 0.094→0.020, OOD-AUROC 0.907). **TODO:** score a real model (Stage-1 seg or Stage-2 det) once weights land; wire `CoronaryDominance` tags into the defer path. Blocks Stage 2.5 full sign-off (numbers are on synthetic data pending a real model).
- [~] `src/eval/cross_vendor.py` (26) — shell only; **TODO:** wire to `train`+`metrics`, emit per-vendor table + worst-case gap. Blocks Stage 3b.
- [~] `src/eval/edge_benchmark.py` (39) — ONNX path works; **TODO:** torch path (param count + cuda/cpu timing).

---

### 3. Stage checklists (detailed)

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
- **Accuracy floor gate:** F1 ≥ 0.57 (earlier text here said 0.55 — 0.57 is the operative floor, matching `configs/stenosis_yolo.yaml target.f1` and the rest of this tracker; corrected 2026-08-02, not silently), **recall-weighted** (a missed stenosis is the costly error). Plain YOLO11n ~0.54 is below floor — step to `s` + SSL, or fall back to RT-DETR-R18.

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

### 4. Grounding DINO integration (new workstream)  `[ ]`

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

### 5. Cross-cutting checklist (applies to every stage)
- [x] Audit trail: input-hash + model version + prediction (`eval/audit.py`)
- [ ] Standardize annotations: COCO JSON (detection) + nnU-Net NIfTI/PNG (semantic)
- [ ] Encoder init from RAD-DINO / BiomedCLIP; SSL on unlabeled angiograms for the grayscale gap
- [ ] Edge metrics on the **real device, INT8**: params(M), FLOPs(G), latency(ms), fps, peak RAM(MB), model size(MB)
- [ ] PhysioNet credentialed access (CITI + DUA) for MIMIC-CXR / VinDr-CXR; register CheXpert

---

### 6. Data inventory (on this laptop, 2026-07-11)
| Dataset | Path | Contents | Use |
|---|---|---|---|
| DCA1 (134 Angiograms) | `datasets/Database_134_Angiograms/` | 134 img + 134 `_gt.pgm` masks | Coronary seg (complete masks, mostly normal) |
| ARCADE syntax | `datasets/arcade/syntax/{train,val,test}` | SYNTAX regions | Coronary seg (task 1) |
| ARCADE stenosis | `datasets/arcade/stenosis/{train,val,test}` | boxes | Stenosis (task 2) |
| Model Selection Matrix | `../Model_Selection_Matrix.xlsx` | scored picks + floors | model choice |
| Dataset Validation Scoring | `../Angiography_Dataset_Validation_Scoring.xlsx` | data QA | data gate |

*XCAD, Danilov, CathAction, DIAS/DSCA, MM-WHS/Seg.A live on the GPU build side (see `DATASETS.md`).*

---

### 7. Immediate next actions (top of the queue)

**Do in this order — each is independently shippable:**

*Done 2026-07-11 (code-side, local, TDD): `preprocess.process_dir`; `train_seg.py` driver; `autolabel_gdino.py` + `grounded_sam.py`; GD Slot-2 SSL-seed wiring + detector speed knobs + notebook speedup. 45 tests passing. Remaining queue is GPU-run + wiring:*

1. **[Stage 1 — coronary]** ~~Re-eval to record Dice/clDice~~ **DONE 2026-07-16 → Dice 0.915 / clDice 0.956, CLEARS the ≥ 0.75 floor** (CLGeoDice run, `outputs/coronary_student_clgeodice/`). Remaining: **clDice vs teacher within ~3%** (compute teacher clDice) + **post-INT8 clDice re-check** (`coreml_validate.py` on the palettized/CoreML student) — the INT8-on-thin-vessels gate is still open.
2. **[Stage 2 — stenosis]** ~~Run kaggle_stenosis_plug_and_play~~ ~~DONE 2026-07-13 → F1 0.214~~ **+CADICA DONE 2026-07-16 → F1 0.291 / recall 0.271, still < 0.57 floor** (up from 0.214; CADICA confirmed patients > frames). Next lever: **more patient diversity** + pseudo-label SSL / GD cold-start — not epochs/model. See archive RESULTS.md.
3. **[Stage 1 refinement]** Extend `qualifies()` to require clDice within ~3% of teacher (not Dice-only).
4. **[Stage 3 close-out]** Record catheter IoU/fps/ID-switch on device; export catheter → CoreML.
5. **[Stage 2.5]** Finish `calibration.py` (reliability + temp-scaling + OOD) once ≥1 seg/det model exists to score.
6. **[GD Slot 3]** OOD flag at the abstention gate using the open-vocab detector.

---

### 8. Changelog

- **2026-08-13** — **`src/ingest/` wiring landed — configs, Makefile, docs (Task 16 of
  `docs/superpowers/plans/2026-08-02-ingest-dicom-pipeline.md`).** Tasks 1–15 of that plan are
  built and tested against **synthetic DICOM only** (Task 6, a standalone PHI-audit CLI, was
  **skipped as a separate module** — its function is covered inline by
  `scripts/ingest_hdd.py`'s `write_phi_audit()` checkpoint, which STOPS a real run until
  `--ack-phi-audit` confirms a human has read the report). This entry adds Task 16, the wiring,
  with four departures from the plan's original text per the 2026-08-03 audit-remediation plan
  (P1, Task 16 row — verdict **blocked**, all four verified live against the actually-built
  CLIs): (1) the plan's `MODE=cleared` is **not a valid mode**
  (`VALID_MODES = ("synthetic", "real")` in `src/ingest/clearance.py`) — the Makefile uses
  `MODE ?= synthetic` with real runs invoked as `MODE=real`; (2) every `make ingest-*` target is
  wired against each CLI's argparse block **as actually built**, not as the plan assumed —
  `scan`/`index_dicom`/`extract`/`pixel_deid` all take a **required** `--mode`, while `deid.py`'s
  CLI only provisions the HMAC salt (`--salt --site --bytes`) and takes **no `--mode` at all**;
  `extract`'s CLI de-identifies and screens one source file per invocation (positional `source`
  argument, not a batch directory), so `ingest-extract` takes a `SOURCE=` variable; (3) this entry
  does not repeat the plan's "every CLI takes `--mode`" claim, because `deid.py` and `doctor.py`
  do not; (4) `pixel_deid.py` is now listed in both the §2 code inventory and the "every phase has
  a CLI" check, which the plan's Definition of Done omitted. New targets: `ingest-scan`,
  `ingest-index`, `ingest-deid` (salt provisioning only), `ingest-extract`, `ingest-labels`,
  `ingest-link`, `ingest-doctor`, and `ingest-hdd` (the `scripts/ingest_hdd.py` end-to-end driver
  — scan → index → PHI audit → deid → extract). All eight were smoke-tested end-to-end against the
  synthetic DICOM fixture (`tests/fixtures/synthetic_dicom.py`) during this change, not just
  `make -n` dry-run. `configs/ingest_sites.yaml` ships with `drive_roots: []` — the B5 gate
  expressed as a file someone has to deliberately edit, with a new config-contract test
  (`tests/test_ingest_doctor.py::test_ingest_sites_config_is_b5_safe`) asserting it stays that
  way; `configs/ingest_clearance.yaml` (the legal gate itself) is untouched by this change and
  still reads `data_agreement_executed: false` / `ip_agreement_executed: false`. The plan's own
  test-count arithmetic (+42/+45) does not apply here — Tasks 1–15 already landed their own tests
  in prior commits, so the suite stood at **641 passing** before this change; this task adds one
  test. Full suite **642 passing** (0 failed) after this change.
  **No real patient data was processed at any point.** Both gates remain CLOSED: **B5**
  (institutional data-use agreement) and **B9** (IP/engagement agreement), per
  `configs/ingest_clearance.yaml`. The HDD driver (`scripts/ingest_hdd.py`) is built and verified
  end-to-end against synthetic DICOM only; a real run against an institutional drive is
  **pending legal clearance** — `require_clearance` refuses `--mode real` until both flags in
  `configs/ingest_clearance.yaml` are an unquoted YAML `true`.
- **2026-08-02** — **Documentation consistency pass** (reconciliation audit, `docs/superpowers/plans/2026-08-01-dialygo-realignment.md` §3b). No code/config/test changes — docs only. (1) **Stenosis floor drift fixed**: the operative floor is **F1 ≥ 0.57** (matches `configs/stenosis_yolo.yaml target.f1`, unchanged); `Model_Pipeline_Playbook.md` (§0, §2.2 heading+metrics, roadmap table) and this tracker's Stage 2 accuracy-floor-gate line (§3.2) previously said 0.55 — corrected to 0.57 with an inline note that 0.55 was the stale figure, not a silent rewrite. `STAGE2_SETUP.md` and `COLAB_MAC_SPLIT.md` already said 0.57 — untouched. (2) **Calibration status contradiction fixed**: the Stage 2.5 summary row (§1) and the calibration code-inventory line (§2) said reliability/temp-scaling/OOD were TODO; the detailed §3.2.5 checklist already had them done (ECE 0.094→0.020, OOD-AUROC 0.907) — summary lines now match, preserving the two genuinely-open items (score a real model; wire `CoronaryDominance` tags into the defer path). (3) **`transformers`-missing claim corrected** in the 2026-07-12(b) changelog entry (§8): `transformers` is present at `requirements.txt:22`; `timm` is the package genuinely missing (needed for the T1.4 frozen-backbone classifier). Old text struck through, not deleted. (4) **`STAGE2_SETUP.md` refreshed**: the intro note and §6 "what's still manual" cited the pre-CADICA F1 0.214 and an unset `target.recall` as current; both superseded 2026-07-17/07-16 — updated to the current honest result (F1 0.291 / recall 0.271 / mAP50 0.209, +CADICA run) and to `target.recall: 0.60` (enabled, still pending clinical sign-off on the value), keeping 0.214 visible as the labeled prior baseline. (5) **Orchestrator plan Task A1 ticked**: `docs/superpowers/plans/2026-07-16-diagnostic-orchestrator.md` Task A1 (the CADICA run) completed 2026-07-16 but was left unchecked; steps now marked done and the gate outcome recorded (**Reject** — F1 0.291/recall 0.271 vs the F1≥0.57 AND recall≥0.60 gate — proceeding to A2 per the task's own branch), archived at `experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/RESULTS.md`. Its stale "F1 0.214 is the anchor" line was rebaselined to 0.291 (0.214 kept as the labeled pre-CADICA figure). Audit note: the audit's second citation for this rebaseline ("around line 865") did not correspond to any stale 0.214 reference at that location — nothing there needed changing.
- **2026-07-18 (b)** — **Dry-run wiring verified + P2.1 harmonizer landed.** Pushed the updated notebook as Kaggle `jugalmodi0111/stenosis` v7 (DRY_RUN=True) → COMPLETE, all new §3c/§3d/§3e/§5b cells ran clean. Confirmed: **CADICA grouping+cap fix works** (leakage audit now `val ~14% by group`, was 34%; danilov 64 patients 0 ungrouped). **§3c annotation QA (model-independent) confirms the convention mismatch**: median box area arcade 0.0108 / cadica 0.0058 / **danilov 0.0029** (danilov `tiny_frac` 0.36) → Danilov is the outlier (matches the arcade-only 0.246 > +danilov 0.214 ablation). Acted on P2.1: **`src/data_prep/harmonize.py`** (+9 tests) clamps tiny boxes to a min w/h floor (config `harmonize.min_box_wh`, notebook §3f `HARMONIZE` flag, TRAIN-ONLY) + **`DROP_DANILOV`** toggle in §3. Fixed my own guardrail test to subprocess-isolate (same fix as `test_train_seg`). Suite **284 passed**. **Real 80-epoch run BLOCKED: jugalmodi0111 hit the 30h/week Kaggle GPU cap** — pending quota reset / alternate account / Colab.
- **2026-07-18** — **Stage 2 Phase 2 tooling landed** (local, TDD, 3 parallel agents; disjoint files). Plan: [`STAGE2_PHASE2_POA.md`](STAGE2_PHASE2_POA.md). (1) **P2.1** `src/eval/annotation_qa.py` (+17 tests) — per-source box-geometry QA (percentiles, `tiny_frac`, boxes/img) to quantify the mAP50→mAP50-95 convention mismatch; notebook §3c. (2) **P2.2** `src/data_prep/balance.py` (+10 tests) — dataset-balanced oversampling (`bal_`-prefixed train-only copies, post-audit → leak-safe), notebook §3d (`BALANCE` flag); XCAD pseudo-label SSL wired via notebook §3e (auto-sets `ssl.unlabeled_dir=data/raw/xcad` when an *xcad* dataset is attached; SSL code already existed in `train_detector`). (3) **P2.3** `yolo_to_coreml.smoketest()` (+7 tests, `--smoketest`) — export + `.mlpackage` sanity-check; commented `yolo11m`/`1024` model toggles in `stenosis_yolo.yaml`. Notebook intro updated; suite **275 passed** (+1 pre-existing unrelated warning). All new modules import torch/cv2/coremltools-free. **Which lever to run is decided by the Phase-1 §5b outputs — GPU runs pending those numbers.**
- **2026-07-17** — **Stage 2 Phase 1 quick-win code landed** (local, TDD, 3 parallel agents; disjoint files). Diagnostic of the below-floor CADICA run (F1 0.291) → recall-starved (74% missed, PR recall ceiling ~0.67), val-saturated by ep16, augmentation on YOLO COCO defaults, val fraction inflated to ~34%. Plan written: [`STAGE2_PHASE1_POA.md`](STAGE2_PHASE1_POA.md). Landed: (1) `src/eval/val_by_source.py` (+test) — per-source ARCADE/CADICA/Danilov val, ultralytics lazy; (2) `train_detector.train_kwargs` augment passthrough (was hardcoded COCO defaults) + `configs/stenosis_yolo.yaml` domain-tuned `augment:` block (mosaic 0.0, scale 0.2, erasing 0.0, HSV 0, box 9.0/dfl 2.0, cos_lr), epochs 150→80, recall-first `target: {f1:0.57, recall:0.60}`; (3) `io_utils.group_key` CADICA `pXX_vYY_NNNNN→pXX` (accounting fix, no live leak) + `cadica_to_yolo` per-patient cap (`datasets.cadica.max_frames_per_patient: 40`). Suite **240 passed** (+1 pre-existing torch-in-`sys.modules` order-pollution failure in `test_train_seg.py` — passes in isolation, unrelated). GPU steps now **wired into `kaggle_stenosis_plug_and_play.ipynb` §5b** (P1.0 per-source val, P1.1a op-point sweep, P1.1b temporal-voting per-video sensitivity over raw CADICA cine — all no-retrain, write `/kaggle/working/phase1_*.txt`); P1.4 combined aug+split re-run flows through the existing Run All on the updated config. Remaining Phase 1 = **run it on GPU**, then archive + pick the operating point / gate reframe.
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
- **2026-07-12 (b)** — Repo audit + cleanup (4 parallel audit agents: bugs / dead-code / tests+config / doc-drift). **outputs/** trimmed 180M→131M (deleted stale `stenosis_output_arcade-only/` [partial ep95, below-floor, curated copy already in `experiments/`], Kaggle-noise logs `run_catheter_clean/`+`run_catheter_honest2/`, `best_stenosis_dry.pt` DRY_RUN weights, `cath_nb/` dup notebook). **Dead code:** deleted `src/models/sam_adapter.py` (stub, 0 callers); removed unused imports (`preprocess` np, `track` time, `app` io) + unused `except ... as e` + dead `_find_img` in `danilov_to_yolo`. 52 tests still pass. Dropped 98M `stage3-catheter_tracking.zip` (single demo mp4 bloating git history). **Doc drift fixed:** test count 39/45/47→52, Stage 1 marked *trained-but-gate-unverified* (coronary `student.pt`+onnx+int8 exist, no Dice/clDice logged). **Bugs fixed (7):** HIGH `train_seg._scores` `device=None`→`x.to(None)` no-op vs cuda model (now resolves device / falls back to model's device); MED `train_detector` pseudo-labels hardcoded class 0 (now keeps predicted class) + SSL/GD-seed added non-CLAHE frames (now CLAHE+resize, and pseudo-label predicts on the CLAHE'd frame); MED `cathaction_to_yolo` single-value mask always class 0 (now value-coded catheter=1→0/guidewire=2→1); LOW `_mask_dirs` dropped all but first clip dir (now iterates all); LOW `calibration.ece` dropped the `prob==0` bin (first bin now inclusive); LOW `dca1_to_nnunet._pairs` matched `"ground"` in the full path — foreground/background dirs poisoned GT detection (now stem-anchored `_gt`/`_ground_truth`). Verified: 52 tests pass, ECE/DCA1/cathaction-mapping checked inline; heavy GPU paths verified by read+AST (no torch locally). **Still open (reported, not actioned):** unenforced stenosis F1 floor (`target.f1` never read → no `qualifies()` gate in `train_detector`); orphan configs (`edge_export`/`avf_tabular`/`tavr_ct_seg`); ~~`transformers` missing from requirements~~ — **correction (2026-08-02): this was mistaken; `transformers` is present at `requirements.txt:22`. `timm` is the package genuinely missing, needed for the frozen-backbone classifier work (T1.4).**
- **2026-07-12 (a)** — Second stenosis run (Kaggle, `arcade+danilov_yolo11s_768_e150`): ARCADE+Danilov, 7861/1464 split, 101/150 epochs → F1 0.885 / mAP50 0.87. **Flagged as leakage-inflated**: Danilov video frames were split per-frame (every patient in both train+val). Fixed `io_utils.split_of` → patient-grouped via `group_key` (Danilov `<site>_<patient>`; ARCADE unchanged), 47 tests pass. Also fixed `danilov_to_yolo` O(n²) image lookup (per-annotation recursive glob → single-walk index) and `.bmp` resolution. Archived `experiments/stenosis_arcade+danilov_yolo11s_768_e150/` (+RESULTS.md). Next: re-run on the patient-grouped split for the honest F1.
- **2026-07-11 (d)** — First real stenosis run (Kaggle): `arcade_yolo11n_640_e150`, ARCADE-only → F1 0.246 / mAP50 0.147, **below floor** (learning confirmed via val previews, not a bug). Added `run_tag(cfg)` (auto run-naming, TDD) + wired Kaggle notebook to use it. Archived run to `experiments/stenosis_arcade_yolo11n_640_e150/` (+ RESULTS.md). Tests 45→**47 passing**. Next: `arcade+danilov_yolo11s_768_e150`.
- **2026-07-11 (c)** — GD Slot 2 wired: `ssl.seed: gdino` cold-start in `train_detector.py` (`_gdino_seed_round` + pure helpers `ssl_seed`/`seed_prompt_and_classes`/`boxes_labels_to_yolo_lines`). Detector speed knobs (`train_kwargs`: cache/workers/patience/amp) threaded into all `model.train` calls; stenosis+catheter configs updated. Notebook speedups (cuDNN autotune + surfaced knobs + GD-seed note) applied to **`colab_stenosis_build.ipynb`** and both **Kaggle** builds (`kaggle_coronary_build.ipynb` cuDNN; `kaggle_stenosis_build.ipynb` cuDNN + gdino toggle) — all quality-neutral. Tests 39→**45 passing**, still torch-free.
- **2026-07-11 (b)** — Implemented (local, TDD, 39 tests passing): `preprocess.process_dir` CLAHE walk; `train_seg.py` coronary driver (unblocks Stage 1); `autolabel_gdino.py` + `grounded_sam.py` (Grounding DINO labeler, Slot 1). All new modules import torch-free. Stage 1 `!`→`~`; GD `[ ]`→`~`.
- **2026-07-11 (a)** — Tracker created. Snapshot: Stage 3 catheter done; Stage 1 blocked on stubbed `train_seg.py`; Stage 2 ready-to-run; Grounding DINO workstream added.