# Dialygo Realignment — Plan Delta

**Date:** 2026-08-01 · **Owner:** <01.jugalmodi@gmail.com>
**Status:** proposed — not yet accepted into `PROJECT_TRACKER.md`

This is **not** a new roadmap. It is a delta against the plans already decided, triggered by
[`Dialygo_Orientation_and_Requirements.md`](../../Dialygo_Orientation_and_Requirements.md)
arriving on 2026-08-01 and revealing that the partner's actual product is **AV-fistula
fistulography triage**, not coronary angiography.

---

## 1. The plans already decided (recovered, with evidence)

| # | Artifact | What it decides | State on 2026-08-01 |
|---|---|---|---|
| A | [`Model_Pipeline_Playbook.md`](../../Model_Pipeline_Playbook.md) | Stage 0→5 roadmap, accuracy floors, `teacher → distill → quantize → edge` pattern | Authoritative rationale; unchanged since Rev 2 |
| B | [`PROJECT_TRACKER.md`](../../PROJECT_TRACKER.md) | Per-stage status + immediate queue | Last updated 2026-07-18 |
| C | [`2026-07-16-diagnostic-orchestrator.md`](2026-07-16-diagnostic-orchestrator.md) | The agreed *next build*: router → registry → diagnosis → report → `/analyze` | 6 phases, **85 steps, 0 executed**; all 17 new files missing |
| D | [`STAGE2_PHASE1_POA.md`](../../STAGE2_PHASE1_POA.md) / [`STAGE2_PHASE2_POA.md`](../../STAGE2_PHASE2_POA.md) | Levers P1.x / P2.x to lift stenosis above floor | Code landed + tested; **GPU runs pending** (Kaggle 30 h/week cap hit 2026-07-18) |
| E | `Clinical_Partner_Handover.docx` (v2, 2026-07-18) | What was promised to, and asked of, the clinical partner | Delivered; **the ask is now wrong** — see §3.7 |

**Tracker snapshot carried forward:** Stage 1 coronary segmentation PASSES its gate (Dice 0.915 /
clDice 0.956). Stage 2 stenosis is **below floor** (F1 0.291 / recall 0.271 vs floor 0.57). Stage 3
catheter is functionally done but its on-device gates are unrecorded. Stage 3b cross-vendor is a
shell. Stages 4 (AVF/TAVR) and 5 (regulatory) are not started. Local suite: **284 passing**
(re-verified 2026-08-01).

---

## 2. What changed

Dialygo defines a product the existing plan only ever listed as a far-future row:

- **Partner:** Institute of Nephro-Urology; clinical lead Dr. G. Gireesh Reddy.
- **Asset:** 2,100+ fistulography procedures — the AVF imaging dataset the Playbook called a
  "data desert."
- **Model One:** juxta-anastomotic segment only, **binary** normal vs. significant stenosis.
- **Method:** frozen DINOv3/DINOv2 backbone + lightweight classification head.
- **Deployment:** hosted / central; **weights are not distributed**.
- **Hard legal gates:** no patient data (identifiable *or* de-identified) may be processed outside
  the institutional agreement (B5); a separate IP/engagement agreement must be executed **before
  development begins** (B9).

---

## 3. Conflicts between the decided plan and Dialygo

| # | Decided plan says | Dialygo requires | Resolution |
|---|---|---|---|
| 3.1 | Coronary is the critical path; AVF is Stage 4, "not started, data-gated" (`PROJECT_TRACKER.md:142-148`) | AVF fistulography is **Model One** | Promote AVF imaging to the critical path; coronary becomes the transfer-learning donor and the honest evidence base |
| 3.2 | AVF = two **non-imaging** tracks: audio bruit + tabular surveillance (`configs/avf_audio.yaml`, `configs/avf_tabular.yaml`) | AVF = **imaging** classification | Neither existing config applies. A new `configs/avf_fistulography.yaml` is required |
| 3.3 | Tasks are detection (YOLO boxes) and segmentation (masks) | Binary **classification** per cropped still frame + calibrated confidence | The repo has **no classification training path**. This is net-new code, not a config change |
| 3.4 | AVF imaging = "lightweight U-Net initialized from coronary weights", floor Dice ≥ 0.75 (`Model_Pipeline_Playbook.md:66`) | Frozen DINOv3/DINOv2 + linear head, sample-efficiency is fixed | Playbook §2.4 imaging bullet is superseded. Dice is the wrong metric — sensitivity/specificity/AUROC/ECE replace it |
| 3.5 | **Golden invariant:** only distilled/quantized students ship to edge; CoreML on the cart, offline (`PROJECT_TRACKER.md:13`) | Hosted/central serving; weights never distributed (B8) | The invariant is not violated — it is *not exercised*. The whole export→INT8→CoreML→edge-bench chain is **out of scope for Model One** |
| 3.6 | Inputs are cine clips; temporal voting aggregates frames (`src/serve/temporal_vote.py`) | Single still PNG, de-identified, cropped to segment (B3) | Frame-level path only for Model One. Temporal aggregation is deferred, not deleted |
| 3.7 | Handover's single ask: "de-identified, patient-diverse set of **coronary angiography** images or clips" | Partner holds **AVF fistulography** and wants an AVF tool | The ask is mis-targeted. Handover needs a v3 |
| 3.8 | Development proceeds on public data; IRB noted as a future concern | No patient data outside the agreement (B5); no development before the IP agreement (B9) | **Hard gate.** All real-data work blocks until both agreements execute |
| 3.9 | Cross-vendor validation is Stage 3b, "blocked, eval harness is a TODO shell" | External validation at ≥ 1 other site is a **first-class deliverable** (B6) | Stage 3b is promoted from blocked-nice-to-have to a contractual deliverable |
| 3.10 | No label-protocol document exists | Labels come from the clinical lead; multi-reader consensus where feasible; **the engineer does not define "abnormal"** (B7) | A written ground-truth protocol is a new required artifact |
| 3.11 | `STAGE_ACCURACY_RESEARCH.md:63` records **REFUTED:** "DINOv2 wins the 8-patient few-shot regime" — and F3 (`:45`) rates frozen-DINOv2 evidence only *medium*, on CT not angiography | B4 fixes the frozen DINOv3/DINOv2 backbone precisely *for* sample-efficiency | **Not a blocker, but a stated risk.** B4 permits alternatives; the sample-efficiency *requirement* is what is fixed, not the architecture. Plan a backbone bake-off (DINOv2 vs RAD-DINO vs BiomedCLIP) as part of Model One rather than assuming DINOv2 wins |
| 3.12 | `docs/HOSTING_QUESTIONNAIRE.md` is **empty (1 byte)** | Hosted/central serving is mandated (B8) | The empty questionnaire is now on the critical path, not a placeholder |

**Note on 3.4:** the architecture half of Dialygo B4 is *already the repo's own decision* —
`PROJECT_TRACKER.md:160` states that for whole-image classification the choice is a
"**DINOv2/RAD-DINO encoder + linear head**, *not* Grounding DINO," and the orchestrator plan's
Phase B builds exactly that. What is new in Dialygo is the **target** (AVF stenosis, not view/quality)
and the **deployment** (hosted, so no MobileNetV3 distillation step).

---

## 3b. Where the decided plan has gone stale (fix before executing it)

These are internal inconsistencies found while reconciling, independent of Dialygo:

- **Orchestrator plan Task A1 is already complete but unchecked.** The plan
  (`2026-07-16-diagnostic-orchestrator.md:142-159`) asks for the CADICA run; that run finished
  2026-07-16 and is archived at `experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150/`.
  The plan was written against the **F1 0.214** baseline and never updated to 0.291.
- **The stenosis floor is 0.55 in two places and 0.57 in four.** `Model_Pipeline_Playbook.md:13,45,117`
  and `PROJECT_TRACKER.md:116` say 0.55; `PROJECT_TRACKER.md:24`, `STAGE2_SETUP.md:3,80`,
  `COLAB_MAC_SPLIT.md:53` and `configs/stenosis_yolo.yaml:48` say 0.57. Pick one.
- **Calibration status contradicts itself.** `PROJECT_TRACKER.md:25,67` mark reliability /
  temperature-scaling / OOD as TODO; `:118-123` mark all three done. The checklist is correct.
- **`transformers` is already in `requirements.txt:22`** — `PROJECT_TRACKER.md:252` still lists it as
  missing. (`timm` genuinely is missing and is needed for T1.4.)
- **`STAGE2_SETUP.md:8-12,101` is stale** — cites F1 0.214 and says `target.recall` is commented out;
  both superseded on 2026-07-17.
- **`STAGE2_PHASE2_POA.md:97` references a "Phase 3"** that has no POA document.
- **No `pipelines/stage5_*.md`** exists, though Stage 5 is in both the playbook roadmap and the tracker.
- **Kaggle GPU quota** (the 2026-07-18 blocker) is weekly and has long since reset.

---

## 4. What survives unchanged — the reusable core

The realignment invalidates the *domain*, not the *machinery*. These already satisfy Dialygo
requirements and need no rework:

- **Patient-grouped splitting** — `io_utils.group_key` / `split_of` / `audit_split_leakage`, plus the
  notebook hard-gate. This is exactly B5's "split by patient, never by image," already enforced with
  a pre-training raise. The 0.885 → 0.214 leakage correction is the proof it works.
- **Calibration** — `src/eval/calibration.py`: temperature scaling, ECE, reliability curves,
  OOD-AUROC, coverage–risk. This *is* B3's "calibrated confidence, not a bare yes/no."
- **Abstention** — `src/serve/stenosis_triage.py` defer logic. This *is* B3's "when confidence is
  low, default to uncertain / refer, never to a false normal."
- **Audit trail** — `src/eval/audit.py` input-hash + model-version + prediction logging.
- **Diagnostic Orchestrator plan (C), Phases B0 / C / D** — pure-logic, no GPU, no data. The router's
  reject buckets are B3's validity gate; `Finding`/`StudyReport` is the triage output contract; the
  defer chain is the safety posture. The plan already names AVF as a future registry row
  (`2026-07-16-diagnostic-orchestrator.md:1504`).
- **`src/eval/cross_vendor.py`** — the leave-one-vendor-out shell becomes the B6 external-validation
  harness (leave-one-*site*-out).
- **Stage 5 / `docs/INTENDED_USE.md`** — becomes the B1/B8 SaMD intended-use statement.
- **The Danilov-is-not-AVF guard** — already stated in three places (`DATASETS.md:23`,
  `DATASET_VALIDATION.md:21,105`, `STAGE2_SETUP.md:30`). The hygiene that prevents the obvious
  wrong-data mistake is already in the repo.
- **The gate-reframe proposal** — `STAGE2_PHASE1_POA.md:43,289` wants per-video sensitivity +
  abstention instead of per-frame F1, but flags that it "needs clinical sign-off (Stage 5
  intended-use)." Dialygo B7 supplies exactly that authority. The blocked decision is now unblockable.

**This is not a pivot away from the plan — it is the plan's own Stage 4 arriving early.**
`Model_Pipeline_Playbook.md:60` already titles §2.4 "Interventional nephrology / AV fistula
(**data desert — your priority**)." AVF was always named the priority; coronary was executed first
only because it had the ready data (`Model_Pipeline_Playbook.md:124`, "First move"). Dialygo removes
the data desert. The ordering rationale expires with it.

---

## 5. Realigned action queue

### Track 0 — Legal (blocking, non-code)

- [ ] Execute the IP / engagement agreement (Dialygo B9) — *required before development begins*.
- [ ] Execute the institutional data-use agreement (Dialygo B5).
- [ ] Until both are signed: **public, synthetic, or own non-patient data only.** No real
      fistulography touches this machine.

### Track 1 — Buildable today (no real data, no legal dependency)

Ordered by dependency:

- [ ] **T1.1** `docs/INTENDED_USE.md` — orchestrator plan Phase F1, now doubling as the Dialygo
      B1/B8 SaMD posture: decision-support / triage aid, hosted serving, never autonomous Dx.
- [ ] **T1.2** Ground-truth protocol doc (B7) — who labels, consensus rule, what "significant
      stenosis" means, anchored to clinical correlates. Written by the clinical lead, not the engineer.
- [ ] **T1.3** `configs/avf_fistulography.yaml` — Model One: `segment: juxta_anastomotic`,
      `task: binary_classification`, frozen backbone id, linear head, `split: patient`,
      `metrics: [sensitivity, specificity, auroc, ece]`. No Dice, no mAP.
- [ ] **T1.4** Frozen-backbone classification path — `src/models/frozen_backbone.py` +
      `src/train/train_classifier.py`, TDD, heavy imports lazy (repo convention). **Net-new
      capability**; requires adding `timm` to `requirements.txt` (`transformers` already present at
      `requirements.txt:22`). Make the backbone id a config field so DINOv2 / RAD-DINO / BiomedCLIP
      can be compared rather than assumed — see conflict 3.11.
- [ ] **T1.4b** Fill `docs/HOSTING_QUESTIONNAIRE.md` (currently 1 byte). B8's hosted/central
      serving posture cannot be designed without it.
- [ ] **T1.5** Orchestrator Phase C — `report.py` / `registry.py` / `diagnosis.py`. Pure logic, fully
      testable now, and it is the triage output contract Dialygo B3 specifies.
- [ ] **T1.6** Orchestrator Phase B0 — `decide_modality`, the B3 validity gate that rejects
      non-angiogram input instead of reading it.
- [ ] **T1.7** `src/ingest/` — DICOM → de-identified cropped PNG frame pipeline, built and tested
      against **synthetic DICOM only**, so it is ready the moment Track 0 clears.
- [ ] **T1.8** Handover v3 — correct the ask from coronary angiography to AVF fistulography +
      clinician labelling time; state the hosted-serving posture instead of the on-device one.
- [ ] **T1.9** Housekeeping from §3b — settle the 0.55-vs-0.57 stenosis floor, fix the calibration
      status contradiction in `PROJECT_TRACKER.md:25,67`, tick orchestrator Task A1 (already done),
      re-baseline that plan from F1 0.214 → 0.291, drop the stale `transformers`-missing note, and
      refresh `STAGE2_SETUP.md:8-12,101`. Cheap, and it stops the next reader re-deriving all of it.

### Track 2 — Coronary, demoted but not dropped

The Kaggle 30 h/week cap that blocked this on 2026-07-18 is weekly and has long since reset — this
track is **unblocked today**. Still worth running: coronary stenosis is the transfer donor and the
published evidence in the handover. But it is **no longer the critical path to the partner
deliverable.** Note `STAGE2_PHASE1_POA.md:280` — "do not invest Phase 2/3 GPU before the P1.0
per-source table exists"; that ordering still holds within this track.

- [ ] **T2.1** Phase-1 GPU runs: P1.0 per-source val, P1.1 op-point sweep + temporal-voting
      sensitivity, P1.4 combined aug+split re-run.
- [ ] **T2.2** Phase-2 levers, chosen by the P1 §5b numbers: P2.1 harmonize / drop-Danilov,
      P2.2 balance + XCAD SSL, P2.3 CoreML smoke-test.
- [ ] **T2.3** Stage 1 open gates: teacher-clDice comparison, post-INT8 clDice re-check.
- [ ] **T2.4** Stage 3 close-out: record catheter IoU / fps / ID-switch on device.

### Track 3 — After Track 0 clears

- [ ] Ingest the real fistulography archive through `src/ingest/`.
- [ ] Patient-level splits (B5) over the 2,100+ procedures.
- [ ] Train Model One; report sensitivity/specificity/AUROC/ECE, and **where it fails** (B6).
- [ ] External validation at ≥ 1 non-source site (B6) via the promoted `cross_vendor.py` harness.

---

## 6. Decision required from the human

1. **Confirm the priority flip** — AVF fistulography becomes the critical path; coronary continues
   as a parallel research track. Everything in §5 assumes this.
2. **Confirm Track 0 status** — has either agreement been executed? Track 3 cannot start otherwise,
   and Track 1 is sized on the assumption that it cannot.
3. **Confirm the deployment posture change** — Model One is hosted/central, so the CoreML/edge
   export chain is parked for it. That contradicts the handover already sent to the partner.
