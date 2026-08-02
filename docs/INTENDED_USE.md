# Intended Use & Regulatory Posture

**Purpose:** this is the repo's Stage 5 gate — [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) §3.5 names it
*"Regulatory / intended-use gate … before any non-research use"* — and it doubles as the Dialygo B1/B8
SaMD intended-use statement required by the clinical partner
([`Dialygo_Orientation_and_Requirements.md`](Dialygo_Orientation_and_Requirements.md)). It is a
**blocking gate**: it must exist and be agreed by the clinical lead before any model in this repository
is used for anything other than research.

**Status of this document:** drafted by engineering from the source documents below. It has **not**
been reviewed or signed off by a clinical stakeholder. Until it is, treat every claim in it as
provisional and every model in this repo as research-only (§7).

**Source documents this is grounded in** — do not extend claims beyond what they say:
[`Dialygo_Orientation_and_Requirements.md`](Dialygo_Orientation_and_Requirements.md) §B1, B3, B6, B7, B8 ·
[`superpowers/plans/2026-08-01-dialygo-realignment.md`](superpowers/plans/2026-08-01-dialygo-realignment.md) ·
[`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) §3.5, §3.2 ·
[`Model_Pipeline_Playbook.md`](Model_Pipeline_Playbook.md) §0, §3.1, §5 ·
[`../src/serve/stenosis_triage.py`](../src/serve/stenosis_triage.py) ·
[`../src/serve/router.py`](../src/serve/router.py).

---

## 1. Intended use statement

The system is a **decision-support tool that helps a clinician decide whether a patient needs
referral to a specialist.** It does not diagnose and does not treat.

**Named intended user:** a general nephrologist without interventional training
([`Dialygo_Orientation_and_Requirements.md`](Dialygo_Orientation_and_Requirements.md) §B1, §A8). This
is the person who sees the majority of dialysis patients but cannot confidently read a vascular-access
image themselves.

**Named intended use (Model One, Dialygo):** given a single de-identified angiographic still frame of
the juxta-anastomotic segment of an arteriovenous fistula (AVF), the system produces a triage
suggestion — refer / reasonable to observe / uncertain — with a calibrated confidence, to help the
intended user decide whether to refer the patient to an interventional specialist
(§B1, §B2, §B3). This is currently the **only** named intended use this document covers; extending it
to other segments of the access circuit, other access types, or other imaging domains requires this
document to be revised and re-agreed, not assumed.

**Other models in this repository** (coronary vessel segmentation, coronary stenosis detection,
catheter/guidewire tracking) are transfer-learning donors and research artifacts for the AVF work
(`superpowers/plans/2026-08-01-dialygo-realignment.md` §3.1, §4). They have no clinical intended use
today — see §7.

---

## 2. Assistive, not autonomous

The system is **decision-support / screening-with-abstention / second-read.** It is explicitly **not**
an autonomous diagnostic device, and must never be marketed, deployed, or described as one
(`Dialygo_Orientation_and_Requirements.md` §B1: *"not an autonomous diagnostic device … always
supports a clinician's judgment; it never replaces it"*).

This is enforced in the output contract, not just in prose: the code never emits a diagnosis. It
emits a suggestion plus a confidence plus, where warranted, a deferral
(`src/serve/stenosis_triage.py::triage_decision`, `src/serve/router.py::decide_modality`).

**Required output copy convention** — this binds any UI or report text built on top of these
functions:

| Situation | Internal signal | Required copy | Forbidden copy |
|---|---|---|---|
| A finding was kept above the confidence threshold | `reason="confident"` | "Possible finding — clinician review required" | "Diagnosis: stenosis" / any bare positive claim |
| Nothing was kept, model confidently negative | `reason="clean"` | "No finding detected — clinician review required" | "Normal" / "Diagnosis: normal" |
| Top confidence lands in the defer band | `reason="low-confidence"`, `deferred=True` | "Uncertain — refer for specialist review" | Any copy implying a resolved answer |
| A sub-threshold detection sits near the band | `reason="no-detection-uncertain"`, `deferred=True` | "Possible finding — clinician review required" (never reported as clean) | "No finding" |
| Input is out-of-distribution / unfamiliar | `reason="ood"`, `deferred=True` | "Input unfamiliar to the model — refer for review" | Any confidence number presented as trustworthy |
| Input fails the modality/view validity gate | `router.decide_modality` → `modality="unknown"` | "Not a valid vascular-access image for this tool" | Attempting to read it anyway |

The rule that generalizes across all rows: **the word "diagnosis" never appears in system output.**
Output is always a suggestion to weigh (§5), and a calibrated confidence is shown, never a bare
yes/no (`Dialygo_Orientation_and_Requirements.md` §B3).

---

## 3. SaMD positioning

The system is positioned as **decision-support Software as a Medical Device (SaMD)**
(`Dialygo_Orientation_and_Requirements.md` §B8). Its intended-use claim must stay inside that
boundary: triage suggestion + calibrated confidence, for a named user, on a named input type
(§1). Any use outside that boundary — autonomous diagnosis, treatment planning, treatment
delivery, use by an unnamed user population, use on an input type not covered by validation — is
out of scope and requires this document to be extended before it is attempted.

**Undetermined — the specific SaMD risk class** (e.g. FDA Class II software, EU MDR Class IIa,
India CDSCO risk class) depends on jurisdiction, target market, and the specific claim language, none
of which are fixed yet. This document does not assign one. **Owner of that decision: the clinical
lead (Dr. G. Gireesh Reddy) together with regulatory/legal counsel for the relevant jurisdiction(s),**
before any regulatory filing or market claim is made.

---

## 4. What the system does not do

- It does not prescribe treatment.
- It does not perform the intervention (e.g. angioplasty) — that remains the specialist's act
  (`Dialygo_Orientation_and_Requirements.md` §A8, §B1).
- It does not replace clinical judgment; every output is designed to be weighed by a clinician, not
  acted on directly (§B1, §B8).
- It does not define what counts as abnormal — that authority is reserved to the clinical lead (§8).
- It does not operate without a human in the loop; there is no code path in this repo that acts on a
  prediction without a clinician reviewing it first.

---

## 5. Automation-bias resistance

Dialygo §B8 requires the interface **"present a suggestion to weigh, not a conclusion — assistant,
not oracle."** Concretely, this constrains both UI copy and output shape:

- **Copy:** follow the table in §2 exactly. No screen, report, or API response may render a bare
  label ("stenosis" / "normal") without the accompanying review-required phrasing and confidence.
- **Shape:** the output is never a single boolean. `triage_decision()` always returns
  `prediction`, `calibrated_confs`, `deferred`, and `reason` together — a UI built on top of it must
  surface all four, not just the first. Hiding the confidence or the deferral flag to make a screen
  look cleaner is a violation of this posture, not a cosmetic choice.
- **Never silently "clean":** `triage_decision()` is deliberately biased toward deferring whenever a
  finding might be present — a detection that is faint-near-threshold is surfaced
  (`reason="no-detection-uncertain"`) rather than reported as a negative
  (`src/serve/stenosis_triage.py` docstring: *"a missed stenosis is the deadly error … 'wrong but
  confident' is the dangerous mode"*). Any UI must preserve this: it must never collapse a deferred
  case into a "normal" result to simplify the screen.
- **No default action:** the tool has no "accept suggestion" button that changes patient status by
  itself. Referral, observation, and any clinical action are always a human decision recorded outside
  this system.

---

## 6. Deployment posture

Deployment for the Dialygo deliverable is **hosted / central: the tool serves predictions, and model
weights are not distributed** (`Dialygo_Orientation_and_Requirements.md` §B8). This is consistent
with the backbone's licence and is a hard requirement, not an implementation preference.

**This explicitly replaces the on-device / edge posture the rest of this repository was built
around, for the Dialygo deliverable.** The repo's "golden invariant"
(`PROJECT_TRACKER.md` §0: *"only distilled/quantized students ship to edge"*) governed a
procedure-cart deployment target for the coronary/stenosis/catheter work
(`Model_Pipeline_Playbook.md` — every recommendation up to §3 targets "edge / laptop"). That
invariant is not violated by Dialygo, because it is simply not exercised: the
export→INT8→CoreML→edge-bench chain is **out of scope for Model One**
(`superpowers/plans/2026-08-01-dialygo-realignment.md` §3, item 3.5). If a future segment or product
line of this repo is deployed on-device instead of hosted, that is a different deployment posture and
needs its own statement here — do not assume the hosted posture carries over, and do not assume the
edge posture carries over either.

---

## 7. Current status: research-only

**No model in this repository is currently cleared for clinical use.** Concretely, as of this
writing:

| Model | Status | Evidence |
|---|---|---|
| AVF fistulography classifier (Model One) | Not yet trained — blocked on Track 0 (legal/data gates, §10) | `superpowers/plans/2026-08-01-dialygo-realignment.md` §5 Track 0, Track 3 |
| Coronary stenosis detector | **Below its accuracy floor**: F1 0.291 / recall 0.271 against a 0.57 floor | `PROJECT_TRACKER.md` §1, §3.2 |
| Coronary vessel segmentation | Clears its own (provisional) floor (Dice 0.915 / clDice 0.956 ≥ 0.75) | `PROJECT_TRACKER.md` §1, §3.1 |
| Catheter/guidewire tracking | Trained; device-level gates (IoU/fps/ID-switch) unverified | `PROJECT_TRACKER.md` §3.3 |

Clearing a floor is **not** the same as clinical clearance — even the coronary segmentation model,
which passes its stated floor, has no intended-use claim of its own and is a research/transfer-learning
artifact only (§1).

**The accuracy floors themselves are provisional placeholders.** `Model_Pipeline_Playbook.md` §0
states floors are *"editable assumptions to be set with clinical stakeholders,"* and §5 repeats:
*"The accuracy floors are provisional placeholders for a research v0; set them with clinical
stakeholders per intended use before they gate any real decision."* `PROJECT_TRACKER.md` §3.5 lists
this as an open Stage 5 item: *"Set the provisional accuracy floors with clinical stakeholders (they
are placeholders today)."* Until that sign-off happens, a model "passing its floor" is evidence of
engineering progress, not of clinical adequacy.

**Consequence:** every prediction produced by any model in this repo today, from any code path, is a
research output. None may be used to inform an actual patient's care.

---

## 8. Ground truth authority

Per `Dialygo_Orientation_and_Requirements.md` §B7: labels come from the clinical lead, with
multi-reader consensus where feasible, anchored to clinical correlates where available.
**The engineer does not define what counts as "abnormal."** That is a clinical judgment reserved to
the clinical lead — engineering implements the labeling and consensus protocol; it does not set its
clinical content.

A written ground-truth protocol (who labels, the consensus rule, and the operational definition of
"significant stenosis") is a required artifact and is **not yet written**
(`superpowers/plans/2026-08-01-dialygo-realignment.md` §5 Track 1, item T1.2). **Owner: the clinical
lead, Dr. G. Gireesh Reddy** — not the engineering team.

---

## 9. Validation preconditions before any non-research use

All three of the following must hold, together, before any model output in this repo may inform a
real clinical decision (`Dialygo_Orientation_and_Requirements.md` §B6):

1. **Patient-level splits only** — no patient's data may appear in both training and evaluation. The
   splitting machinery for this already exists and is enforced with a pre-training hard gate
   (`io_utils.group_key` / `split_of` / `audit_split_leakage`), but it must be applied to the AVF
   dataset itself once real data is available, not assumed from the coronary/stenosis work.
2. **External validation at a site other than the source institute** — required as a first-class
   deliverable before any performance claim or deployment, not an optional nice-to-have. The harness
   for this (`src/eval/cross_vendor.py`) exists as a leave-one-vendor-out shell and is promoted, per
   the realignment plan, to the leave-one-*site*-out mechanism for this requirement — it is not yet
   wired up or run (`PROJECT_TRACKER.md` §3.3b: *"blocked … eval harness is a TODO shell"*).
3. **Report where the model fails, not only where it succeeds** — any validation report accompanying
   a non-research-use request must include failure analysis (false negatives especially, given the
   recall-first posture in §5), not only aggregate accuracy figures.

None of these three has been satisfied for any model in this repo as of this writing.

---

## 10. Data-handling gate

**No patient data — identifiable or de-identified — may be copied, transmitted, stored, or processed
outside the environment and terms set by the institutional data-use agreement**
(`Dialygo_Orientation_and_Requirements.md` §B5). **Until that agreement is executed, development uses
public, synthetic, or non-patient sample data only.** This is stated in the source document as
non-negotiable, not a preference.

A second, related gate applies: a separate IP/engagement agreement covering ownership and the
engineer's role must be executed **before development begins** (§B9). Both gates are tracked as
Track 0 in the realignment plan and are, as of this writing, **not confirmed executed**
(`superpowers/plans/2026-08-01-dialygo-realignment.md` §5 Track 0, §6 item 2). **Owner of that
confirmation: the project's human principal (tech@manufex.io) and the Institute of
Nephro-Urology / Dr. Reddy** — engineering cannot self-certify that these agreements are in place and
must not proceed to real-data work without an explicit confirmation from that owner.

---

## 11. Open items and their owners

| Item | Status | Owner |
|---|---|---|
| SaMD risk class | Undetermined — jurisdiction-dependent | Clinical lead + regulatory/legal counsel (§3) |
| Ground-truth / "abnormal" definition protocol | Not yet written | Clinical lead, Dr. G. Gireesh Reddy (§8) |
| Accuracy floors — clinical sign-off | Provisional placeholders, not yet reviewed | Clinical stakeholders (§7) |
| IP/engagement agreement (B9) execution | Not confirmed | Project principal (tech@manufex.io) + Institute (§10) |
| Institutional data-use agreement (B5) execution | Not confirmed | Project principal (tech@manufex.io) + Institute (§10) |
| External (cross-site) validation | Not yet run — harness exists as a shell | Engineering, once Track 0 clears (§9) |

This document should be revised whenever any row above resolves, and re-agreed with the clinical lead
whenever §1's named intended use changes.
