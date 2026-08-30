# Interventional Imaging Pipeline — Dialygo / Model One

**Primary deliverable: Model One**, an AI decision-support classifier that screens a single
de-identified angiographic still frame for possible juxta-anastomotic stenosis in a haemodialysis
arteriovenous fistula (AVF), built for the **Dialygo** engagement with the Institute of
Nephro-Urology (clinical lead: Dr. G. Gireesh Reddy). The earlier coronary angiography work
(vessel segmentation, stenosis detection, catheter/guidewire tracking) is a **parallel
research/transfer-learning track** that funds the technique and donates code paths to Model One —
it is not the product itself. See [`docs/Dialygo_Orientation_and_Requirements.md`](docs/Dialygo_Orientation_and_Requirements.md)
(the binding B1–B9 requirements) for the full posture.

**Deployment posture: hosted / central serving** (Dialygo B8) — the tool serves predictions over a
network API and model weights are not distributed to the point of care. This *replaces* the
project's earlier on-device/edge posture for Model One. That earlier posture (CoreML export,
INT8 quantization, procedure-cart deployment) is retained only as **research-track** guidance for
the coronary segmentation/stenosis work and does not govern Model One.

**Not a medical device.** No model in this repository is cleared for clinical use — see
[`docs/INTENDED_USE.md`](docs/INTENDED_USE.md), which is drafted but not yet signed off by a
clinical stakeholder. Every prediction in this repo today is a research output.

---

## Status today

*(spot-checked against the working tree; see [`docs/PROJECT_TRACKER.md`](docs/PROJECT_TRACKER.md)
for the maintained, line-by-line version of this list — numbers below can drift as work lands)*

- **Model One (AVF classifier):** code-complete scaffold — the train → serve path is proven
  end-to-end, but **on synthetic frames only**. No model has been trained on a real image: the
  institutional data-use and IP agreements (B5/B9) are unexecuted, so no AVF frames exist on disk,
  and the registry entry ships `floor_ok: false` — every finding defers by construction until a
  clinician signs a sensitivity/specificity floor.
- **Coronary vessel segmentation** (research track): accuracy-floor gate **passed** — Dice 0.915 /
  clDice 0.956 against a 0.75 floor; the CoreML and static-INT8 edge exports also pass their
  clDice-drop gate.
- **Coronary stenosis detection** (research track): **below its per-frame accuracy floor** — F1
  0.291 vs a 0.57 floor. A reframe to per-video sensitivity/specificity (rather than per-frame F1)
  is proposed and evidence-backed but not yet signed off by the clinical lead.
- **Catheter/guidewire tracking** (research track): trained, but its device-level gates (IoU / fps
  / ID-switch) were never measured, and the tracking code itself (ByteTrack) was deleted along with
  the rest of the realtime/video serving path on 2026-08-13.
- **AngioCAD proxy corpus** (a coronary-angiography dataset used to validate the classifier code
  path and run a backbone bake-off — **not** AVF data, and never used for a clinical claim): a
  patient-level severity-sheet adapter (`src/data_prep/angiocad_to_cls.py`) resolves it to videos
  with per-video positive labels; re-verify the current video/patient/positive-rate counts against
  `docs/PROJECT_TRACKER.md` §2.5/§10 before quoting them, since a `parse_series_spec` parsing bug
  affecting the corpus size was found and fixed on 2026-08-28.
- **Test suite:** verify with `python -m pytest tests/ -q` — several agents are landing work
  concurrently, so any number printed here would already be stale.
- **AVF real-world data:** a worldwide survey (2026-08-13) found **no public AVF fistulography
  dataset exists anywhere.** The institutional HDD is the only path to real training data, and it
  is blocked on B5/B9 sign-off (legal/institutional, not an engineering task).

## What actually ships today

- **`src/serve/`** — a hosted decision-support API (`/health`, `/infer`, `/analyze`) that accepts
  **one still frame per request**. There is **no realtime overlay, no video ingestion, and no
  on-device object tracking** in the serve path today: `track.py` (ByteTrack), `realtime.py`,
  `predict_image.py`, `stenosis_infer.py`, and the original `temporal_vote.py` were all deleted
  from `src/serve/` on 2026-08-13 together with the rest of the video path (Model One is
  single-still-frame by design). `temporal_vote.py` was later restored, but under **`src/eval/`**,
  for *offline* cine scoring only (e.g. scoring a detector over a research-track video) — it is not
  reachable from any live request.
- A **B3 validity gate** (`src/serve/validity.py`) screens acquisition plausibility — corrupt,
  blank/blown-out, wrong-shape, or colour input — before any model runs. It does **not** yet
  discriminate imaging modality (e.g. "is this actually an AVF angiogram"); that needs a learned
  OOD head trained on real in-distribution data, which does not exist yet.
- **`src/ingest/`** — the institutional DICOM de-identification and frame-extraction pipeline for
  the Dialygo HDD. Code-complete and verified end-to-end on synthetic DICOM only; the real run
  against the institutional drive is blocked on B5/B9.
- **`src/data_prep/`, `src/train/`, `src/eval/`, `src/export/`, `src/models/`** — dataset adapters,
  training entrypoints, metrics/calibration, and CoreML/ONNX export used by both the Model One
  classifier path and the coronary research track (segmentation, stenosis detection, catheter
  tracking).

## Layout

```
configs/       per-problem YAML (accuracy floors, dataset roots, orchestrator registry)
data/          download + access instructions (no data committed)
notebooks/     thin GPU orchestrators (Colab/Kaggle) — import src, call it; see notebooks/README.md
src/env.py     Colab/Kaggle/local detection + paths
src/ingest     Dialygo institutional DICOM de-identification + frame extraction
src/data_prep  dataset adapters -> COCO / nnU-Net / YOLO / classifier examples + CLAHE
src/models     TinyU-Net student, distillation, frozen-backbone classifier head
src/train      training entrypoints (coronary seg, stenosis detector, AVF classifier, audio stub)
src/eval       Dice/clDice/HD95, calibration, classifier metrics, offline temporal voting, audit
src/export     ONNX / INT8 / CoreML export + clDice gate (research-track edge path)
src/serve      hosted single-frame decision API: validity gate -> orchestrator -> typed finding
pipelines/     stage-by-stage runbooks (some predate the Dialygo pivot — cross-check against
               docs/PROJECT_TRACKER.md before following one verbatim)
docs/          tracker, playbook, dataset docs, intended-use/regulatory posture
```

## Quick start

```bash
conda env create -f environment.yml && conda activate intv-img   # or: pip install -r requirements.txt

# run the test suite
pytest tests/ -q

# serve the hosted decision API locally (single still frame per request)
MODEL=outputs/coronary_student_clgeodice/student.mlpackage TASK=seg \
  uvicorn src.serve.app:app --host 127.0.0.1 --port 8000

# research-track coronary build on GPU (notebooks/colab_coronary_build.ipynb or
# notebooks/kaggle_coronary_build.ipynb), then on the Mac:
make export-coreml   MODEL=runs/coronary/student.pt
make validate-coreml CORE=runs/coronary/student.mlpackage WEIGHTS=runs/coronary/student.pt \
                     IMAGES=data/processed/coronary/val/img MASKS=data/processed/coronary/val/msk
```

See `Makefile` for the full target list, including the `ingest-*` targets for the institutional
DICOM pipeline (blocked on B5/B9) and `train-avf-cls` for the Model One classifier.

## Docs

- [`docs/PROJECT_TRACKER.md`](docs/PROJECT_TRACKER.md) — **live status + checklist**: what's done,
  what's next, per-workstream gates. Start here.
- [`docs/Dialygo_Orientation_and_Requirements.md`](docs/Dialygo_Orientation_and_Requirements.md) —
  the binding B1–B9 clinical/technical requirements for Model One.
- [`docs/INTENDED_USE.md`](docs/INTENDED_USE.md) — intended-use statement and regulatory posture
  (drafted, not yet clinically signed off).
- [`docs/Model_Pipeline_Playbook.md`](docs/Model_Pipeline_Playbook.md) — model-selection rationale
  for the coronary research track (predates the Dialygo pivot; some picks and floors are stale —
  cross-check against `configs/` and the tracker).
- [`docs/DATASETS.md`](docs/DATASETS.md) — datasets used, download links, how each is used.
- [`notebooks/README.md`](notebooks/README.md) — the `.py`-library / `.ipynb`-runner split, and
  which notebooks currently run vs. import a deleted module.

## Status

Not a medical device. Research use only, not for clinical care. No model output in this repo may
inform an actual patient's care until the preconditions in `docs/INTENDED_USE.md` §9 are met.
