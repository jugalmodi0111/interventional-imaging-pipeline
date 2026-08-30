# Notebooks — GPU orchestrators (thin)

**Split by design:**

- **`src/*.py` = the library.** All heavy lifting (models, training loops, prep, export, metrics,
  serving) lives here, importable, import-safe (no work runs on `import`).
- **`notebooks/*.ipynb` = thin GPU runners.** They `import src` and call functions. Run them on
  **Colab or Kaggle GPU** — local CPU training/eval is far too slow.

So: edit logic in `src/` (version-controlled, testable); run it from a notebook on a GPU. A notebook
should be a handful of cells: `env.setup()` → prep → train → export handoff.

## The pattern

```python
import sys; sys.path.insert(0, REPO)      # make src importable
from src import env
E = env.setup()                            # Colab Drive / Kaggle / local + nnU-Net paths + device
from src.train.train_detector import train # import the heavy lifting
best = train(cfg, project=f"{E['runs']}/stenosis")
```

## Notebooks — all 8, verified against current `src/` imports

| Notebook | Purpose | Imports | Status |
|---|---|---|---|
| `colab_coronary_build.ipynb` | Coronary seg (research track): nnU-Net teacher → distill `src.models.seg_student`/`distill` | `src.models.seg_student`, `src.models.distill`, `src.eval.metrics` | **OK** — all imports resolve |
| `kaggle_coronary_build.ipynb` | Same as above, Kaggle variant | same as above | **OK** |
| `kaggle_stenosis_arcade_only.ipynb` | Stenosis detector on ARCADE+Danilov only (earlier/simpler variant) | `src.data_prep.danilov_to_yolo`, `src.train.train_detector` | **OK** |
| `kaggle_stenosis_plug_and_play.ipynb` | Main stenosis pipeline: ARCADE+Danilov+CADICA prep, split-leakage audit, harmonize/balance, YOLO train, per-source val, per-video sensitivity/specificity (P1.0–P1.1c) | `src.data_prep.{danilov_to_yolo,cadica_to_yolo,harmonize,io_utils,balance}`, `src.train.train_detector`, `src.eval.{annotation_qa,val_by_source,temporal_vote}` | **OK** — `src.eval.temporal_vote` resolves (restored 2026-08-16 under `src/eval/`, not `src/serve/`) |
| `kaggle_angiocad_acquire.ipynb` (+ `.py` sidecar) | Downloads/extracts the AngioCAD archive on Kaggle and builds the video-level classification corpus | `src.data_prep.angiocad_to_cls` | **OK** |
| `kaggle_angiocad_bakeoff.ipynb` (+ `.py` sidecar) | Feature-caches frozen backbones (DINOv2/v3/v1, ResNet-50) over the AngioCAD corpus and ranks them for the Model One classifier | `src.train.train_classifier`, `src.eval.calibration`, `src.eval.cls_metrics` | **OK** |
| `colab_catheter_build.ipynb` | Catheter/guidewire detector + tracking build (research track) | `src.data_prep.cathaction_to_yolo`, `src.train.train_detector`, **`src.serve.track`** | **BROKEN** — imports `src.serve.track` (ByteTrack), deleted 2026-08-13 with the rest of the realtime/video serving path. Will raise `ImportError` on the tracking cell. Detector-only cells (prep/train, no tracking) still work; do not run the tracking cell until `track.py` is restored or reimplemented (`docs/PROJECT_TRACKER.md` §4.6). |
| `predict_demo.ipynb` | Standalone single-image prediction demo | **`src.serve.predict_image`** | **BROKEN** — `src.serve.predict_image` was deleted 2026-08-13 along with the rest of the video/realtime path. The notebook will fail on its first import cell. For an equivalent still-frame prediction today, use `src.serve.infer` / `src.serve.orchestrator` (via `POST /analyze`) instead, per `docs/PROJECT_TRACKER.md` §4.2. |

**Two of the eight notebooks currently fail on import** — `colab_catheter_build.ipynb`'s tracking
cell and all of `predict_demo.ipynb` — because they target `src/serve` modules removed in the
2026-08-13 video-path deletion (see `docs/PROJECT_TRACKER.md` §4.2, §10 changelog). Neither has been
updated to the current serve API. Fix by either restoring the deleted modules under `src/eval/` (the
pattern already used for `temporal_vote.py`) or rewriting the notebook cells against
`src/serve/orchestrator.py` + `src/serve/infer.py`.

## Colab vs Kaggle

`src/env.py::setup()` handles both:

- **Colab** — mounts Google Drive; persistent root `/content/drive/MyDrive/intv-img`. Data + nnU-Net
  caches + `runs/` live on Drive so a dropped session doesn't lose the teacher.
- **Kaggle** — root `/kaggle/working/intv-img`; attach datasets as **Kaggle Datasets** (they mount
  read-only under `/kaggle/input/`). `runs/` under `/kaggle/working` is downloadable after the run.

## Handoff to the Mac (deploy side, research track only)

Notebooks stop at the portable artifact (seg: `student.pt` state_dict; det: `best.pt`). CoreML
conversion + the clDice gate + on-device benchmark run on the Mac — see `docs/COLAB_MAC_SPLIT.md`.
This handoff applies to the coronary/stenosis/catheter research track; **Model One (AVF classifier)
is hosted-only (Dialygo B8) and does not go through CoreML export.**

```bash
make export-coreml      MODEL=runs/coronary/student.pt      # seg student -> palettized .mlpackage
make export-coreml-yolo MODEL=runs/stenosis/.../best.pt     # YOLO -> .mlpackage (one call)
```
