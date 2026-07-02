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

## Notebooks
| Notebook | Stage | Imports | GPU |
|---|---|---|---|
| `colab_coronary_build.ipynb` | 1 — coronary seg | nnU-Net teacher + `src.models.distill` | yes |
| `colab_stenosis_build.ipynb` | 2 — stenosis det | `src.train.train_detector` | yes |

## Colab vs Kaggle
`src/env.py::setup()` handles both:
- **Colab** — mounts Google Drive; persistent root `/content/drive/MyDrive/intv-img`. Data + nnU-Net
  caches + `runs/` live on Drive so a dropped session doesn't lose the teacher.
- **Kaggle** — root `/kaggle/working/intv-img`; attach datasets as **Kaggle Datasets** (they mount
  read-only under `/kaggle/input/`). `runs/` under `/kaggle/working` is downloadable after the run.

## Handoff to the Mac (deploy side)
Notebooks stop at the portable artifact (seg: `student.pt` state_dict; det: `best.pt`). CoreML
conversion + the clDice gate + on-device benchmark run on the Mac — see `docs/COLAB_MAC_SPLIT.md`.
```bash
make export-coreml      MODEL=runs/coronary/student.pt      # seg student -> palettized .mlpackage
make export-coreml-yolo MODEL=runs/stenosis/.../best.pt     # YOLO -> .mlpackage (one call)
python -m src.serve.realtime --model runs/coronary/student.mlpackage --task seg --source clip.mp4 --show
```
