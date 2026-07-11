# Interventional Imaging Pipeline (edge / laptop)

**Teacher → distill → quantize → deploy** deep-learning pipeline for interventional imaging:
coronary vessel segmentation, stenosis detection, catheter/guidewire tracking, cerebral DSA,
AV-fistula, and TAVR. Every production model is exportable to run **on-device** (Apple-silicon
laptop / procedure-cart mini-PC); heavy models appear only as GPU **teachers** or **offline** steps.

**Build side = Colab/Kaggle GPU. Inference side = the Mac (CoreML).** Training/testing on a laptop
CPU is far too slow, so all heavy lifting lives in importable `src/*` modules that thin GPU
notebooks call; the portable artifact (student `state_dict` / YOLO `best.pt`) is converted to CoreML
on the Mac.

## What makes it clinical-grade, not just accurate
- **Accuracy floor gate** — a fast, tiny model can't win on speed while sitting below a per-problem
  clinical accuracy floor (see `Model_Selection_Matrix.xlsx`).
- **clDice re-checked after quantization** — INT8/palettization breaks thin vessels even when Dice holds.
- **Calibration + abstention** — ECE ≤ 0.05, defer-to-human path; "wrong but confident" is the danger.
- **Cross-vendor validation** — leave-one-vendor-out across Siemens/GE/Philips.
- **Audit trail** — every inference logs input-hash + model version + prediction.

## Problems, picks & data (v1 core in bold)
| Problem | Edge model | Datasets |
|---|---|---|
| **Coronary seg** | nnU-Net teacher → **TinyU-Net** student → CoreML | **ARCADE** t1, **DCA1**, **XCAD** (SSL) |
| **Stenosis** | **YOLO11n** (+ pseudo-label SSL) | **ARCADE** t2, **Danilov** |
| **Catheter/guidewire** | **YOLO11n** + ByteTrack | **CathAction** |
| Cerebral DSA | keyframe 2D + ConvLSTM (DSANet teacher, offline) | DIAS, DSCA |
| AVF audio / tabular | small ViT / LightGBM | institutional (IRB) |
| TAVR CT / fluoro / risk | 3D nnU-Net (offline) / YOLO / XGBoost | MM-WHS, Seg.A proxies |

See [`docs/DATASETS.md`](docs/DATASETS.md) for download links + how each dataset is used.

## Layout
```
configs/       per-problem YAML (edge-tuned defaults)
data/          download + access instructions (no data committed)
notebooks/     THIN GPU orchestrators (Colab/Kaggle) — import src, call it
src/env.py     Colab/Kaggle/local detection + paths
src/data_prep  standardize datasets -> COCO / nnU-Net / YOLO + CLAHE
src/models     TinyU-Net student, distillation, SAM/LoRA adapter
src/train      training entrypoints (seg distill, YOLO detector, audio)
src/eval       Dice/clDice/HD95, calibration, cross-vendor, edge benchmark, audit
src/export     ONNX / INT8 / CoreML (palettize) + clDice gate
src/serve      on-device CoreML inference, real-time overlay, ByteTrack, FastAPI
pipelines/     stage-by-stage runbooks
docs/          playbook, dataset validation, hosting questionnaire, Colab↔Mac split
```

## Quick start
```bash
conda env create -f environment.yml && conda activate intv-img   # or: pip install -r requirements.txt

# smoke-test (runs today, CPU)
python -m src.eval.metrics
python -m src.eval.edge_benchmark --model <model>.onnx

# coronary: build on Colab GPU (notebooks/colab_coronary_build.ipynb), then on the Mac:
make export-coreml   MODEL=runs/coronary/student.pt
make validate-coreml CORE=runs/coronary/student.mlpackage WEIGHTS=runs/coronary/student.pt \
                     IMAGES=data/processed/coronary/val/img MASKS=data/processed/coronary/val/msk
make bench-coreml    MODEL=runs/coronary/student.mlpackage
```

## Stages (see `pipelines/`)
0 setup/prep · 1 coronary seg · 2 stenosis · 2.5 calibration · 3 temporal+catheter ·
3b cross-vendor · 4 domain (AVF/TAVR). Each perception stage has two gates: accuracy floor
*before* edge optimization, calibration + cross-vendor *before* sign-off.

## Docs
- [`docs/PROJECT_TRACKER.md`](docs/PROJECT_TRACKER.md) — **live status + checklist**: what's done, what's next, per-stage gates
- [`docs/Model_Pipeline_Playbook.md`](docs/Model_Pipeline_Playbook.md) — rationale + model choices
- [`docs/DATASETS.md`](docs/DATASETS.md) — what to download, links, usage
- [`docs/DATASET_VALIDATION.md`](docs/DATASET_VALIDATION.md) — fact-check of dataset claims
- [`docs/COLAB_MAC_SPLIT.md`](docs/COLAB_MAC_SPLIT.md) — build↔deploy runbook
- [`docs/HOSTING_QUESTIONNAIRE.md`](docs/HOSTING_QUESTIONNAIRE.md) — hosting decisions, simple→advanced
- [`notebooks/README.md`](notebooks/README.md) — the .py-library / .ipynb-runner split

## Status
Runnable scaffold: metrics, calibration, edge benchmark, TinyU-Net, distillation loop, CoreML/ONNX
export, clDice gate, YOLO train + ByteTrack, serve loop, and prep converters (ARCADE/DCA1/Danilov/
CathAction) are implemented. Dataset-format edge cases and later stages (DSA sequences, SAM adapter,
AVF audio) are marked with TODOs. Not a medical device; research use, not for clinical care.
