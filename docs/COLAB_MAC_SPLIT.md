# Stage 1 runbook — Colab build → Mac CoreML (coronary)

Locked answers: **build on Colab (GPU), deploy on Apple-silicon Mac, scope = coronary → stenosis → catheter.**
This splits the system into a build side and an inference side. CoreML can be *converted* on Linux but only
*run/benchmarked* on macOS — so conversion + the clDice gate + the on-device benchmark live on the Mac.

```
        COLAB (GPU build)                         MAC (Apple-silicon deploy)
  download data ─┐
  nnU-Net teacher │                          student.pt ─┐
  cache logits    ├─► distill TinyU-Net ─►   (Drive)      ├─► CoreML .mlpackage
  (all on Drive)  │   → student.pt + .onnx               │   palettize 6-bit
                 ─┘            │                          ├─► clDice GATE (≤0.03 drop)
                              Drive ◄────── pull ─────────┤   bench on Neural Engine
                                                         ─┘   demo
```

## Build env vs export env (two machines, two envs)
- **Colab build:** `torch`-CUDA + `nnunetv2` + `ultralytics` + `onnx`/`onnxruntime`. No coremltools.
- **Mac export:** `coremltools` + `onnx` + `opencv-python` + `scikit-image`/`scipy` (for clDice). No CUDA.

## Colab — run `notebooks/colab_coronary_build.ipynb`
1. Mount Drive; set `DRIVE` project root.
2. Point `nnUNet_raw / nnUNet_preprocessed / nnUNet_results`, `runs/`, teacher cache → **Drive** (survives disconnect).
3. Place ARCADE (Zenodo 8386059/10390295) + DCA1 (CIMAT) under `data/raw/`; `make prep-coronary`.
4. nnU-Net teacher (2D ResEncM). T4 16 GB clears it; cut epochs (~250) for v1; `--c` resumes across sessions.
5. `nnUNetv2_predict --save_probabilities` → cache teacher logits.
6. `distill(...)` → **`student.pt` (state_dict)** + `student.onnx` on Drive.

## Mac — pull `student.pt`, then
```bash
make export-coreml   MODEL=runs/coronary/student.pt
make validate-coreml CORE=runs/coronary/student.mlpackage WEIGHTS=runs/coronary/student.pt \
                     IMAGES=data/processed/coronary/val/img MASKS=data/processed/coronary/val/msk
make bench-coreml    MODEL=runs/coronary/student.mlpackage
```
- **export-coreml** — rebuild TinyU-Net from state_dict → trace → CoreML (mlprogram) → 6-bit palettize.
- **validate-coreml** — HARD gate: clDice drop fp32→CoreML must be **≤ 0.03**. Palettization breaks thin
  vessels even when Dice holds. If it fails: raise nbits (6→8), try `--method linear`, or distill a wider student.
- **bench-coreml** — latency/fps/size with `compute_units=all` (CPU+GPU+ANE). Lab fps ≠ cart fps.

## Why state_dict, not the pickled model
`distill()` now saves a **state_dict**; `to_onnx`/`to_coreml` rebuild `TinyUNet` via `load_student()`. A pickled
whole-module needs the class importable on the Mac and is brittle across torch versions — state_dict is portable.

## Apple quantization ≠ ONNX INT8
On Apple silicon you don't do ONNX static INT8 PTQ — you use `coremltools.optimize` weight compression
(palettization / linear). `configs/edge_export.yaml` now carries both paths; the `calib_images: 200` idea maps
to coremltools **activation** quant if you later need it (weight palettization is data-free).

## v1 sequencing (two stacks, not three)
1. **Coronary (seg stack)** — proves the manual ONNX/state_dict → CoreML path. Floor Dice ≥ 0.75 (distill SE-RegUNet/TinyU-Net past it).
2. **Stenosis (YOLO stack)** — `YOLO11n`; Ultralytics has first-class CoreML export (`model.export(format='coreml', nms=True)`), so export ≈ one call. Floor F1 > 0.57.
3. **Catheter (YOLO + tracking)** — reuses stenosis export; **ByteTrack runs as Python in the app loop, not inside the CoreML graph** (CoreML does per-frame detection, tracking sits on top). Do last — CathAction (~500k frames) strains Colab/Drive storage; selectively download or stream.

## Audit trail — wire it from the first demo
`src/eval/audit.py::record(model_version, input_arr, summary)` logs (ts, model, input sha, summary) to
`runs/audit.jsonl`. Nearly free now, expensive to retrofit for incident review / regulatory traceability.
