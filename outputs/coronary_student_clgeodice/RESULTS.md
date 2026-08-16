# Coronary Stage-1 — nnU-Net teacher → TinyU-Net student, **CLGeoDice distillation**

**Run:** Kaggle GPU, kernel `jugalmodipoiro/coronary` (committed/headless), 2026-07-16
**Config:** `configs/coronary_seg.yaml` — `distill: {alpha: 0.5, temperature: 2.0, clgeodice_weight: 0.5, clgeodice_r_th: 8}`
**Pipeline:** nnU-Net v2 teacher (Dataset001_Coronary, ARCADE syntax + DCA1) → distill TinyU-Net student with KD + **CLGeoDice topology+geometry term on GT** → export ONNX + INT8.
**Artifacts (this dir, gitignored — local/release only):** `student.pt`, `student.onnx` (+`.data`), `student.int8.onnx`.

## Result — CLEARS the Stage-1 accuracy floor
| Metric | Value | Floor / target |
|---|---|---|
| **Dice** | **0.915** (best mid-run 0.927) | ≥ 0.75 ✅ |
| **clDice** | **0.956** (best mid-run 0.980) | within ~3% of teacher (topology gate) |
| KD loss (final) | 0.2228 | converged |
| Epochs | 200 / 200 | full run, no early stop |

Final epoch log: `epoch 200/200  kd_loss 0.2228  |  Dice 0.915 clDice 0.956`.
ONNX (Kaggle CPU) micro-bench: `student.onnx  size 0.01 MB  latency 114.91 ms  fps 8.7` — CPU path only; the real edge number is the CoreML student on Apple silicon (`make bench-coreml`).

## Why this matters
- **First coronary run with metrics on record.** The prior build (`outputs/coronary_student/`, 2026-07-12) produced `student.pt`+onnx but **Dice/clDice were never logged**, so PROJECT_TRACKER Stage 1 was "trained; gate UNVERIFIED." This run verifies it — and passes.
- **CLGeoDice (clgeodice_weight 0.5) wired into `distill()` this branch** (previously config-set but never passed → silently 0). The high clDice 0.956 is consistent with the topology+geometry loss doing its job on thin vessels — the connectivity axis `STAGE_ACCURACY_RESEARCH.md` (F4/F5) flags as the real lever (loss design, not SSL).

## CoreML export + compression clDice gate — DONE 2026-07-16, **PASSED** ✅
Exported `student.pt` → `student.mlpackage` (coremltools 9.0, **palettize 6-bit** — the repo's edge default; ~400K, palettized `weight.bin` 370,464 B), then ran `src/export/coreml_validate.py` on 50 held-out coronary val pairs (fetched from the Kaggle output; layout is `data/processed/coronary/{img,msk}/val_*.png`):

```
n=50
fp32    Dice 0.9156  clDice 0.9783
coreml  Dice 0.9138  clDice 0.9759
clDice drop +0.0023   gate (<= 0.03)  ->  PASS
```

**6-bit palettization does not break thin vessels** — clDice drop 0.0023, Dice drop 0.0018, both far under the 0.03 gate. The CoreML student is edge-trustworthy on the connectivity axis (`STAGE_ACCURACY_RESEARCH.md` F6's open question, now answered for *this* model + compression). Env note: use the **pyenv 3.12.9** interpreter (`~/.pyenv/versions/3.12.9/bin/python3`) — it has torch; the Homebrew `python3` (3.14) does not.

## ONNX static-PTQ INT8 clDice gate — DONE 2026-08-16, **PASSED** ✅

The 2026-08-13 static-PTQ landing measured **Dice only**. Dice can stay flat while quantization
thins or severs vessel centerlines, and `configs/edge_export.yaml` declares the HARD gate on
**clDice** (`gate.cldice_drop_max: 0.03`), so the gate was not actually evidenced until now.
Re-run on the same 50 held-out coronary val pairs, onnxruntime CPU EP, threshold 0.5, with the
identical preprocessing `quantize_int8.PngCalibrationReader` feeds (grayscale → resize 512 →
float32/255 → `[1,1,H,W]`):

```
n=50
fp32         Dice 0.9156  clDice 0.9783   (1.93 MB: student.onnx + student.onnx.data)
int8-static  Dice 0.9156  clDice 0.9786   (0.52 MB)
Dice drop    -0.0000
clDice drop  -0.0003   gate (<= 0.03)  ->  PASS
mask agreement 0.9996
```

**Static PTQ INT8 does not break thin vessels either.** The clDice difference is negative (INT8
marginally higher) — that is noise at n=50, not a gain; the honest read is "no measurable
connectivity cost" at a 73% size reduction. Both edge paths (CoreML 6-bit palettize, ONNX static
INT8) are now clDice-gated on evidence rather than on Dice-level inference.

## Still pending before full deploy sign-off
1. **clDice vs teacher within ~3%** — need the nnU-Net teacher's own clDice to confirm the teacher-relative bound formally. Raw student clDice 0.956–0.978 is high; compute the teacher number and the gap.
2. **On-device benchmark** on Apple silicon: `make bench-coreml MODEL=outputs/coronary_student_clgeodice/student.mlpackage` (latency/fps on the ANE, not the Kaggle-CPU 8.7 fps).
3. ~~`student.int8.onnx` gate (non-Apple targets only)~~ — **closed 2026-08-16** by the static-PTQ clDice gate above. Note it gates `student.int8.static.onnx`; the older dynamic `student.int8.onnx` is an unquantified fallback and should not ship.

## Provenance note
Retrieved via direct Kaggle output-file URLs (4 files, ~5 MB) — the kernel had saved its entire `/kaggle/working` (18,514 files, ~15k regenerable nnUNet cache PNGs), so a full `kaggle kernels output` pull was infeasible. **Fix for future coronary runs:** point nnU-Net caches at `/kaggle/tmp` (not `/kaggle/working`) so committed output stays small, mirroring the stenosis notebook.
