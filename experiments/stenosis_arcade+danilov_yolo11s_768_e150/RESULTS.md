# Stenosis Stage-2 — ARCADE + Danilov, YOLO11s @ 768

**Run tag:** `arcade+danilov_yolo11s_768_e150`
**Date:** 2026-07-12 (Kaggle T4, committed run `notebook60c9605cf3` / script version 334396784)
**Config:** `configs/stenosis_yolo.yaml` — yolo11s, imgsz 768, batch 16, lr0 1e-3, AMP, RAM cache, patience 30
**Data:** ARCADE task-2 stenosis (1500) + Danilov (8325 VOC/bmp) → 7861 train / 1464 val images

## Headline metrics (per-frame split — see caveat)

| | epoch | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **best F1** | 80 | 0.963 | 0.818 | **0.885** | 0.868 | 0.411 |
| best mAP50 | 77 | 0.961 | 0.816 | 0.882 | **0.871** | 0.415 |
| best.pt (fitness) | 71 | 0.954 | 0.810 | 0.876 | 0.866 | 0.417 |
| last | 101 | 0.947 | 0.821 | 0.879 | 0.855 | 0.407 |

- Ran **101/150 epochs** in 6.4 h — early-stopped by `patience=30` (best fitness at epoch 71, no improvement through 101). Well under the 12 h Kaggle wall; metrics had plateaued ~ep50.
- **vs baseline** (arcade-only 11n/640): F1 0.246 → 0.885, mAP50 0.147 → 0.87.
- Gate `F1 ≥ 0.57`: **PASS on paper.** Recall peaks 0.829 → still misses ~17 % of stenoses (recall is the clinically costly axis).

## ⚠️ Caveat — metrics are inflated by video-frame leakage

Danilov ships **8325 frames from ~100 patients** (`<site>_<patient>_<seq>_<frame>`, e.g. `14_002_5_0016`). Consecutive frames are near-identical. The split at run time was **per-frame** (`split_of` hashed the full filename), so a lesion's frame 0016 could train while 0017 validated — **every patient appeared in both splits**. mAP50 0.87 vs ARCADE's ~0.57 SOTA is the signature. Real generalization is lower by an unknown margin.

**This run's number is NOT a trustworthy Stage-2 result.** It proves the pipeline (data merge, yolo11s/768, convergence) works, not that the model generalizes to unseen patients.

## Observations (training dynamics + visual audit)

**Training curves (`results.png`):** all three losses (box/cls/dfl) fall monotonically; val cls-loss flattens ~0.90 late with no upward divergence → no gross overfitting *on the leaky split*. Metrics plateau by ~epoch 50; the 51→101 tail adds almost nothing (12 h cap cost little).

**Metric shape:**
- **Precision ≫ recall (0.96 vs 0.83).** Model rarely false-positives but misses lesions — the *clinically costly* direction (a missed stenosis > a false alarm).
- **mAP50 0.87 but mAP50-95 only 0.41.** Boxes are roughly right at IoU 0.5 but localize loosely at strict IoU — expected for tiny stenosis boxes where a few px shifts tank IoU.

**Visual audit (`val_batch0_pred.jpg` vs `_labels.jpg`, 16 tiles):**
- GT has boxes on ~15/16 tiles (several with 2–4 lesions). **Prediction at ≥0.25 conf fires on only ~7/16, confidences 0.3–0.6**, and misses the multi-lesion frames (800.png, 758.png: 3–4 GT boxes → 0–2 predicted).
- So at a *deployable* confidence the detector visibly **under-detects**. The reported recall 0.83 reflects a low-confidence operating point + near-duplicate-frame leakage, not usable-threshold behaviour.
- Where it does fire (822, 870, 875, 756, 772) the boxes sit correctly on stenotic segments — the model has learned the right feature, it's just under-confident / under-covering.

**Takeaway:** even setting leakage aside, this checkpoint is recall-limited and low-confidence — not deployment-ready. Demo: `outputs/output_stenosis/stenosis_demo.mp4` (GT-vs-pred side-by-sides + curves).

## Fix applied

`io_utils.split_of` now hashes a **patient group key** (`group_key`): Danilov frames group by `<site>_<patient>` so all frames of a patient share a split; ARCADE (numeric names) unchanged. Verified: 0 patients span both splits; 47 tests pass.

## Next

1. **Re-run** `kaggle_stenosis_plug_and_play.ipynb` with the patient-grouped split → this is the honest Stage-2 F1. Expect a drop from 0.885; judge against the 0.57 floor on that number.
2. If short after the honest split: SSL pseudo-label round, `ssl.seed=gdino` cold-start, or yolo11m.
3. Only after an honest pass: export `best.pt` → CoreML, edge bench, Stage 2.5 calibration.

## Artifacts
- `results.csv` — full 101-epoch curve (this folder).
- `best.pt` — epoch-71 weights, fitness-selected (gitignored; on Kaggle output + local `outputs/output_stenosis/best.pt`). **Trained on the leaky split — retrain after the fix before shipping.**
