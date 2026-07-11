# Stenosis run — `arcade_yolo11n_640_e150`

**Date:** 2026-07-11 · **Platform:** Kaggle T4 · **git:** 7dd41661 · ultralytics 8.4.92 / torch 2.10.0+cu128
**Run name (auto):** `run_tag(cfg)` → `arcade_yolo11n_640_e150`

## Config
| | |
|---|---|
| Data | ARCADE task-2 stenosis **only** (841 train / 159 val imgs; 1,358 / 267 boxes) |
| Model | YOLO11n (nano), imgsz 640, pretrained |
| Epochs | 150 (real run, `DRY_RUN=False`) |
| SSL | off (pseudo_label disabled on this pass) |

## Result — BELOW FLOOR
| Metric | Value | Floor |
|---|---|---|
| **F1** | **0.246** | **0.57** ❌ |
| Precision | 0.242 | |
| Recall | 0.251 | |
| mAP50 | 0.147 | |
| mAP50-95 | 0.054 | |

## Verdict: learning, not broken
- Labels verified clean (`val_batch0_labels.jpg`) — CLAHE'd angiograms, stenosis boxes on real vessel segments.
- Predictions (`val_batch0_pred.jpg`) fire low-conf boxes (0.3–0.5) **on vessels**, several on true lesions (117/142/16/148/162) — but **miss many** frames entirely → low P and R.
- val loss > train loss → overfitting on the small set.
- Conclusion: weakest-config baseline. Fix = **more data + bigger model + SSL**, not a debug.

## Next run (recipe to cross the floor)
Cumulative levers, biggest first:
1. **+Danilov** (~9,800 imgs total) — Mendeley `ydrm75xywg` (v3 = 8.3 GB zip; URL-import into Kaggle).
2. **YOLO11n → YOLO11s** (playbook §2.2 sanctioned).
3. **imgsz 640 → 768** (tiny lesions need resolution).
4. **pseudo-label SSL** on unlabeled frames.

Expected next run name → `arcade+danilov_yolo11s_768_e150`.

## Files
`run/` — full ultralytics output: `results.csv`, curves (BoxF1/PR/P/R), confusion matrices, val/train batch previews, `args.yaml`, `weights/{best,last}.pt`.
