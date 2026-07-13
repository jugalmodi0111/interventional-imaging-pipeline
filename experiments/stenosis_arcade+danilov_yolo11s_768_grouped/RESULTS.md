# Stenosis Stage-2 — ARCADE + Danilov, YOLO11s @ 768 (HONEST patient-grouped split)

**Run tag:** `arcade+danilov_yolo11s_768_grouped`
**Date:** 2026-07-13 (Kaggle T4, kernel `jugalmodi0111/stenosis`)
**Config:** `configs/stenosis_yolo.yaml` — yolo11s, imgsz 768, batch 16, lr0 1e-3, AMP, RAM cache, patience 30, epochs 150
**Data:** ARCADE task-2 stenosis + Danilov → **patient-grouped** split (the leakage fix, `io_utils.split_of` via `group_key`)

## Split (leakage-free — verified in the run log)
```
LEAKAGE CHECK PASSED — split is patient-grouped and honest
  train 8766 imgs / 1349 groups | val 1059 imgs / 215 groups (val ~14% by group)
  danilov: 8325 frames -> 64 patients, 0 ungrouped (0%)
```
Every Danilov patient's frames are confined to ONE split — no frame of a val patient appears in train.

## Result — BELOW FLOOR
| Metric | Value | Floor |
|---|---|---|
| **F1** | **0.214** (best@ep38 0.215) | **0.57** ❌ |
| Precision | 0.299 | |
| Recall | 0.166 | |
| mAP50 | 0.108 | |
| mAP50-95 | 0.034 | |

Final val: `all  1059 imgs  1193 instances  P 0.299  R 0.166  mAP50 0.108  mAP50-95 0.034`
`best.pt | F1: 0.2136 | mAP50: 0.1082`. Ran **68/150 epochs** — early-stopped by `patience=30` (best fitness ~ep38).

## The leakage was the whole story
| Run | Split | F1 | mAP50 |
|---|---|---|---|
| arcade-only 11n/640 | — | 0.246 | 0.147 |
| arcade+danilov 11s/768 | **per-frame (LEAKY)** | **0.885** | 0.87 |
| arcade+danilov 11s/768 | **patient-grouped (HONEST)** | **0.214** | 0.108 |

The 0.885 was almost entirely frame-leakage: neighbouring frames of the same patient sat in both train and val. On the honest split the model collapses to ~arcade-only level — **Danilov's 8325 frames add essentially nothing in generalization** because they are only 64 patients (heavy per-patient redundancy).

## Read-out
- Stage-2 stenosis is **not near the floor** on honest data. The scarce *patient* diversity (ARCADE + 64 Danilov patients) is the bottleneck, not epochs or model size.
- **Recall 0.17 is the alarming axis** — a missed stenosis is the costly error; this model misses ~83%.

## Next levers (in rough ROI order)
1. **More patient diversity** — pull additional stenosis sources; 64 patients is too few regardless of frame count.
2. **Pseudo-label SSL** on unlabeled frames (now preprocessing-consistent after the CLAHE fix) to lift recall.
3. **GD cold-start** (`ssl.seed: gdino`) open-vocab seed before self-training.
4. Bigger student (11m) / RT-DETR-R18 fallback — but data is the limiter, so expect small gains until (1).
