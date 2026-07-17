# Stenosis Stage-2 — ARCADE + CADICA + Danilov, YOLO11s @ 768 (HONEST patient-grouped split)

**Run tag:** `arcade+cadica+danilov_yolo11s_768_e150`
**Date:** 2026-07-16 (Kaggle T4, kernel `jugalmodipesurr/stenosis`, committed/headless)
**Config:** `configs/stenosis_yolo.yaml` — yolo11s, imgsz 768, batch 16, lr0 1e-3, AMP, RAM cache, patience 30, epochs 150
**Data:** ARCADE task-2 stenosis + **CADICA (patient-diverse add, 3996 keyframes)** + Danilov (capped 5/patient → 320 frames) → **patient-grouped** split (`io_utils.split_of` via `group_key`)

## Split (leakage-free — verified in the run log)
```
LEAKAGE CHECK PASSED — split is patient-grouped and honest
  train 3909 imgs / 3685 groups | val 1907 imgs / 1875 groups (val ~34% by group)
  cadica: 3996 frames | danilov: 320 frames (64 patients, capped 5/patient)
```
Every patient's frames are confined to ONE split — no frame of a val patient appears in train.

## Result — BELOW FLOOR (but a real lift from CADICA)
| Metric | Value | Floor |
|---|---|---|
| **F1** | **0.291** | **0.57** ❌ |
| Precision | 0.314 | |
| Recall | 0.271 | |
| mAP50 | 0.209 | |
| mAP50-95 | 0.080 | |

`best.pt | scores: {'precision': 0.3137, 'recall': 0.2714, 'f1': 0.291, 'map50': 0.2086, 'map': 0.0804}`
Ran **69/150 epochs** — early-stopped by `patience=30`.

## CADICA moved the needle — patients > frames, confirmed
| Run | Split | F1 | Recall | mAP50 |
|---|---|---|---|---|
| arcade-only 11n/640 | — | 0.246 | — | 0.147 |
| arcade+danilov 11s/768 | per-frame (LEAKY) | 0.885 | — | 0.870 |
| arcade+danilov 11s/768 | **patient-grouped (HONEST)** | 0.214 | 0.166 | 0.108 |
| **arcade+cadica+danilov 11s/768** | **patient-grouped (HONEST)** | **0.291** | **0.271** | **0.209** |

Adding CADICA's ~42 lesion patients: **F1 +0.077 (0.214→0.291), recall +0.105 (0.166→0.271, ~+63% relative), mAP50 +0.101**. This is the largest honest single-lever gain to date and directly confirms the read-out from the `_grouped` run: **patient diversity is the bottleneck, not epochs/model.**

## Read-out
- Still **below the 0.57 floor** — CADICA closed roughly a third of the 0.214→0.57 gap but not all of it.
- **Recall 0.27 is the axis that matters** — a missed stenosis is the costly error; the model still misses ~73%. The orchestrator (Stage 6) therefore keeps this model `floor_ok: false` → its finding surfaces as a **deferred screening flag**, never a confident positive.
- **Groups ≈ images (3685/3909)**: CADICA keyframes are near-unique per patient-video, so grouping barely collapses them and the val fraction ballooned to ~34%. That means less train data than the raw count suggests — a bigger CADICA-heavy corpus (or lower val fraction) is itself a lever.

## Next levers (in rough ROI order)
1. **More patient diversity still** — CADICA helped exactly as predicted; pull additional stenosis-labelled patients (open-vocab GD cold-start on unlabeled cine, more public sets).
2. **Pseudo-label SSL** on a disjoint unlabeled dir (XCAD) — raise recall specifically. Auto-disabled here (no disjoint `ssl.unlabeled_dir` attached) so it couldn't re-leak val.
3. **Bigger student (11m) / RT-DETR-R18** — small gains expected until (1), per `STAGE_ACCURACY_RESEARCH.md` (S2 is data-limited).
4. Re-balance the split (target val ~15–20% by group) so CADICA frames feed train instead of inflating val.
