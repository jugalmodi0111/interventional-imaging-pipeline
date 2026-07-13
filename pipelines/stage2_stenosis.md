# Stage 2 — Stenosis detection (weeks 3-6)

- **YOLO11s @ 768** on ARCADE task-2 + **Danilov (capped 5 frames/patient)** + **CADICA**. StenUNet / U-Mamba BOT stay as offline accuracy teachers.
- **Prep:** `make prep-stenosis` runs `danilov_to_yolo` (ARCADE + Danilov) then `cadica_to_yolo` (patient-diverse add; skips if `datasets.cadica.root` absent). Split is **patient-grouped** — a leakage audit hard-fails on any train/val patient overlap.
- **Recall-first gate:** `qualifies_det` enforces the F1 floor and, when `target.recall` is set, a recall floor too — a missed stenosis is the deadly error. Eval runs at `val.conf: 0.001` so recall isn't throttled.
- **Patients > frames:** the honest patient-grouped F1 is 0.214 (ARCADE+Danilov) vs 0.246 ARCADE-only vs the 0.57 floor — the bottleneck is patient count, so CADICA (and more patients) is the #1 lever, not more frames.
- **Inference safety:** temporal voting over a cine window (`src/serve/temporal_vote.py`) recovers missed frames + drops one-frame flicker; triage/abstention (`src/serve/stenosis_triage.py`) defers OOD / low-confidence cases to a human while the model is sub-floor.
- **Exit:** F1 ≥ 0.57 on ARCADE stenosis (recall-weighted); COCO AP tracked on Danilov.
- **Setup + run guide:** [`docs/STAGE2_SETUP.md`](../docs/STAGE2_SETUP.md) (datasets, Kaggle attach names, config knobs, ROI-ranked levers).
