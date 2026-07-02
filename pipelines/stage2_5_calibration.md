# Stage 2.5 — Calibration & abstention (weeks 5-7)
For a real-time tool, "wrong but confident" is the failure that hurts. Measure and contain it.
- ECE + reliability diagram + Brier (`src/eval/calibration.py`); temperature-scale post-hoc.
- Coverage-risk curve -> defer-to-human threshold.
- OOD detector to flag unfamiliar vendor/view/artifact; exploit CoronaryDominance quality/uncertainty tags.
- **Exit:** ECE < ~0.05; the defer path demonstrably fires on OOD inputs.
