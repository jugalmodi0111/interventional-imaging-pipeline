# Stage 1 — Coronary segmentation (weeks 1-4)
- Train nnU-Net teacher on ARCADE task 1; cache predictions.
- Distill a small student (TinyU-Net / MobileUNETR) toward the Dice >= 0.75 floor. SE-RegUNet (~0.72) does NOT qualify until pushed past the floor; CoroSAM (~0.78) qualifies.
- Export ONNX -> INT8; benchmark on the target device.
- **Exit gate:** Dice >= 0.75 AND clDice within ~3% of the teacher (Dice alone hides broken topology). Re-check clDice AFTER INT8.
- **Fallback if connectivity drops post-quant:** QAT -> larger student -> keep teacher as offline second-read.
