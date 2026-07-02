# Stage 3 — Temporal + catheter (weeks 5-10)
- Keyframe 2D + ConvLSTM-lite + MinIP on DIAS/DSCA. Realistic edge target Dice ~0.85, NOT 0.90.
- Keep DSANet (Dice 0.9033) as an OFFLINE second-read for DSA (like TAVR CT) — full temporal fusion is too heavy for edge.
- YOLO11n + ByteTrack on CathAction; optical-flow warp fallback for thin guidewires.
- **Exit:** DSA Dice ~0.85 (+ clDice); real-time tracking with fps + ID-switch tracked. (Cross-vendor check -> Stage 3b.)
