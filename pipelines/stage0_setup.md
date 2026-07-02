# Stage 0 — Setup & data prep (week 1)
- Create env, place datasets in `data/raw/`.
- `make prep-coronary` -> COCO + nnU-Net splits with CLAHE.
- Smoke-test `src/eval/metrics.py` and `src/eval/edge_benchmark.py` on your target device.
- **Exit:** one command reproduces a split + prints a latency report on the laptop.
