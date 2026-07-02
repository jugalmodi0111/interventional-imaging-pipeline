# Stage 3b — Cross-vendor validation (weeks 8-11)
Domain shift across Siemens/GE/Philips is a deliverable, not a caveat.
- Leave-one-vendor-out using `src/eval/cross_vendor.py`: ARCADE (Philips/Siemens), DCA1 (Mexico), XCAD (GE), Danilov (Siemens+GE).
- Train on a subset of vendors; test on the held-out vendor.
- **Exit:** report the held-out-vendor Dice/F1 gap; it must be within the agreed bound (else add the vendor or domain-adapt).
