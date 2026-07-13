# Accuracy Research — 3-Stage XCA Pipeline (cited)

**Question:** highest-ROI, current (2023–2025) techniques to raise accuracy across coronary segmentation (S1), stenosis detection (S2), catheter/guidewire tracking (S3), under tight data + edge (CoreML) constraints.
**Method:** deep-research harness — 109 sub-agents, fan-out web search → source fetch → 3-vote adversarial verification (a claim needs 2/3 refutes to be killed). 21 raw claims → 8 semantically-deduped verified findings below.
**Date:** 2026-07-13. **Status:** evidence is deep for S1 + the leakage question; **thin/absent for S2, S3, calibration** (see Coverage Gaps).

---

## TL;DR — what the evidence supports

1. **In-domain SSL pretraining on unlabeled XCA is the single highest-leverage S1 lever** — but the confirmed lift is on **area Dice**, and the claim that it also fixes **thin-vessel topology/clDice was REFUTED**.
2. **Topology-aware losses**: `clDice` (proven) → 2024 successors **`cbDice`** (radius/diameter-balanced) and **`clCE`** (connectivity without accuracy loss).
3. **INT8 on clDice is UNVALIDATED** — quantization preserves mean Dice on big structures, never measured on thin-vessel connectivity → your post-INT8 clDice re-check is **essential, not optional**.
4. **Patient leakage confirmed at the source**: ARCADE ships frame-level splits, ≤12 frames/patient, **no patient IDs** → patient-grouped eval is mandatory (explains your 0.885→0.214 collapse).
5. Heavy foundation encoders (DINOv2 ViT-g/14) + temporal-fusion nets (TVS-Net) help accuracy but are **teacher/GPU-side only**, never the CoreML student.

---

## Per-stage ROI actions

### Stage 1 — Coronary segmentation  *(well-supported by literature)*
1. **SSL-pretrain the encoder on unlabeled XCA** (XCAD + cine) then fine-tune on DCA1/ARCADE. Domain-specific pretexts win: **DeepSA** Dice 0.828 on 40 labels; **CM-UNet** holds Dice 0.56 vs 0.38 scratch at 79:1 unlabeled:labeled; **VasoMIM** (anatomy-guided masking) beats MAE/SimMIM/DINO on ARCADE-V. ⚠️ magnitudes came from **170K/56K**-image corpora — your ~1621 XCAD may lift less. `[F1,F2]`
2. **Train with a topology loss, not just Dice:** `Dice + soft-clDice` (small α); prefer **cbDice** for thin-vessel diameter balance or **clCE** where accuracy must not regress (Dice+clCE beat Dice+clDice on coronary ASOCA 84.80 vs 83.42). `[F4,F5]`
3. **Keep the post-INT8 clDice gate** — literature does NOT de-risk INT8 for thin vessels. Consider QAT / FP16 / 6-bit palettize for the seg student. `[F6]`
4. **Temporal fusion (teacher-side)** where cine exists: TVS-Net 83.4% Dice / 84.3% recall on 173 labels. GPU-only (3D encoder). `[F7]`
5. ⚠️ **Don't expect SSL to fix connectivity** — the topology benefit of SSL at extreme scarcity was refuted; loss design (#2) is the connectivity lever, not SSL.

### Stage 2 — Stenosis detection  *(literature-thin → judgment-based)*
No verified sources found for stenosis-detection SOTA / recall tricks / pseudo-label SSL / open-vocab cold-start. The one hard datapoint is the **leakage/patient-diversity** result (below). Judgment-based plan (already implemented this branch): **more patients (CADICA) > frames**, recall-first gate + low-conf eval, temporal voting, triage/abstention, SSL backbone. Refresh this stage's SOTA before heavy investment. `[F8, Gap]`

### Stage 3 — Catheter/guidewire tracking  *(literature-thin → judgment-based)*
No verified sources for thin-structure attention heads / temporal consistency / ByteTrack alternatives. Plan stays judgment-based (AttWire-style multi-scale head, motion-model tracking). `[Gap]`

### Cross-cutting — calibration/abstention  *(literature-thin)*
No verified sources for temperature-scaling/OOD/coverage-risk in this domain — but it's the safety net for a below-floor S2 model. Keep the implemented triage path; treat thresholds as empirical. `[Gap]`

---

## Verified findings (confidence + sources)

**F1 · SSL pretraining on unlabeled XCA — highest S1 lever.** *(high)* CM-UNet: 18 labels → Dice drop only 15.2% vs 46.5% scratch; DeepSA (mask↔live cycle-adversarial pretext) → Dice **0.828 on 40 labels**. SSL wins most when unlabeled ≫ labeled (matches XCAD 1621 vs DCA1 134). Sources: [arxiv 2507.17779](https://arxiv.org/html/2507.17779v1), [Nature s41598-024-71063-5](https://www.nature.com/articles/s41598-024-71063-5), [PMC10998380](https://pmc.ncbi.nlm.nih.gov/articles/PMC10998380/)

**F2 · VasoMIM (anatomy-aware masked modeling) beats generic SSL.** *(high)* ARCADE-V DSC 80.25 vs MAE 79.39 / SimMIM 77.81 / DINO 76.37 / scratch 71.44; 10% labels → 76.01 DSC (> fully-supervised TransUNet @100%). ⚠️ single-group, thin margin over MAE. Source: [arxiv 2602.11536](https://arxiv.org/html/2602.11536) (conf. 2508.10794)

**F3 · Frozen DINOv2 + light decoder beats from-scratch (teacher-side only).** *(medium)* ViT-g/14 @448 + U-Net decoder Dice 0.642 vs TransUNet-scratch 0.535 on AMOS. ⚠️ CT not XCA; baselines are from-scratch not nnU-Net SOTA; ViT-g is heavy → teacher-side. "DINOv2 wins few-shot" and "Radio-DINO beats ImageNet SSL" were **REFUTED**. Source: [arxiv 2312.02366](https://arxiv.org/html/2312.02366v4)

**F4 · clDice/soft-clDice — canonical topology loss for thin vessels.** *(high)* Skeleton-intersection metric; soft-clDice is differentiable, proven topology-preserving; better connectivity/graph-similarity than soft-Dice/BCE. Use combined with Dice/BCE at small α. Source: [arxiv 2003.07311](https://arxiv.org/abs/2003.07311) (CVPR 2021)

**F5 · 2024 successors cbDice / clCE fix clDice's weaknesses.** *(high)* clDice+Dice → diameter imbalance (favors big vessels). **cbDice** (MICCAI'24, code) adds radius/boundary awareness → uniform across diameters. **clCE** (MICCAI'24, code) improves DSC & cl-DSC; Dice+clCE > Dice+clDice on ASOCA (84.80 vs 83.42). ⚠️ validated on retinal/CTA, not XCA/INT8. Sources: [arxiv 2407.01517](https://arxiv.org/html/2407.01517v1), [MICCAI 2024 #1081](https://papers.miccai.org/miccai-2024/770-Paper1081.html)

**F6 · INT8 preserves mean Dice but is UNVALIDATED on connectivity/clDice.** *(high)* TensorRT INT8 across 7 3D-seg models kept mDSC ~flat (nnU-Net 0.901→0.895) at 2.4–3.9× smaller / 2–2.7× faster — **but only mean Dice on large structures was measured; no clDice/centerline/thin-tubular metric**. Source: [arxiv 2501.17343](https://arxiv.org/pdf/2501.17343)

**F7 · Temporal fusion raises XCA Dice+recall (teacher-side).** *(medium)* TVS-Net 83.4% Dice / 84.3% recall on 173 coarse labels; cross-hospital 78.5/82.4 (~5pt drop) still beats single-frame. ⚠️ private dataset; 3D encoder = GPU-only. Source: [Medical Image Analysis 2025](https://www.sciencedirect.com/science/article/pii/S1361841525000441)

**F8 · ARCADE frame-level split, no patient IDs → leakage mandatory to fix.** *(medium)* 3000 frames, ≤12 frames/patient across 6 angles, no patient IDs → same-patient frames span train/test unless re-split by patient. Directly corroborates your 0.885 (leaked) vs 0.214 (patient-grouped). Source: [Nature s41597-023-02871-z](https://www.nature.com/articles/s41597-023-02871-z)

---

## Refuted / tempered (honesty)

- **REFUTED:** SSL improves thin-vessel **topology/clDice** at extreme scarcity (only area-Dice confirmed).
- **REFUTED:** DeepSA 0.755 on public XCAD "meets the ≥0.75 target" — do not cite as clearing the bar (no clDice measured).
- **REFUTED:** DINOv2 wins the 8-patient few-shot regime; domain-specific Radio-DINO beats ImageNet SSL.
- **Magnitude risk:** top SSL results used 170K (VasoMIM) / 56K (DeepSA) unlabeled images vs your ~1621 — label-efficiency may not reproduce at your scale.
- **Transfer risk:** cited studies use different datasets than ARCADE/DCA1/XCAD; cross-dataset transfer unverified.

## Coverage gaps (research found little/nothing)
- **S2 stenosis SOTA** — ARCADE-2023 winners, recall/small-object tricks, pseudo-label SSL, GD cold-start.
- **S3 catheter/guidewire** — thin-structure heads, temporal consistency, ByteTrack alternatives.
- **Calibration/abstention** — temperature scaling, OOD, coverage-risk in this domain.
- **Distillation loss** — how much clDice is lost nnU-Net→TinyU-Net, and whether topology-aware distillation targets recover it.
- Whether **any** SSL/topology gain survives **INT8 for thin vessels** at your data scale.

*These stay judgment-based; refresh before implementation — detection/tracking SOTA moves fast.*
