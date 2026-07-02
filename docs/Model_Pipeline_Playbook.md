# Model & Pipeline Playbook — Interventional Angiography, Fluoroscopy & Nephrology

**Target deployment: edge / laptop.** Every production recommendation below is something you can export and run on a laptop-class device (or a procedure-cart mini-PC). Heavy models appear only as *teachers* (trained on a GPU, then distilled) or as *offline* workstation steps where edge deployment is impossible (3D CT). Companion files: `Model_Selection_Matrix.xlsx` (scored, with the accuracy-floor gate), `Angiography_Dataset_Validation_Scoring.xlsx` (data), and the `interventional-imaging-pipeline/` repo scaffold.

> **Rev 2 — clinical-safety hardening.** This version treats accuracy as a *floor*, not a tradeable term; adds calibration + abstention to the eval harness; carries clDice into the exit gates; and names cross-vendor validation and a regulatory gate as explicit stages. These changes matter because the deployment target is a live procedure, where the dangerous failure is "wrong but confident."

---

## 0. The governing principle: accuracy is a floor, not a trade

For procedural guidance you do **not** let a fast, light model compensate for missing a stenosis or under-segmenting a vessel. Selection is therefore **two-step**:

1. **Qualify** — a model must clear its problem's *clinical accuracy floor* (e.g. Dice ≥ 0.75 for coronary vessels, F1 ≥ 0.55 for stenosis). Floors live on the `Accuracy Floors` sheet and are editable assumptions to be set with clinical stakeholders.
2. **Rank** — only among qualifiers do you optimize the **edge-suitability composite** (footprint, speed, edge-readiness, etc.).

The composite in the matrix (column J) is an **edge-suitability index, not an overall quality score**: a teacher at 3.1 can be more accurate than a student at 4.4. Always read it together with the Role and Qualifies columns. Applying the gate already changes two recommendations: SE-RegUNet (Dice ~0.72) drops below the coronary floor, and YOLO11n (F1 ~0.54) drops below the stenosis floor — so the qualifying picks become a distilled student/CoroSAM and YOLOv8s+SSL respectively.

---

## 1. The core design pattern: teacher → distill → quantize → deploy

Because the target is edge, the same pattern repeats for every perception problem:

1. **Train a strong teacher on a GPU** (nnU-Net, U-Mamba, DSANet, StenUNet). This sets the accuracy ceiling and, for segmentation, doubles as a label-quality oracle.
2. **Distill into a small student** (TinyU-Net / MobileUNETR / YOLO-nano). Knowledge distillation recovers most of the teacher's accuracy at a fraction of the parameters — but it is **data-bound** (see §3.4).
3. **Export and quantize** — ONNX → INT8 (ONNX Runtime / OpenVINO on Intel laptops), CoreML on Apple silicon, or TensorRT on a Jetson / RTX laptop. Benchmark latency, fps, model size and peak RAM on the *actual* target device — and re-check connectivity (clDice), because INT8 tends to break thin vessels.

A foundation-model track runs in parallel: **CoroSAM / MedficientSAM / Rep-MedSAM** for prompt-based, parameter-efficient (LoRA/adapter) segmentation — most useful for semi-automatic *labeling* of institutional data and human-in-the-loop correction.

---

## 2. Problem-by-problem recommendations (gated picks)

### 2.1 Coronary vessel segmentation — floor Dice ≥ 0.75

**Data:** ARCADE (task 1, 25 SYNTAX regions), DCA1 (binary masks), XCAD (126 labeled + 1,621 unlabeled).

- **Teacher:** nnU-Net v2 (ResEnc-M).
- **Qualifying edge pick:** a **distilled TinyU-Net / MobileUNETR** pushed to ≥ 0.75, or **CoroSAM** (~0.78). Note: SE-RegUNet is fast (41.6 fps) but at ~0.72 it sits *below* the floor — use it as a distillation target and push it past 0.75 before it qualifies.
- **Labeling / interactive:** CoroSAM and MedficientSAM (CVPR'24 MedSAM-on-Laptop winner) for prompted masks on your own cine.
- **Pretraining:** self-supervised on XCAD's 1,621 unlabeled frames + institutional cine (Dice ~0.83 with as few as 40 labels has been reported).
- **Preprocessing:** CLAHE + unsharp.
- **Metrics:** Dice ≥ 0.75 **and clDice** (connectivity) + HD95.

### 2.2 Stenosis / lesion detection — floor F1 ≥ 0.55 (recall-weighted)

**Data:** ARCADE (task 2), Danilov (8,325 coronary images, COCO boxes).

- **Qualifying edge pick:** **YOLOv8s + pseudo-label SSL** (~0.56). Plain YOLO11n (~0.54) is below the floor — step up to 's', add SSL/distillation, or fall back to RT-DETR-R18.
- **Accuracy teachers:** U-Mamba BOT (F1 0.6879 — the bar to chase) and StenUNet (F1 0.5348); distill or run as an offline second-read.
- **Metrics:** F1 ≥ 0.55 with a **recall-weighted** operating point (missing a stenosis is the costly error); COCO AP/AR on Danilov.

### 2.3 Cerebral DSA (temporal) + catheter/guidewire tracking

**Data:** DIAS, DSCA, CathAction.

- **DSA floor Dice ≥ 0.80.** Qualifying edge pick: **2D keyframe segmenter + ConvLSTM-lite + MinIP**, realistic target **~0.85** — *not* 0.90. DSANet's 0.9033 comes precisely from full temporal fusion, so keep **DSANet as an offline second-read** for DSA (the same way TAVR CT is handled offline). Set the target to what the edge model can actually reach, and accept the gap explicitly.
- **Catheter/guidewire (floor IoU ≥ 0.50, but fps + ID-switches matter more):** YOLO11n + ByteTrack, or optical-flow-guided mask warping for thin guidewires. CathAction labels catheter and guidewire separately. Reimplement AttWire's multi-scale Gaussian-derivative attention head if thin-wire recall is short.

### 2.4 Interventional nephrology / AV fistula (data desert — your priority)

No public AV-fistula imaging benchmark, so this is transfer-learning + primary collection. Three edge-friendly tracks:

- **Audio (bruit) — ship first, floor Sensitivity ≥ 0.85:** small ViT on blood-flow spectrograms (Zhou et al., *npj Digital Medicine* 2023) or CNN-BiLSTM (Ota 2020). Tiny enough for a wearable/phone. **Framed as screening/triage, not confirmation** — hold that line in any clinical messaging (reported specificity ~0.79–0.92).
- **Surveillance (tabular) — best ROI, floor AUROC ≥ 0.80:** XGBoost / LightGBM with SHAP. Edge-trivial, interpretable; add calibration (ECE) since it triages patients.
- **Imaging (ultrasound/fistulography), floor Dice ≥ 0.75:** lightweight U-Net initialized from coronary/peripheral vessel weights, fine-tuned on institutional ultrasound. **Requires data collection + IRB** — and note that student quality is gated on collecting that data (§3.4).

### 2.5 TAVR / structural heart

- **Pre-procedural CT sizing (offline, GPU workstation — NOT edge), floor ICC ≥ 0.95:** 3D nnU-Net or SwinUNETR for aortic root / annulus / sinus / coronary-ostia, trained on **MM-WHS + Seg.A.** then domain-adapted to institutional TAVR CT. Reuse TAVI-PREP's measurement extraction (MeshDeformNet + 3D residual U-Net, 22 measurements). This is the *pick* for its problem precisely because no edge option exists.
- **Intra-procedural fluoroscopy (edge), floor detection ≥ 0.85:** YOLO11n / AttWire-style detector for valve and catheter tracking on the cart.
- **Outcome / risk (edge, tabular), floor C-stat ≥ 0.75:** XGBoost / random forest (TAVI Risk Machine style), with calibration.

---

## 3. Cross-cutting engineering & safety

### 3.1 Calibration & abstention (mandatory — new)

Dice/AP say nothing about *confidence*. For a real-time tool, "wrong but confident" is the dangerous mode, and it is currently easy to leave unmeasured. Add:

- **Calibration:** Expected Calibration Error (ECE), reliability diagrams, Brier score. Temperature-scale post-hoc; target ECE < ~0.05 before deployment.
- **Abstention / OOD:** a defer-to-human path driven by a coverage–risk curve and an OOD detector (flag inputs unlike training — unfamiliar vendor, view, artifact). CoronaryDominance ships poor-quality/artifact/uncertainty tags you can use to train exactly this gate.

### 3.2 clDice in the exit gates + connectivity fallback (new)

The Stage-1 exit gate is **not** Dice alone. Require **both** Dice ≥ floor **and** clDice within ~3% of the teacher. If the distilled+quantized student loses connectivity, the fallback ladder is: quantization-aware training (QAT) → a larger student → keep the teacher as an offline second-read. Re-measure clDice *after* INT8, not before.

### 3.3 Cross-vendor validation as a named stage (new)

Domain shift across Siemens / GE / Philips is operationalized, not just noted. You have a natural **leave-one-vendor-out** setup — ARCADE (Philips/Siemens), DCA1 (Mexico/IMSS), XCAD (GE), Danilov (Siemens+GE). Train on a subset of vendors, report the held-out-vendor Dice/F1 gap as a deliverable in Stage 3b.

### 3.4 Distillation is data-bound (new)

Distillation and SSL both transfer best with lots of (even unlabeled) frames, and public coronary data is small (ARCADE ~3k, DCA1 134, XCAD 126+1,621). Institutional cine collection therefore feeds *both* the teacher's labels and the distillation set — so **student quality is gated on collecting that cine.** State this dependency in the plan rather than discovering it late.

### 3.5 Standards & init

Standardize annotations on COCO JSON (detection/instance — ARCADE, Danilov, CADICA) and nnU-Net NIfTI/PNG (semantic — DCA1, XCAD, DIAS). Init encoders from RAD-DINO (882k CXRs, DINOv2) or BiomedCLIP; self-supervise on unlabeled angiograms for the grayscale gap.

### 3.6 Edge metrics (on the real device)

Report params (M), FLOPs (G), latency (ms), fps, peak RAM (MB), model size (MB) — measured on the target, INT8. Lab fps ≠ cart fps.

### 3.7 Optimization toolkit & access logistics

PyTorch + MONAI + nnU-Netv2 + Ultralytics; HuggingFace PEFT (LoRA/adapters) for SAM-family on small data; ONNX Runtime / OpenVINO / TensorRT / CoreML + INT8 PTQ (QAT fallback) + pruning + distillation. Apply early for PhysioNet credentialed access (CITI + DUA) for MIMIC-CXR and VinDr-CXR; register for CheXpert.

---

## 4. Staged roadmap

| Stage | Weeks | Focus | Exit gate |
|---|---|---|---|
| 0 | 1 | Repo + data prep: standardize ARCADE/DCA1/Danilov; CLAHE; edge-benchmark harness | One command reproduces a split + a latency report on the target device |
| 1 | 1–4 | Coronary segmentation: nnU-Net teacher → distill → ONNX-INT8 | **Dice ≥ 0.75 AND clDice within ~3% of teacher**; real-time on target |
| 2 | 3–6 | Stenosis detection: YOLOv8s + pseudo-label SSL; U-Mamba/StenUNet teachers | **F1 ≥ 0.55, recall-weighted**; COCO AP tracked on Danilov |
| 2.5 | 5–7 | **Calibration & abstention:** ECE + reliability + OOD defer path | ECE < ~0.05; defer path demonstrably fires on OOD inputs |
| 3 | 5–10 | Temporal + catheter: keyframe 2D + ConvLSTM-lite (DSANet offline second-read); YOLO11n+ByteTrack | DSA Dice ~0.85; real-time tracking |
| 3b | 8–11 | **Cross-vendor validation:** leave-one-vendor-out | Held-out-vendor Dice/F1 gap reported and within agreed bound |
| 4 | 8+ | Domain extensions: AVF audio + tabular first; AVF imaging + TAVR CT (data-gated) | AVF audio screen validated vs duplex; TAVR CT ICC ≥ 0.95 vs expert |
| 5 | future | **Regulatory / intended-use gate:** assistive vs autonomous, SaMD class, prospective validation | Named before any non-research use — not discovered late |

**First move:** Stage 0 + Stage 1 on ARCADE — highest-readiness data, clearest edge win, and it de-risks the whole teacher→distill→quantize→**gate** loop before the harder temporal and data-desert problems.

---

## 5. Caveats

- Coronary XCA datasets are small and vendor-heterogeneous; expect domain shift and validate cross-vendor (§3.3).
- ARCADE masks are disjoint branch regions, not full vessel trees; DCA1 gives complete masks but mostly *normal* anatomy — combine deliberately.
- AV-fistula and TAVR open *imaging* data are essentially unavailable; budget for primary collection + IRB.
- Several strong numbers (CathAction, some Mamba/CASR-Net results, private TAVR/AVF cohorts) come from arXiv preprints or in-house data — treat as indicative, confirm on your own splits.
- The accuracy floors are **provisional placeholders** for a research v0; set them with clinical stakeholders per intended use before they gate any real decision.
