# Datasets to download — what, why, where

Scope-ordered. **Download the v1 core first** (coronary → stenosis → catheter); everything else is
later-stage or data-gated. Metadata here is the *corrected* version from
`Angiography_Dataset_Validation_Scoring.xlsx` (the raw source doc had errors — noted inline).

Place each under `data/raw/<name>/`, then run the matching `make prep-*`. On Kaggle, attach as a
Kaggle Dataset (mounts read-only at `/kaggle/input/`) and symlink into `data/raw/`.

## v1 CORE — download these now

| Dataset | Problem | Access | Scale | Where | Feeds |
|---|---|---|---|---|---|
| **ARCADE** | coronary seg **+** stenosis | Open | 3,000 img (1,500 seg + 1,500 sten) | Zenodo **8386059** / **10390295**; github.com/cmctec/ARCADE | `arcade_to_coco` (task1), `danilov_to_yolo` (task2) |
| **DCA1** | coronary seg | Open | 134 img (+GT) | personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html | `dca1_to_nnunet` |
| **XCAD** | coronary seg (SSL pretrain) | Open | 126 labeled + 1,621 unlabeled | released w/ ICCV 2021 paper | SSL pool + `train_detector` pseudo-labels |
| **Danilov** | stenosis | Open | 8,325 img / 100 patients | data.mendeley.com/datasets/ydrm75xywg/1 | `danilov_to_yolo` |
| **CathAction** | catheter/guidewire | Open | ~500k frames / 25k masks | airvlab.github.io/cathaction ; HF `airvlab/CathAction` | catheter track (YOLO+ByteTrack) |

Corrections to respect (from the audit):
- **ARCADE** = MICCAI **2023**, **25** SYNTAX regions, Zenodo **8386059** (not 7981245).
- **DCA1** = Cervantes-Sanchez **2019**, CIMAT/IMSS (not "Moorchung 2016 / Kaggle"). ~130–134 imgs.
- **Danilov** = **coronary** stenosis only (Kemerovo). Do **NOT** use it as AV-fistula data.
- **CADICA** (optional extra coronary video, Mendeley p9bpx9ctcv) = 668 videos / 42 patients — ignore the "3,000 athlete images" description; it's fabricated.

## Stage 3 — cerebral DSA (temporal). Download when you reach it.
| Dataset | Access | Scale | Where |
|---|---|---|---|
| **DIAS** | Open | 120 sequences | Zenodo / DIAS benchmark page |
| **DSCA** | Open | 224 seq / 1,792 img / 58 patients | github.com/jiongzhang-john/DSCA |

## Stage 4 — TAVR (offline, proxy-only). No TAVR-specific public data exists.
| Dataset | Access | Role | Where |
|---|---|---|---|
| **MM-WHS** | Open (register) | whole-heart proxy | MM-WHS challenge |
| **Seg.A. 2023** | Open | aorta proxy | Seg.A. 2023 challenge |

## Pretraining / validation only (optional — not required for the core)
CXR sets for encoder pretrain + external validation: **NIH ChestX-ray14** / **PadChest** (truly open),
**CheXpert** (register), **MIMIC-CXR / VinDr-CXR** (PhysioNet CITI+DUA — **apply EARLY**, weeks of lead time).

## Data deserts — nothing to download; primary collection + IRB
- **AVF imaging** (US / fistulography / DSA): no public benchmark. Institutional collection + IRB.
- **AVF audio (bruit)** & **AVF tabular**: build from your own duplex-labeled cohort (Zhou/Ota style).
- **TAVR CT**: no public; use MM-WHS/Seg.A proxies + domain-adapt to institutional CT.

## Direct download links

✅ = link confident · 🔎 = verify the exact record on the landing page (I'm less sure of the precise URL).

### v1 core (download now)
- **ARCADE** ✅ **use the COCO release: https://zenodo.org/records/10390295** (Zenodo "Version COCO", Dec 2023 — matches `arcade_to_coco`/`coco_to_yolo`). Optional YOLO labels: https://zenodo.org/records/10390265 . Older `8386059` (final_phase, May 2023) superseded. Code https://github.com/cmctec/ARCADE
- **DCA1** ✅ http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html (also Kaggle mirrors)
- **Danilov** ✅ https://data.mendeley.com/datasets/ydrm75xywg/1
- **XCAD** 🔎 released with ICCV-2021 "Self-Supervised Vessel Segmentation" — repo https://github.com/AISIGSJTU/SSVS (+ Kaggle "XCAD" mirror); confirm the frames download
- **CathAction** ✅ https://airvlab.github.io/cathaction/ · HF https://huggingface.co/datasets/airvlab/CathAction · paper https://arxiv.org/abs/2408.13126

### Recommended add (see optimality note)
- **CADICA** ✅ https://data.mendeley.com/datasets/p9bpx9ctcv (Jiménez-Partinen 2024)
- **CoronaryDominance** 🔎 search Zenodo/GitHub "CoronaryDominance 2024" (dominance + quality tags)

### Stage 3 — cerebral DSA
- **DIAS** 🔎 DIAS benchmark page → Zenodo (search "DIAS intracranial artery segmentation DSA")
- **DSCA** 🔎 https://github.com/jiongzhang-john/DSCA (verify org/repo name on the DSANet paper)

### Stage 4 — TAVR proxies
- **MM-WHS** ✅ https://zmiclab.github.io/zxh/0/mmwhs/
- **Seg.A. 2023** ✅ https://multicenteraorta.grand-challenge.org/
- **ImageCAS** (3D coronary, optional) ✅ https://github.com/XiaoweiXu/ImageCAS

### Fluoro extras (optional complements)
- **WEISS** 🔎 figshare — search "Mazomenos catheter fluoroscopy" (CC-BY, 2023 release)
- **AttWire** ✅ paper https://arxiv.org/abs/2503.06190 · Kaggle "X-ray Fluoroscopic images" (Ying Liang Ma)
- **DeepFluoro** ✅ https://github.com/rg2/DeepFluoroLabeling-IPCAI (data link in README)
- **Veriserum** ✅ https://arxiv.org/abs/2509.05483 (MICCAI 2025; data link in paper)
- **Ljubljana DSA** ✅ https://lit.fe.uni-lj.si/en/research/resources/3D-2D-GS-CA/

### CXR (pretrain / validation only)
- **MIMIC-CXR** ✅ https://physionet.org/content/mimic-cxr-jpg/ (CITI + DUA)
- **CheXpert** ✅ https://stanfordmlgroup.github.io/competitions/chexpert/ (register)
- **CheXpert Plus** ✅ https://github.com/Stanford-AIMI/chexpert-plus (register)
- **NIH ChestX-ray14** ✅ https://nihcc.app.box.com/v/ChestXray-NIHCC · Kaggle `nih-chest-xrays/data`
- **PadChest** ✅ https://bimcv.cipf.es/bimcv-projects/padchest/
- **BRAX** ✅ https://physionet.org/content/brax/ (CITI + DUA)
- **VinDr-CXR** ✅ https://physionet.org/content/vindr-cxr/ (CITI + DUA)
- **RSNA Pneumonia** ✅ https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

### Eval
- **MIRA** 🔎 AAAI 2026 — https://ojs.aaai.org/index.php/AAAI/article/view/37549 (find the linked project/HF page)

## Config wiring
Each `configs/*.yaml` `datasets:` block points `root:` at `data/raw/<name>/`. Change the path there,
not in code. `arcade_to_coco` / `dca1_to_nnunet` auto-discover the COCO json + image/GT pairs under
the root — confirm the download unzipped into that folder.
