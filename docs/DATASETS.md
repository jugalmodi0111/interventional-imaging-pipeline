# Datasets to download why, where

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


---

## AngioCAD — VERIFIED 2026-08-23 (labels file inspected, not inferred)

**Zenodo `10.5281/zenodo.15826856`** · CC-BY-4.0 · 16.4 GB (4 RAR parts + 2 xlsx) · sequential PNG
frames de-identified from DICOM.

**IT HAS NO BOUNDING BOXES.** This was checked by downloading `AngioCAD_Labels.xlsx` (43 kB) and
reading it — cheap, and it contradicts secondary sources that describe per-frame boxes. The file is
**one row per patient**, 413 rows x 18 columns:

```
ID | Right Coronary Series | Left Coronary Series | LM | Prox LAD | Mid LAD | Dist LAD |
1st dig | 2nd dig | Prox LCX | Mid LCX | Dist LCX | OM | Prox RCA | Mid RCA | Dist RCA | PDA | PLB
```

Values are one of seven severity grades: `NL`, `1-25`, `26-50`, `51-75`, `76-90`, `91-99`, `100`.
A 43 kB spreadsheet cannot hold per-frame boxes for 413 videos; the size alone was the tell.

| | count |
|---|---|
| segment labels | 6,195 (4,603 `NL` = 74%, 1,592 diseased) |
| patients | 413 (360 with disease, **53 all-normal = 12.8%**) |

**Consequence — this dataset CANNOT train the YOLO stenosis detector.** No localization targets. It is
a **patient/segment-level classification** dataset, and a good one: real negatives at both segment
(74%) and patient (12.8%) level, plus a series->artery mapping so videos can be joined to the label
of the artery they actually show.

**So the "more patients for the detector" and "reformulate as a study-level classifier" options are
the same path** — the label format forces it. That also makes the work dual-use: a study-level
classifier is the same machinery Model One (AVF) needs, so the proxy path and the product path share
code rather than diverging. This retroactively confirms the tracker's proposed Task 8 name,
`angiocad_to_cls.py`, was right.

**Operational notes:** 4-part RAR needs `unrar`/`7z` (not stdlib).

### MEASURED 2026-08-25 on Kaggle — the "3.4 TB extracted" figure was WRONG

~~the record reports 3.4 TB extracted, so extraction must be selective, not wholesale~~ — struck,
not deleted, because it governed planning for two days. It was **unsourced**: it appears nowhere in
the Zenodo record, and PNG is already DEFLATE-compressed so a 200x expansion was never physically
possible. Measured from the RAR headers without extracting
(`notebooks/kaggle_angiocad_acquire.ipynb` Cell 3):

| | measured |
|---|---|
| entries | **121,566**, all `.png` |
| uncompressed | **16.37 GB** (archive on disk 16.40 GB) |
| expansion ratio | **1.00x** |
| claimed | 3.4 TB — a **208x** overstatement |
| full extraction | **3.2 min**; `/kaggle/temp` had 1,102 GB free |

**Extract wholesale. There is no selective-extraction problem.** Download is 37 min at Kaggle's
5-16 MB/s (a 0.5 MB/s reading from a laptop is a local-network artifact, not Zenodo's ceiling).

**Real layout — VERIFIED against the tree, not assumed:** `AngioCAD_Dataset/<patient>/<series>/frame_%04d.png`.
`angiocad_to_cls`'s assumed `<root>/<patient>/<series>/` resolves **2,606 of 2,644 videos (98.6%)**.

| | |
|---|---|
| patient dirs on disk | **412, not 413** — one patient is absent from the archive entirely |
| unresolved videos | 38 across 16 patients (worst: 157 x6, 63 x5, 413 x5, 393 x4) |
| orphan folders | 105 on disk the sheet never names (e.g. patient 136 series 11) — unlabelled videos, unusable |
| frames per video | min 11, **median 53**, max 512 |
| positives | 1,686 @50% (63.8%) / 1,524 @70% (57.6%) — 162 videos flip; **Dr. Reddy's call** |
