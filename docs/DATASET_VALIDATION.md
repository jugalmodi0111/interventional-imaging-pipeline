# Dataset validation — fact-check of the "Comprehensive Analysis" document

Cross-referenced against `Angiography_Dataset_Validation_Scoring.xlsx` (which independently verified
these with source URLs) and general knowledge. **Verdict: the document is ~85% accurate.** Every
dataset it names is REAL. Errors are concentrated in (a) internal contradictions where one dataset is
described two different ways, (b) two clear misattributions, and (c) conflating private systems/registries
with downloadable datasets. This is the same document the audit workbook scored — the previously-flagged
errors reappear here verbatim.

Legend: ✅ real + metadata OK · ⚠️ real but metadata/claim wrong · ❌ misattributed/fabricated ·
🔒 real but not a downloadable dataset (system/registry/private) · **in-repo** = wired in this pipeline's configs.

---

## Coronary angiography

| Dataset | Verdict | Reality check | In-repo |
|---|---|---|---|
| **ARCADE** | ✅ / ⚠️ | REAL. Doc's first section is right (MICCAI **2023**, 3,000 img 512×512, Philips Azurion + Siemens Artis Zee, Research Inst. of Cardiology & Internal Diseases, Almaty **Kazakhstan**, 1,500 seg + 1,500 stenosis). ⚠️ Its OWN second section calls it "MICCAI **2024**" — self-contradiction. 25 SYNTAX regions (not 26). Zenodo **8386059/10390295**. | ✅ yes (coronary + stenosis) |
| **CADICA** | ❌ / ✅ | Split personality. First section = **FABRICATED**: "3,000 grayscale images from endurance athletes, exactly 1,000 per severity class" does not exist. Second section = **CORRECT**: Jiménez-Partinen et al. 2024, **42 patients / 668 videos**, lesion bboxes. Use the second; delete the first. | not wired (optional) |
| **Danilov (8,325 img / 100 patients)** | ❌ | REAL **coronary** stenosis set (Danilov VV et al., *Sci Reports* 2021, Kemerovo; Mendeley ydrm75xywg). Document **MISATTRIBUTES** it to "AV-fistula / peripheral access vessels" in the nephrology section — HIGH severity. It is coronary only; do not seed AVF claims with it. | ✅ yes (stenosis, correctly as coronary) |
| **DCA1** | ⚠️ | REAL. Doc says "**Moorchung et al. 2016, 130 frames, Kaggle**" — WRONG author/year. Actual: **Cervantes-Sánchez et al. 2019**, CIMAT/IMSS Mexico, 134 img (100 train/34 test). | ✅ yes |
| **XCAD** | ✅ | REAL (Ma et al., ICCV 2021). 126 labeled + ~1,621 unlabeled. Not named in the doc but used in this repo for SSL. | ✅ yes (SSL) |
| **CoronaryDominance** | ✅ | REAL. 1,574 studies; dominance + quality/artifact tags. Breakdown (1,025 train / 400 real-dist / domain-shift) plausible. Moderate adoption. | not wired |

## Cerebral / DSA (temporal)

| Dataset | Verdict | Reality check | In-repo |
|---|---|---|---|
| **DSCA** | ✅ | REAL. 58 patients / 224 sequences / 1,792 img; AP + lateral; ICA/ECA/VA. DSANet Dice ~0.9033. All consistent. | ✅ yes (stage 3) |
| **DIAS** | ✅ | REAL (*Med Image Analysis* 2024, Beijing Tiantan). Weak/semi-supervised intracranial artery seg, ~120 sequences. | ✅ yes (stage 3) |
| **Ljubljana cerebral DSA** | ✅ | REAL (Mitrović et al. 2013). 10 patients / 20 projections; 2D-3D registration. Small, old. | not wired |

## Fluoroscopy / device tracking

| Dataset | Verdict | Reality check | In-repo |
|---|---|---|---|
| **CathAction** | ✅ | REAL & largest public fluoroscopy set. ~500k frames / 569 videos / 25k catheter+guidewire masks. arXiv preprint (not yet peer-reviewed). | ✅ yes (catheter) |
| **AttWire** + **"Kaggle Ying Mao"** | ⚠️ | REAL but listed as **two datasets — they are ONE**. Author is **Ying Liang Ma** ("Ying Mao" = typo). 12,438 img / 72 cases (6,533 TAVR + 250 AF + 5,655 ablation). arXiv 2503.06190 is primarily a detection **model** paper; access is request, not "open." Kaggle counts unverified in peer-reviewed lit. | not wired (planned) |
| **WEISS** | ⚠️ | REAL (Mazomenos et al., *CMPB* 2020; figshare). 2,000 phantom + 1,207 in vivo (836 TAVI + 371 diagnostic) — counts correct. ⚠️ masks are **semi-automated (tracking)**, doc's "expert segmentation masks" overstates. | not wired |
| **DeepFluoro** | ✅ | REAL (Grupp et al. 2020). 366 frames / 6 cadaveric pelves + CT. Doc honest that it's **orthopedic/pelvis, not vascular** — geometry transfer only. | not wired |
| **Veriserum** | ✅ | REAL (MICCAI 2025, arXiv 2509.05483). ~110k dual-plane **knee-implant phantom** X-rays. Doc honest: orthopedic, transfer value only. Low relevance. | not wired |

## Chest X-ray (pretrain / validation only)

| Dataset | Verdict | Reality check | In-repo |
|---|---|---|---|
| **MIMIC-CXR** | ✅ | REAL. ~377,110 img / 227,835 studies (doc's 377,095 ≈ correct), 14 NLP labels (NegBio + CheXpert). Credentialed (PhysioNet CITI+DUA). | data/README (pretrain) |
| **CheXpert** | ⚠️ | REAL (224,316 img / 65,240 patients, 14 obs w/ uncertain). ⚠️ Doc calls it "Open Access" — it is **registration/agreement-gated**. | data/README |
| **CheXpert Plus** | ⚠️ | REAL (223,462 image-text pairs, 36M tokens). ⚠️ Labeled "Open Access" — **registration-gated** in practice. | not wired |
| **NIH ChestX-ray14** | ✅ | REAL (112,120 img / 30,805 patients; small radiologist bbox subset). Truly open. | data/README |
| **PadChest** | ✅ | REAL (~160,000 img / 67,000 patients; 174 findings, 19 diff-dx, 104 locations, UMLS). Truly open. ~27% radiologist-labeled. | data/README |
| **BRAX** | ✅ | REAL (40,967 img / 24,959 studies, Hospital Israelita Albert Einstein, Brazil; Portuguese NLP). Credentialed. Correctly labeled. | not wired |
| **VinDr-CXR** | ✅ | REAL (18,000 img; doc's "14,000+" understates the 15k train). Radiologist bbox. PhysioNet credentialed. | data/README |
| **RSNA Pneumonia** | ✅ | REAL (~30k CXR w/ pneumonia bboxes). Correct. | not wired |

## Eval / reasoning

| Dataset | Verdict | Reality check | In-repo |
|---|---|---|---|
| **MIRA** | ✅ | REAL (AAAI 2026). 184K img / 1.2M QA; DSA/fluoro/CT/US + plots. Zero-shot <0.35, fine-tune ~0.80. It is an **evaluation** benchmark, NOT training data. | not wired (late-stage) |

## TAVR

| Dataset | Verdict | Reality check | In-repo |
|---|---|---|---|
| **MM-WHS** | ✅ | REAL open (register). Whole-heart CT/MR **proxy** for aortic complex — not TAVR-specific. | ✅ yes (stage 4 proxy) |
| **Seg.A. 2023** | ✅ | REAL open. Aortic CT segmentation **proxy**. No valve/annulus labels. | ✅ yes (stage 4 proxy) |
| **TAVR CT (native)** | ❌ | No public TAVR-specific CT dataset exists. Must use proxies + institutional CT + IRB. | data desert |

## AV fistula (interventional nephrology)

| Item | Verdict | Reality check |
|---|---|---|
| **AVF imaging benchmark** | ❌ | CONFIRMED **data desert** — no public downloadable hemodialysis AVF imaging set. The doc's "8,325-image peripheral-access" set is Danilov (coronary) misused. |
| **AVF audio (bruit)** | ⚠️ | Real research (Zhou npj Digital Medicine 2023, 2,565 sounds, ViT). ⚠️ Doc **blends metrics** across studies: "100% sensitivity / 75% specificity" (one cohort) vs "0.924 sensitivity" (Zhou) are different studies — cite per-study. |

---

## Claims that are NOT downloadable datasets (do not treat as such) 🔒

The document mixes real public datasets with private systems, trial registries, and single-paper cohorts.
These are **real work but you cannot download them** as training data:

- **CVPILOT** (Dice 0.9806 aorta) — a segmentation *system* trained on a private cohort.
- **Alpha Registry (NCT07016477)** — a clinical *trial* (photon-counting CT), not a dataset.
- **TAVI Risk Machine / TRIMpost** (C-stat 0.79) — a *model* on the German Aortic Valve Registry (GARY, not public).
- **SmartPatch** (100% sens / 75% spec) — a wearable *device* + private data.
- **269-parameter AVF model** (acc 0.992) — private EMR/IoMT cohort.
- **SPARTAN** — explicitly non-public.

Treat every number attached to these as *indicative*, not a reproducible public benchmark (audit flag #9).

## Model / metric claims — these held up

StenUNet F1 **0.5348** (ARCADE 3rd), U-Mamba BOT **68.79%**, DSANet Dice **0.9033**, AttWire 99.8%/97.8%/58fps,
CathAction counts, MIRA fine-tune 0.80 — all **verified**. The document is stronger on methods/metrics than on
dataset metadata. Consistent with `Model_Selection_Matrix.xlsx` picks.

---

## Fix before citing (the repeat offenders)

1. **CADICA** — drop the "3,000 athlete images / 1,000-per-class" description (fabricated); it's 42 patients / 668 videos.
2. **Danilov** — stop using its 8,325-image set as AV-fistula/peripheral data; it is coronary only.
3. **ARCADE** — MICCAI **2023**, 25 regions, Zenodo 8386059 (fix the "2024" in the second section).
4. **DCA1** — Cervantes-Sánchez **2019**, not "Moorchung 2016."
5. **AttWire = "Kaggle Ying Mao"** — one dataset; author Ying Liang **Ma**.
6. **CheXpert / CheXpert Plus** — registration-gated, not "Open Access."

## Repo mapping (what this pipeline actually wires)

**Used now:** ARCADE, DCA1, XCAD, Danilov (as coronary), CathAction, DIAS, DSCA, MM-WHS, Seg.A + CXR pretrain sets.
**Scored but not wired (optional/later):** CADICA, CoronaryDominance, AttWire, WEISS, DeepFluoro, Veriserum, Ljubljana, MIRA.
See `docs/DATASETS.md` for download order and `configs/*.yaml` for the `root:` paths.
