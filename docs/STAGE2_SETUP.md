# Stage 2 — Stenosis detection: setup & run guide

YOLO11s @ 768px on **ARCADE task-2 + Danilov (capped) + CADICA**. Target **F1 ≥ 0.57**
(recall-weighted). Build on a Kaggle/Colab **T4 GPU**; the portable `best.pt` converts to CoreML on
the Mac. Runbook: [`pipelines/stage2_stenosis.md`](../pipelines/stage2_stenosis.md).
Notebook: [`notebooks/kaggle_stenosis_plug_and_play.ipynb`](../notebooks/kaggle_stenosis_plug_and_play.ipynb).

> **The lever is patients, not frames.** The honest, **patient-grouped** F1 is **0.214**
> (ARCADE+Danilov) — *below* the **0.57** floor, and *below* ARCADE-only's **0.246**. Adding
> Danilov's 8,325 near-duplicate frames (only 64 patients) actually **lowered** the honest metric,
> which is why Danilov is now capped to 5 frames/patient. The bottleneck is **patient count**, so the
> #1 accuracy move is adding patient-diverse data (CADICA, then others) — not more frames.

---

## 1 · Datasets needed

Place each under `data/raw/<attach-as>/`, then run the matching prep (or attach on Kaggle and let the
notebook symlink them — see §2). Only ARCADE is required; Danilov/CADICA/XCAD auto-skip if absent.

| Dataset | What it is | Patients / frames | How to get it (exact URL) | Attach-as |
|---|---|---|---|---|
| **ARCADE** task-2 | Coronary stenosis boxes (COCO); the labeled base + backbone diversity | ~1,500 stenosis images | COCO release: `https://zenodo.org/records/10390295` (optional YOLO labels `https://zenodo.org/records/10390265`; code `https://github.com/cmctec/ARCADE`) | `arcade` |
| **Danilov** | Coronary stenosis (Kemerovo); redundant per-patient frames, **now capped 5/patient** | 8,325 frames / **64 patients** | `https://data.mendeley.com/datasets/ydrm75xywg/1` (Mendeley `ydrm75xywg`) | `danilov` |
| **CADICA** | 2024 patient-diverse lesion angiography video — **the key patient-diversity add** | **42 patients** (p1–p42), 668 videos, keyframes-only boxes (~2.86 GB) | `https://data.mendeley.com/datasets/p9bpx9ctcv` (DOI `10.17632/p9bpx9ctcv`, Jiménez-Partinen 2024) | `cadica` |
| **XCAD** (unlabeled) | Unlabeled XCA frames — **only if running SSL** (pseudo-label / gdino seed / backbone pretrain) | 1,621 unlabeled (+126 labeled) | `https://github.com/AISIGSJTU/SSVS` (ICCV-2021 "Self-Supervised Vessel Segmentation"; + Kaggle "XCAD" mirror) | point `ssl.unlabeled_dir` at it |

Notes:
- **ARCADE**: attach only the **stenosis** split (config `datasets.arcade_stenosis` = task 2). MICCAI 2023, 25 SYNTAX regions.
- **Danilov**: coronary stenosis **only** — do not use as AV-fistula data. Ships `.bmp` frames with Pascal-VOC XML or YOLO `.txt` boxes; all severity classes collapse to the single `stenosis` class.
- **CADICA** layout is `selectedVideos/pX/vY/input/*.png` with sibling lowercase `groundtruth/*.txt` (`x y w h [severity]`, absolute-px top-left). Split is **patient-grouped** (`pX`), keyframes-only (a frame converts iff it has a GT `.txt`). Download does **not** auto-run — fetch + attach it by hand.
- **XCAD is not auto-detected** by the notebook (only arcade/danilov/cadica are). SSL is a manual, opt-in path: attach a disjoint unlabeled-only dir and set `ssl.unlabeled_dir`, or SSL stays off (it would otherwise re-leak val patients into train).

---

## 2 · Kaggle attach names

The notebook auto-wires inputs from `/kaggle/input` with **no paths to edit**:

- **ARCADE** — detected by its content (`**/stenosis/*/annotations/*.json`), so any title works; name it **`arcade`** for clarity. Required.
- **Danilov** — detected by **folder name containing `danilov`** (e.g. title it **`danilov-stenosis`**). Symlinked to `data/raw/danilov`.
- **CADICA** — detected by **folder name containing `cadica`** (title it **`cadica`**; the folder holds `selectedVideos/`). Symlinked to `data/raw/cadica`.

Any subset works; a missing dataset is dropped from the config and skipped. XCAD (if used) is attached separately and referenced via `ssl.unlabeled_dir`, not by auto-detect.

---

## 3 · Chore checklist (Kaggle)

1. **Settings → Accelerator → GPU (T4).** The notebook asserts CUDA is available and fails loud otherwise.
2. **Settings → Internet → ON** (needed for `pip install` + `git clone`).
3. **+ Add Input**: attach ARCADE, Danilov, CADICA (names per §2).
4. **pip deps** (installed by the notebook; here for reference):
   - `ultralytics>=8.2 pycocotools opencv-python pyyaml`
   - **only if seeding SSL with Grounding DINO** (`ssl.seed: gdino`): add `transformers` + `Pillow`.
5. **DRY_RUN flow** — first pass keeps `DRY_RUN = True` (3-epoch wiring check, SSL off), then set **`DRY_RUN = False`** and Run All again for the real **150-epoch** run. OOM at batch 16 auto-falls back to batch 8.
6. **Where outputs land** — copied to `/kaggle/working/` (downloadable from the Output panel):
   - `best.pt`, `results.csv`, `stenosis_run.zip` (full run dir), `stenosis_demo.mp4` (val-only GT-vs-pred overlay).
   - The repo + 9k processed PNGs live in `/kaggle/tmp` (not saved to output). Weights also sit at `runs/stenosis/<TAG>/base/weights/best.pt`.
7. **Pull the outputs locally**:
   ```bash
   kaggle kernels output jugalmodi0111/stenosis -p <dir>
   # then on the Mac:
   python -m src.export.yolo_to_coreml --weights best.pt
   ```

> A **leakage audit** (`io_utils.audit_split_leakage`, notebook §3b) runs **before** training and
> **hard-fails** if any patient/clip is in both train and val, or if Danilov's frames weren't
> collapsed by patient. If it raises, any reported F1 would be inflated — fix the split first.

---

## 4 · Config knobs (`configs/stenosis_yolo.yaml`)

| Key | Default | What it does |
|---|---|---|
| `datasets.danilov.max_frames_per_patient` | `5` | Caps each Danilov patient to N evenly-spaced frames (`io.cap_frames_per_patient`) so 8,325 near-duplicate frames from 64 patients can't dilute the honest per-patient metric. `null` = keep all. |
| `datasets.cadica.root` | `data/raw/cadica` | Path to the CADICA download; `format: native`. `cadica_to_yolo` **skips silently** if the root is absent — set it to include the patient-diversity add. |
| `val.conf` | `0.001` | Low eval confidence so recall (the clinically costly axis) isn't throttled at validation (`train_detector.val_kwargs`). Add `val.iou` to override NMS IoU. |
| `target.f1` | `0.57` | F1 floor gate — `qualifies_det` PASS/FAIL. (Note: ARCADE stenosis SOTA ≈0.5, so 0.57 may exceed published — confirm with clinical stakeholders.) |
| `target.recall` | *(commented)* | Optional **recall-first** second gate on top of F1 — a missed stenosis is the deadly error. Uncomment + set the clinical recall floor (e.g. `0.60`) to enable. |
| `model.pretrained_weights` | *(commented)* | Drop-in path to an SSL/angiography-pretrained backbone; `_load_pretrained_backbone` loads it `strict=False` before training (missing file/shape-mismatch = warn, not fatal). |

Also relevant: `model: {name: yolo11s, imgsz: 768}` (the 's'+768 combo needed to clear the floor — nano/640 saturated at ~0.25, still edge-deployable), and `ssl: {pseudo_label, conf, seed: gdino}` for the SSL path.

---

## 5 · Accuracy levers, ROI-ranked

1. **More patients** (CADICA now, others next) — patient count is the bottleneck; `cadica_to_yolo.py` adds 42 patient-diverse patients with a patient-grouped split. Patients > frames. *(Highest ROI.)*
2. **Recall-first gate + low-conf eval** — `qualifies_det` (optional `target.recall`) + `val_kwargs` (`val.conf: 0.001`) in `src/train/train_detector.py` stop a missed lesion from passing as "good enough."
3. **Temporal voting at inference** — `src/serve/temporal_vote.py` `aggregate_sequence` links per-frame detections into tracks over a cine window: `min_hits` persistence drops one-frame flicker (precision) and gap-interpolation recovers missed frames (recall).
4. **Triage / abstention for safe shipping** — `src/serve/stenosis_triage.py` `triage_decision` runs the sub-floor model as high-recall screening that **defers** OOD / low-confidence / possible-miss cases to a human instead of being confidently wrong.
5. **SSL backbone** — `model.pretrained_weights` + `_load_pretrained_backbone` (self-supervised pretrain on unlabeled XCA), plus `ssl.seed: gdino` / `ssl.pseudo_label` self-training. *(Lowest ROI until the backbone exists.)*

---

## 6 · What's still manual

- **SSL backbone weights** — the self-supervised/angiography-pretrained checkpoint (`runs/ssl/xca_backbone.pt`) must be produced GPU-side; only the load hook is wired.
- **Clinical recall floor** — `target.recall` is commented out; the actual floor needs clinical-stakeholder sign-off before the recall-first gate is enabled.
- **CADICA download** — fetch from Mendeley (`10.17632/p9bpx9ctcv`) and attach/point `datasets.cadica.root` by hand; the converter runs automatically once the root exists, but the download is not automated.
