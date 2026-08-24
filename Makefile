PY=python
prep-coronary:
	$(PY) -m src.data_prep.arcade_to_coco --config configs/coronary_seg.yaml
	$(PY) -m src.data_prep.dca1_to_nnunet --config configs/coronary_seg.yaml
prep-stenosis:
	$(PY) -m src.data_prep.danilov_to_yolo --config configs/stenosis_yolo.yaml
	$(PY) -m src.data_prep.cadica_to_yolo --config configs/stenosis_yolo.yaml   # patient-diverse add (skips if cadica.root absent)
train-coronary:
	$(PY) -m src.train.train_seg --config configs/coronary_seg.yaml
train-stenosis:
	$(PY) -m src.train.train_detector --config configs/stenosis_yolo.yaml
train-avf-audio:
	$(PY) -m src.train.train_audio --config configs/avf_audio.yaml
# Model One head: default --backbone is test-tiny, the OFFLINE TEST backbone. A real run passes
# ARGS=--backbone dinov2_vitb14 --imgsz 224 and is gated on B5/B9 (no real frames exist yet).
train-avf-cls:            # FRAMES=<frame store> LABELS=<labels.jsonl> OUT=<dir> [ARGS=...]
	$(PY) -m src.train.train_classifier --frames $(FRAMES) --labels $(LABELS) --out $(OUT) $(ARGS)
export:
	$(PY) -m src.export.to_onnx --weights $(MODEL)
	$(PY) -m src.export.quantize_int8 --model $(MODEL:.pt=.onnx)
bench:
	$(PY) -m src.eval.edge_benchmark --model $(MODEL)

# --- Mac side (Apple silicon): CoreML export + compress + clDice gate + benchmark ---
# Run these on macOS after pulling the student state_dict from Drive.
export-coreml:            # MODEL=runs/coronary/student.pt  (state_dict from Colab build)
	$(PY) -m src.export.to_coreml --weights $(MODEL) --method palettize --nbits 6
validate-coreml:          # CORE=...mlpackage WEIGHTS=...pt IMAGES=... MASKS=...  (HARD gate)
	$(PY) -m src.export.coreml_validate --coreml $(CORE) --weights $(WEIGHTS) \
		--images $(IMAGES) --masks $(MASKS)
bench-coreml:             # MODEL=runs/coronary/student.mlpackage
	$(PY) -m src.eval.edge_benchmark --model $(MODEL)
export-coreml-yolo:       # MODEL=runs/stenosis/.../weights/best.pt  (Ultralytics, NMS baked in)
	$(PY) -m src.export.yolo_to_coreml --weights $(MODEL)

# --- Stenosis (YOLO stack) ---
prep-stenosis-yolo:
	$(PY) -m src.data_prep.danilov_to_yolo --config configs/stenosis_yolo.yaml
	$(PY) -m src.data_prep.cadica_to_yolo --config configs/stenosis_yolo.yaml   # patient-diverse add (skips if cadica.root absent)
train-detector:
	$(PY) -m src.train.train_detector --config configs/stenosis_yolo.yaml

# --- Catheter / guidewire (YOLO + ByteTrack) ---
prep-catheter:
	$(PY) -m src.data_prep.cathaction_to_yolo --config configs/catheter_track.yaml
train-catheter:
	$(PY) -m src.train.train_detector --config configs/catheter_track.yaml
# (track / track-eval / realtime targets removed with the serve video path -- 2026-08-03 audit P3:
#  Model One is single-still-frame only; src/serve/{track,realtime,temporal_vote}.py were deleted)

# --- Inference (Mac): local API ---
serve:                    # local API; MODEL=...mlpackage TASK=seg|det
	MODEL=$(MODEL) TASK=$(TASK) uvicorn src.serve.app:app --host 127.0.0.1 --port 8000

# --- Dialygo ingest (institutional fistulography DICOM -> de-identified PNG frames) ---
# HARD GATE (Dialygo B5): real patient data may not be processed until the institutional
# data-use agreement executes, and B9 (IP/engagement agreement) must execute before
# real-data development begins. MODE defaults to `synthetic` so the safe path is the one
# you get by typing nothing. VALID_MODES = ("synthetic", "real") in src/ingest/clearance.py
# -- there is no "cleared" mode. `MODE=real` is checked by src/ingest/clearance.py against
# configs/ingest_clearance.yaml and refuses unless BOTH B5/B9 flags there are true. deid.py's
# CLI only provisions key material and takes NO --mode -- do not pass one.
SITE       ?= inu
MODE       ?= synthetic
SRC        ?=                       # drive root to scan (ingest-scan/ingest-hdd); leave empty until B5 clears
SOURCE     ?=                       # single DICOM/video file to extract (ingest-extract)
WORK       ?= .ingest/$(SITE)
CLEAN_ROOT ?= $(HOME)/dialygo_clean
SALT       ?= $(CLEAN_ROOT)/$(SITE)/_keys/salt.bin
LINK_NAME  ?= avf_fistulography
INGEST_CFG ?= configs/ingest_sites.yaml
LABELS     ?=                       # CSV / COCO json / mask dir (ingest-labels)
KIND       ?= csv                   # csv|coco|mask_dir (ingest-labels)
LIMIT      ?=                       # ingest-hdd: cap new instances processed per phase (pilot runs)
ACK_PHI    ?=                       # ingest-hdd: set to 1 once <work>/phi_audit.md has been read (real mode)

ingest-scan:               # Phase 1: SRC=/Volumes/<drive> -- read-only inventory -> $(WORK)/files.jsonl
	$(PY) -m src.ingest.scan --src $(SRC) --out $(WORK) --site $(SITE) --mode $(MODE)
ingest-index:               # Phase 2: DICOM headers -> $(WORK)/dicom_index.jsonl (patient/study/series/SOP)
	$(PY) -m src.ingest.index_dicom --files $(WORK)/files.jsonl --out $(WORK) --site $(SITE) --mode $(MODE)
ingest-deid:                # provisions the 0600 HMAC salt at $(SALT) -- no --mode on this CLI (see header note)
	$(PY) -m src.ingest.deid --salt $(SALT) --site $(SITE)
ingest-extract:              # SOURCE=<dicom-or-video-file> -- de-identified PNG frames + sidecar for ONE study
	$(PY) -m src.ingest.extract $(SOURCE) --out-root $(CLEAN_ROOT)/$(SITE) --site $(SITE) --salt $(SALT) --mode $(MODE)
ingest-labels:               # LABELS=<csv|coco.json|maskdir> KIND=csv|coco|mask_dir -- join clinician labels (B7)
	$(PY) -m src.ingest.labels --index $(WORK)/dicom_index.jsonl --labels $(LABELS) \
		--kind $(KIND) --key StudyInstanceUID --out $(WORK)/labels.jsonl --mode $(MODE)
ingest-link:                # data/raw/$(LINK_NAME) -> $(CLEAN_ROOT)/$(SITE)/frames  (symlink, never a copy)
	$(PY) -m src.ingest.link --clean-frames $(CLEAN_ROOT)/$(SITE)/frames \
		--data-raw data/raw --name $(LINK_NAME) --mode $(MODE)
ingest-doctor:               # health check: drives mounted, links resolve, manifests parse, no PHI in repo
	$(PY) -m src.ingest.doctor --config $(INGEST_CFG) --repo-root .
ingest-hdd:                  # end-to-end driver: scan -> index -> PHI audit -> deid -> extract (scripts/ingest_hdd.py)
	$(PY) scripts/ingest_hdd.py --src $(SRC) --site $(SITE) --mode $(MODE) --work $(WORK) --clean-root $(CLEAN_ROOT) \
		$(if $(LIMIT),--limit $(LIMIT)) $(if $(ACK_PHI),--ack-phi-audit)
