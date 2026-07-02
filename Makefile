PY=python
prep-coronary:
	$(PY) -m src.data_prep.arcade_to_coco --config configs/coronary_seg.yaml
	$(PY) -m src.data_prep.dca1_to_nnunet --config configs/coronary_seg.yaml
prep-stenosis:
	$(PY) -m src.data_prep.danilov_to_yolo --config configs/stenosis_yolo.yaml
train-coronary:
	$(PY) -m src.train.train_seg --config configs/coronary_seg.yaml
train-stenosis:
	$(PY) -m src.train.train_detector --config configs/stenosis_yolo.yaml
train-avf-audio:
	$(PY) -m src.train.train_audio --config configs/avf_audio.yaml
export:
	$(PY) -m src.export.to_onnx --weights $(MODEL)
	$(PY) -m src.export.quantize_int8 --onnx $(MODEL:.pt=.onnx)
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
train-detector:
	$(PY) -m src.train.train_detector --config configs/stenosis_yolo.yaml

# --- Catheter / guidewire (YOLO + ByteTrack) ---
prep-catheter:
	$(PY) -m src.data_prep.cathaction_to_yolo --config configs/catheter_track.yaml
train-catheter:
	$(PY) -m src.train.train_detector --config configs/catheter_track.yaml
track:                    # WEIGHTS=runs/catheter/.../best.pt SOURCE=clip.mp4
	$(PY) -m src.serve.track --weights $(WEIGHTS) --source $(SOURCE)

# --- Inference (Mac): per-frame overlay + audit trail ---
realtime:                 # MODEL=...mlpackage TASK=seg|det SOURCE=clip.mp4|frames/|camera
	$(PY) -m src.serve.realtime --model $(MODEL) --task $(TASK) --source $(SOURCE) --show
serve:                    # local API; MODEL=...mlpackage TASK=seg|det
	MODEL=$(MODEL) TASK=$(TASK) uvicorn src.serve.app:app --host 127.0.0.1 --port 8000
