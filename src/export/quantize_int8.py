"""Static INT8 PTQ via onnxruntime — configs/edge_export.yaml `int8:` is the contract.

Calibration must feed the model EXACTLY what it sees at inference. The processed val PNGs
already carry CLAHE+unsharp (data_prep.preprocess.process_dir applies it before writing), so
here it is just: grayscale read -> resize(size,size) -> float32/255 -> [1,1,H,W] — the same
tensor coreml_validate._load_pairs builds. Re-check clDice after quantizing thin-vessel models.

    python -m src.export.quantize_int8                     # config-driven, proven artifacts
    python -m src.export.quantize_int8 --model m.onnx --calib-dir imgs/ --out m.int8.onnx
    python -m src.export.quantize_int8 --dynamic           # explicit data-free fallback
"""
import argparse, glob, os
import numpy as np

_MODEL = "outputs/coronary_student_clgeodice/student.onnx"
_CALIB = "data/processed/coronary/val/img"
_CONFIG = "configs/edge_export.yaml"


class PngCalibrationReader:
    """onnxruntime CalibrationDataReader (duck-typed via get_next): streams grayscale PNGs
    as {input_name: [1,1,size,size] float32 in [0,1]}, capped at what exists on disk."""

    def __init__(self, calib_dir, input_name="input", size=512, limit=200):
        found = sorted(glob.glob(os.path.join(calib_dir, "*.png")))
        assert found, f"no .png calibration images under {calib_dir}"
        if len(found) < limit:
            print(f"calibration: using {len(found)} of {limit} requested images ({calib_dir})")
        self.paths, self.input_name, self.size = found[:limit], input_name, size
        self._it = iter(self.paths)

    def get_next(self):
        import cv2
        p = next(self._it, None)
        if p is None:
            return None
        im = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (self.size, self.size))
        return {self.input_name: im[None, None].astype(np.float32) / 255.0}

    def rewind(self):
        self._it = iter(self.paths)


def _int8_cfg(config):
    import yaml
    if not (config and os.path.exists(config)):
        return {}
    with open(config) as f:
        return (yaml.safe_load(f) or {}).get("int8", {})


def quantize(model, out=None, calib_dir=_CALIB, config=_CONFIG, dynamic=False):
    from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static
    out = out or model.replace(".onnx", ".int8.onnx")
    cfg = _int8_cfg(config)
    method = "dynamic" if dynamic else cfg.get("method", "static_ptq")
    if method == "dynamic":
        quantize_dynamic(model, out, weight_type=QuantType.QInt8)
        print("wrote", out, "(dynamic INT8 — explicit fallback; config contract is static_ptq)")
        return out
    if method != "static_ptq":
        raise ValueError(f"unknown int8 method {method!r} in {config}")

    import onnxruntime as ort
    inp = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).get_inputs()[0]
    size = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
    reader = PngCalibrationReader(calib_dir, input_name=inp.name, size=size,
                                  limit=int(cfg.get("calib_images", 200)))
    # pre-quant optimization + shape inference; quantize_static fails cryptically without shapes
    src, pre = model, out + ".preproc.tmp.onnx"
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(model, pre, skip_symbolic_shape=True)
        src = pre
    except Exception as e:                      # best-effort: quantize the raw graph instead
        print(f"quant_pre_process skipped ({e}); quantizing unoptimized graph")
    quantize_static(src, out, reader,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    if src == pre and os.path.exists(pre):
        os.remove(pre)
    print(f"wrote {out} (static PTQ, {len(reader.paths)} calib images)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_MODEL, help="fp32 .onnx from to_onnx.py")
    ap.add_argument("--out")
    ap.add_argument("--calib-dir", default=_CALIB, help="processed (CLAHE'd) grayscale PNGs")
    ap.add_argument("--config", default=_CONFIG)
    ap.add_argument("--dynamic", action="store_true", help="data-free dynamic INT8 fallback")
    a = ap.parse_args()
    quantize(a.model, a.out, calib_dir=a.calib_dir, config=a.config, dynamic=a.dynamic)
