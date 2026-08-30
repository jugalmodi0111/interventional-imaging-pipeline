"""Static INT8 PTQ via onnxruntime — configs/edge_export.yaml `int8:` is the contract.

Calibration must feed the model EXACTLY what it sees at inference. The processed val PNGs
already carry CLAHE+unsharp (data_prep.preprocess.process_dir applies it before writing), so
here it is just: grayscale read -> resize(size,size) -> float32/255 -> [1,1,H,W] — the same
tensor coreml_validate._load_pairs builds. Re-check clDice after quantizing thin-vessel models.

    python -m src.export.quantize_int8                     # config-driven, proven artifacts
    python -m src.export.quantize_int8 --model m.onnx --calib-dir imgs/ --out m.int8.static.onnx
    python -m src.export.quantize_int8 --dynamic           # explicit data-free fallback

OUTPUT NAMES CARRY THE METHOD. static PTQ and dynamic INT8 are different artifacts with
different accuracy, and only the static one has ever been through the clDice gate. They used
to share one default name ("<model>.int8.onnx"), so a bare run of this module silently
replaced whichever build was already there — two provenances, one filename, no way to tell
them apart on disk. Defaults are now `<model>.int8.static.onnx` / `<model>.int8.dynamic.onnx`,
and an explicit --out onto an existing file that is not this method's own artifact is refused
unless --force. See default_out() / _check_out().
"""
import argparse, glob, os
import numpy as np

_MODEL = "outputs/coronary_student_clgeodice/student.onnx"
_CALIB = "data/processed/coronary/val/img"
_CONFIG = "configs/edge_export.yaml"
_METHOD_TAG = {"static_ptq": "static", "dynamic": "dynamic"}


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


def default_out(model, method):
    """`<model>.onnx` -> `<model>.int8.<static|dynamic>.onnx`. The method is in the FILENAME
    so a static build cannot land on a dynamic one (or vice versa) and leave the artifact's
    provenance — and therefore whether the clDice gate ever scored it — unknowable."""
    stem = model[:-len(".onnx")] if model.endswith(".onnx") else model
    return f"{stem}.int8.{_METHOD_TAG[method]}.onnx"


def _check_out(out, method, force=False):
    """Refuse to overwrite an existing artifact that is not this method's own. Re-running the
    same method over its own (method-tagged) output is normal and allowed; landing a static
    build on `student.int8.onnx` or `student.int8.dynamic.onnx` is the collision we are here
    to stop. --force is the deliberate escape hatch."""
    tag = _METHOD_TAG[method]
    if os.path.exists(out) and f".int8.{tag}." not in os.path.basename(out) and not force:
        raise FileExistsError(
            f"refusing to overwrite {out} with a {tag} INT8 build: that filename is not this "
            f"method's artifact, so the result's provenance (and which gate scored it) would "
            f"be unrecoverable. Use --out {default_out(out, method)} or pass --force.")
    return out


def quantize(model, out=None, calib_dir=_CALIB, config=_CONFIG, dynamic=False, force=False):
    from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static
    cfg = _int8_cfg(config)
    method = "dynamic" if dynamic else cfg.get("method", "static_ptq")
    if method not in _METHOD_TAG:
        raise ValueError(f"unknown int8 method {method!r} in {config}")
    out = _check_out(out or default_out(model, method), method, force)
    if method == "dynamic":
        quantize_dynamic(model, out, weight_type=QuantType.QInt8)
        print("wrote", out, "(dynamic INT8 — explicit fallback; config contract is static_ptq)")
        return out

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
    ap.add_argument("--out", help="default: <model>.int8.<static|dynamic>.onnx")
    ap.add_argument("--calib-dir", default=_CALIB, help="processed (CLAHE'd) grayscale PNGs")
    ap.add_argument("--config", default=_CONFIG)
    ap.add_argument("--dynamic", action="store_true", help="data-free dynamic INT8 fallback")
    ap.add_argument("--force", action="store_true",
                    help="allow --out to overwrite another method's INT8 artifact")
    a = ap.parse_args()
    quantize(a.model, a.out, calib_dir=a.calib_dir, config=a.config, dynamic=a.dynamic,
             force=a.force)
