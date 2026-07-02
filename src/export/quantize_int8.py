"""Static INT8 PTQ via onnxruntime. Re-check clDice after quantizing thin-vessel models."""
import argparse
def quantize(onnx_path, out=None):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out = out or onnx_path.replace(".onnx", ".int8.onnx")
    quantize_dynamic(onnx_path, out, weight_type=QuantType.QInt8)
    print("wrote", out, "(dynamic INT8; swap for static PTQ with a calibration set)")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--onnx", required=True); ap.add_argument("--out")
    a = ap.parse_args(); quantize(a.onnx, a.out)
