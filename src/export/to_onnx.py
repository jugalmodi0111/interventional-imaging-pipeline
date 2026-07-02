"""Export a torch student to ONNX (dynamic batch). Rebuilds TinyU-Net from a state_dict."""
import argparse, torch
from src.models.seg_student import load_student
def export(weights, out=None, shape=(1, 1, 512, 512), base=16, depth=4):
    out = out or weights.replace(".pt", ".onnx")
    model = load_student(weights, base=base, depth=depth)
    torch.onnx.export(model, torch.randn(*shape), out, opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}})
    print("wrote", out)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--out")
    ap.add_argument("--base", type=int, default=16); ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args(); export(a.weights, a.out, base=a.base, depth=a.depth)
