"""Mac-side export: rebuild the TinyU-Net student -> CoreML (.mlpackage) + weight compression.

Run on macOS (Apple silicon). Conversion itself is data-free (palettize/linear weight quant).
ALWAYS re-check clDice afterwards with `src/export/coreml_validate.py` — compression breaks
thin vessels even when Dice looks fine.
"""
import argparse
from src.models.seg_student import load_student

_CU = {"all": "ALL", "cpuAndNeuralEngine": "CPU_AND_NE", "cpuOnly": "CPU_ONLY"}


def export(weights, out=None, shape=(1, 1, 512, 512), method="palettize", nbits=6,
           base=16, depth=4, compute_units="all", deploy="macOS13"):
    import torch, coremltools as ct
    out = out or weights.replace(".pt", ".mlpackage").replace(".onnx", ".mlpackage")
    m = load_student(weights, base=base, depth=depth)
    ts = torch.jit.trace(m, torch.randn(*shape))
    mlmodel = ct.convert(
        ts,
        inputs=[ct.TensorType(name="input", shape=shape)],
        compute_units=getattr(ct.ComputeUnit, _CU[compute_units]),
        minimum_deployment_target=getattr(ct.target, deploy),
        convert_to="mlprogram",                       # required for coremltools.optimize
    )
    if method and method != "none":
        import coremltools.optimize.coreml as cto
        if method == "palettize":                     # data-free k-means LUT (nbits)
            cfg = cto.OptimizationConfig(
                global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=nbits))
            mlmodel = cto.palettize_weights(mlmodel, cfg)
        elif method == "linear":                      # int8 symmetric weight quant
            cfg = cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
            mlmodel = cto.linear_quantize_weights(mlmodel, cfg)
        else:
            raise ValueError(f"unknown method {method!r}")
    mlmodel.save(out)
    print("wrote", out, f"({method} nbits={nbits})" if method and method != "none" else "(fp16)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="student state_dict .pt from the Colab build")
    ap.add_argument("--out")
    ap.add_argument("--method", default="palettize", choices=["palettize", "linear", "none"])
    ap.add_argument("--nbits", type=int, default=6)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()
    export(a.weights, a.out, method=a.method, nbits=a.nbits, base=a.base, depth=a.depth)
