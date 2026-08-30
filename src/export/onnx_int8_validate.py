"""HARD GATE — re-check Dice/clDice of the INT8 ONNX model vs the fp32 ONNX it came from.

The non-Apple edge target (Intel laptop / Jetson) ships the ONNX+INT8 artifact, and INT8
breaks thin vessels the same way palettization does: Dice can hold while clDice (connectivity)
collapses. configs/edge_export.yaml has always declared `gate: {cldice_drop_max: 0.03}`;
this script is what reads and enforces it — the CoreML twin of it is coreml_validate.py.

    python -m src.export.onnx_int8_validate \
        --fp32 outputs/coronary_student_clgeodice/student.onnx \
        --int8 outputs/coronary_student_clgeodice/student.int8.static.onnx \
        --images data/processed/coronary/val/img --masks data/processed/coronary/val/msk

EXIT STATUS IS THE GATE: cli() exits 0 on PASS and 1 on FAIL so a Makefile/CI step stops on
a regression. The gate value comes from configs/edge_export.yaml's gate.cldice_drop_max
(--gate overrides it); a config with no such key is REFUSED rather than defaulted, because
an unconfigured gate that silently passes is exactly the failure mode this file exists to end.

Scored on whatever image/mask set you point it at — it inherits, and cannot improve on, the
held-out-ness of that set. Point it at held-out stems if you want a held-out number.
"""
import argparse, os, sys
import numpy as np

from src.export.coreml_validate import _load_pairs      # same tensor prep as the CoreML gate

_CONFIG = "configs/edge_export.yaml"


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _onnx_pred(sess, x):
    name = sess.get_inputs()[0].name
    logits = np.asarray(sess.run(None, {name: x[None, None].astype(np.float32)})[0])
    return (1.0 / (1.0 + np.exp(-logits))).squeeze() >= 0.5


# separate names so a test (or a debug session) can stub one side without stubbing both
def _fp32_pred(sess, x):
    return _onnx_pred(sess, x)


def _int8_pred(sess, x):
    return _onnx_pred(sess, x)


def gate_from_config(config=_CONFIG):
    """Read `gate.cldice_drop_max` from configs/edge_export.yaml. Missing file or missing
    key is an AssertionError, not a default: a gate nobody configured must not report PASS."""
    import yaml
    assert config and os.path.exists(config), f"gate config not found: {config}"
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}
    g = (cfg.get("gate") or {}).get("cldice_drop_max")
    assert g is not None, (f"{config} declares no gate.cldice_drop_max — refusing to run an "
                           "unconfigured HARD gate (pass --gate to override deliberately)")
    return float(g)


def resolve_gate(a):
    return float(a.gate) if a.gate is not None else gate_from_config(a.config)


def main(a):
    """Returns the bool verdict (True = PASS). Use cli() for a process exit status."""
    from src.eval.metrics import dice, cldice

    gate = resolve_gate(a)
    xs, ys = _load_pairs(a.images, a.masks, size=a.size, limit=a.limit)
    assert xs, f"no paired image/mask found under {a.images} / {a.masks}"
    s32, s8 = _session(a.fp32), _session(a.int8)

    dc_t = dc_q = cl_t = cl_q = agree = 0.0
    n = 0
    for x, gt in zip(xs, ys):
        p32, p8 = _fp32_pred(s32, x), _int8_pred(s8, x)
        d_t = dice(p32, gt)
        if d_t != d_t:                       # NaN -> empty-GT frame (dice & cldice both NaN); exclude
            continue
        dc_t += d_t; dc_q += dice(p8, gt)
        cl_t += cldice(p32, gt); cl_q += cldice(p8, gt)
        agree += float((np.asarray(p32).astype(bool) == np.asarray(p8).astype(bool)).mean())
        n += 1
    assert n, "every frame had an empty GT mask — nothing to score"
    dc_t, dc_q, cl_t, cl_q, agree = dc_t / n, dc_q / n, cl_t / n, cl_q / n, agree / n
    drop = cl_t - cl_q
    ok = bool(drop <= gate)
    print(f"n={n}")
    print(f"fp32    Dice {dc_t:.4f}  clDice {cl_t:.4f}")
    print(f"int8    Dice {dc_q:.4f}  clDice {cl_q:.4f}")
    print(f"mask agreement {agree:.4f}")
    print(f"clDice drop {drop:+.4f}  gate(<= {gate})  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True, help="fp32 .onnx from to_onnx.py")
    ap.add_argument("--int8", required=True, help="quantized .onnx from quantize_int8.py")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--config", default=_CONFIG, help="source of gate.cldice_drop_max")
    ap.add_argument("--gate", type=float, default=None,
                    help="override the configured gate.cldice_drop_max")
    return ap


def cli(argv=None):
    """HARD gate entry point: 0 when the clDice drop is within gate, 1 when it is not."""
    return 0 if main(_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    sys.exit(cli())
