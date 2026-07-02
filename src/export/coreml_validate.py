"""HARD GATE — re-check Dice/clDice of the compressed CoreML model vs the fp32 torch student.

Palettization / INT8 tends to break thin vessels: Dice can hold while clDice (connectivity)
collapses. Run on macOS against a small paired image/mask val set (PNG: matching filenames).

    python -m src.export.coreml_validate \
        --coreml runs/coronary/student.mlpackage --weights runs/coronary/student.pt \
        --images data/processed/coronary/val/img --masks data/processed/coronary/val/msk
"""
import argparse, glob, os
import numpy as np


def _load_pairs(images, masks, size=512, limit=50):
    import cv2
    xs, ys = [], []
    for ip in sorted(glob.glob(os.path.join(images, "*")))[:limit]:
        mp = os.path.join(masks, os.path.basename(ip))
        if not os.path.exists(mp):
            continue
        im = cv2.resize(cv2.imread(ip, cv2.IMREAD_GRAYSCALE), (size, size))
        gt = cv2.resize(cv2.imread(mp, cv2.IMREAD_GRAYSCALE), (size, size))
        xs.append((im.astype(np.float32) / 255.0))
        ys.append((gt > 127).astype(np.uint8))
    return xs, ys


def _coreml_pred(model, x):
    name = model.get_spec().description.input[0].name
    out = model.predict({name: x[None, None].astype(np.float32)})
    logits = np.asarray(list(out.values())[0])
    return (1.0 / (1.0 + np.exp(-logits))).squeeze() >= 0.5


def _torch_pred(m, x):
    import torch
    with torch.no_grad():
        logits = m(torch.from_numpy(x)[None, None])
    return torch.sigmoid(logits).numpy().squeeze() >= 0.5


def main(a):
    import coremltools as ct
    from src.models.seg_student import load_student
    from src.eval.metrics import dice, cldice

    xs, ys = _load_pairs(a.images, a.masks, size=a.size, limit=a.limit)
    assert xs, f"no paired image/mask found under {a.images} / {a.masks}"
    cm = ct.models.MLModel(a.coreml)
    tm = load_student(a.weights, base=a.base, depth=a.depth)

    dc_t = dc_c = cl_t = cl_c = 0.0
    for x, gt in zip(xs, ys):
        pt, pc = _torch_pred(tm, x), _coreml_pred(cm, x)
        dc_t += dice(pt, gt); dc_c += dice(pc, gt)
        cl_t += cldice(pt, gt); cl_c += cldice(pc, gt)
    n = len(xs)
    dc_t, dc_c, cl_t, cl_c = dc_t / n, dc_c / n, cl_t / n, cl_c / n
    drop = cl_t - cl_c
    print(f"n={n}")
    print(f"fp32    Dice {dc_t:.4f}  clDice {cl_t:.4f}")
    print(f"coreml  Dice {dc_c:.4f}  clDice {cl_c:.4f}")
    print(f"clDice drop {drop:+.4f}  gate(<= {a.gate})  ->  {'PASS' if drop <= a.gate else 'FAIL'}")
    return drop <= a.gate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coreml", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--gate", type=float, default=0.03)
    main(ap.parse_args())
