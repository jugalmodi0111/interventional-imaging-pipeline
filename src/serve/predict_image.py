"""Single-image coronary demo: upload an angiogram -> mark the blockage (stenosis).

- Stenosis detector (YOLO) draws a RED box on each narrowing = "blockage here" + confidence.
- Vessel segmentation (optional TinyU-Net student) tints the vessel tree GREEN for context.

    python -m src.serve.predict_image --image frame.png \
        --detector runs/stenosis/base/weights/best.pt \
        --seg runs/coronary/student.pt --out annotated.png

Note: coronary angiogram = ARTERIES only (veins aren't contrast-filled), so this marks arterial
blood blockage. AVF/vein blockage needs a different scan + data (not this model).
"""
import argparse, os
import numpy as np, cv2
from src.data_prep.preprocess import clahe_unsharp


def _vessel_mask(seg_weights, gray, size=512, base=16, depth=4):
    import torch
    from src.models.seg_student import load_student
    m = load_student(seg_weights, base=base, depth=depth)
    x = cv2.resize(clahe_unsharp(gray), (size, size)).astype(np.float32) / 255.0
    with torch.no_grad():
        p = torch.sigmoid(m(torch.from_numpy(x)[None, None])).numpy().squeeze()
    return cv2.resize((p >= 0.5).astype(np.uint8), (gray.shape[1], gray.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def predict(image, detector, seg=None, out=None, conf=0.25, imgsz=640):
    from ultralytics import YOLO
    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    assert gray is not None, f"cannot read image: {image}"
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if seg:                                                    # green vessel overlay (context)
        vm = _vessel_mask(seg, gray)
        vis[vm == 1] = (0.5 * vis[vm == 1] + np.array([0, 150, 0])).astype(np.uint8)

    clahe_bgr = cv2.cvtColor(clahe_unsharp(gray), cv2.COLOR_GRAY2BGR)   # detector trained on CLAHE
    res = YOLO(detector).predict(clahe_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    n = 0
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist()); s = float(b.conf[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"blockage {s:.0%}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        n += 1

    banner = f"{n} blockage(s) detected" if n else "no blockage detected"
    cv2.putText(vis, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    out = out or os.path.splitext(image)[0] + "_annotated.png"
    cv2.imwrite(out, vis)
    print(f"{banner}  ->  {out}")
    return {"out": out, "n_blockages": n, "vis": vis}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--detector", required=True, help="stenosis YOLO best.pt")
    ap.add_argument("--seg", help="optional coronary vessel-seg student .pt")
    ap.add_argument("--out")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    predict(a.image, a.detector, seg=a.seg, out=a.out, conf=a.conf)
