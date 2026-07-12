"""Train YOLO11n stenosis detector (+ optional pseudo-label SSL). Importable; run on GPU.

Library entrypoint `train(cfg)` — call it from the Colab/Kaggle notebook. Heavy (GPU); do not
run on a laptop CPU. Saves best weights to runs/stenosis/ and reports F1/mAP.

SSL cold-start: set `ssl.seed: gdino` to seed round-0 pseudo-labels with an open-vocabulary
Grounding DINO pass on the unlabeled frames (a better cold start than a YOLO trained on scarce
labels), then the usual pseudo-label self-training refines. GD deps (torch/transformers) load
only inside `_gdino_seed_round`; this module imports fine without them.
"""
import argparse, glob, os, yaml

from src.data_prep.autolabel_gdino import DEFAULT_PROMPTS, dino_boxes_to_yolo_lines


def _detector(cfg):
    return cfg.get("model") or cfg.get("detector")           # stenosis uses 'model', catheter 'detector'


def _data_yaml(cfg):
    task = cfg.get("task", "")
    sub = "catheter" if "catheter" in task else "stenosis"
    return f"data/processed/{sub}/data.yaml"


def ssl_seed(cfg):
    """The SSL cold-start seed, e.g. 'gdino', or None (from cfg['ssl']['seed'])."""
    return cfg.get("ssl", {}).get("seed")


def seed_prompt_and_classes(cfg):
    """(open-vocab text prompt, {label_string: yolo_class_id}) for the detector's task.

    Catheter pulls its class names from the config; stenosis is single-class. Prompt words must
    match the class names so Grounding DINO labels map cleanly onto YOLO ids.
    """
    task = cfg.get("task", "")
    if "catheter" in task:
        names = cfg.get("datasets", {}).get("cathaction", {}).get("classes", ["catheter", "guidewire"])
        return DEFAULT_PROMPTS["catheter"], {n: i for i, n in enumerate(names)}
    return DEFAULT_PROMPTS["stenosis"], {"stenosis": 0}


def boxes_labels_to_yolo_lines(boxes_xyxy, labels, class_map, W, H):
    """Pixel boxes + label strings -> YOLO lines, mapping labels via class_map (unknown skipped)."""
    lines = []
    for box, lab in zip(boxes_xyxy, labels):
        cid = class_map.get(lab)
        if cid is not None:
            lines += dino_boxes_to_yolo_lines([box], W, H, cid)
    return lines


def run_tag(cfg):
    """Stable run name encoding data+model+imgsz+epochs, e.g. 'arcade_yolo11n_640_e150'.
    Use it as the ultralytics run folder so different configs don't clobber each other's outputs."""
    m = _detector(cfg) or {}
    names = [k.replace("_stenosis", "").replace("_detection", "") for k in cfg.get("datasets", {})]
    data = "+".join(sorted(names)) or "data"
    return f"{data}_{m.get('name', 'yolo')}_{m.get('imgsz', 640)}_e{cfg.get('train', {}).get('epochs', 100)}"


def train_kwargs(cfg):
    """Ultralytics train() kwargs incl. speed knobs. Fast, quality-neutral defaults:
    cache (RAM/disk dataset cache), amp (mixed precision), and patience (early-stop once the
    val metric plateaus — keeps the same best.pt, skips wasted epochs)."""
    m, tr = _detector(cfg) or {}, cfg.get("train", {})
    return {"imgsz": m.get("imgsz", 640),
            "epochs": tr.get("epochs", 100), "batch": tr.get("batch", 16),
            "lr0": tr.get("lr", 1e-3),
            "cache": tr.get("cache", True), "workers": tr.get("workers", 8),
            "patience": tr.get("patience", 30), "amp": tr.get("amp", True)}


def train(cfg, project=None, data_yaml=None, device=0):
    # device=0 forces GPU (fails loud if none); pass 'cpu' to override.
    from ultralytics import YOLO
    m = _detector(cfg)
    data_yaml = data_yaml or _data_yaml(cfg)
    project = project or f"runs/{'catheter' if 'catheter' in cfg.get('task','') else 'stenosis'}"
    tk = train_kwargs(cfg)                                    # imgsz/epochs/batch/lr0 + cache/workers/patience/amp
    model = YOLO(m["name"] + ".pt")
    model.train(data=data_yaml, project=project, name="base", exist_ok=True, device=device, **tk)
    best = os.path.join(project, "base", "weights", "best.pt")

    if ssl_seed(cfg) == "gdino":                              # open-vocab cold start before self-training
        best = _gdino_seed_round(cfg, project, data_yaml, device=device)

    if cfg.get("ssl", {}).get("pseudo_label"):
        best = _pseudo_label_round(best, cfg, project, data_yaml, device=device)

    val = YOLO(best).val(data=data_yaml, device=device)
    print("best:", best, "| mAP50:", round(float(val.box.map50), 4))
    return best


def _gdino_seed_round(cfg, project, data_yaml, unlabeled_dir=None, device=0):
    """Round-0 cold start: Grounding DINO open-vocab labels the unlabeled frames -> YOLO labels in
    the train split, then retrain from scratch. Heavy: GD (torch/transformers) loads lazily here."""
    import cv2
    from ultralytics import YOLO
    from src.data_prep import io_utils as io
    from src.data_prep.autolabel_gdino import detect, filter_detections
    ssl = cfg.get("ssl", {})
    conf = ssl.get("conf", 0.4)
    size = (_detector(cfg) or {}).get("imgsz", 640)
    prompt, class_map = seed_prompt_and_classes(cfg)
    unlabeled_dir = unlabeled_dir or ssl.get("unlabeled_dir", "data/raw/xcad")
    imgs = glob.glob(os.path.join(unlabeled_dir, "**", "*.png"), recursive=True)
    if not imgs:
        print("GD-seed: no unlabeled frames found; skipping")
        return os.path.join(project, "base", "weights", "best.pt")
    proc = os.path.dirname(data_yaml)
    out_i, out_l = os.path.join(proc, "images/train"), os.path.join(proc, "labels/train")
    os.makedirs(out_i, exist_ok=True); os.makedirs(out_l, exist_ok=True)
    kept = 0
    for ip in imgs:
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H, W = g.shape
        boxes, scores, labels = detect(g, prompt, box_thresh=conf, device=device)
        boxes, scores, labels = filter_detections(boxes, scores, labels, conf)
        lines = boxes_labels_to_yolo_lines(boxes, labels, class_map, W, H)
        if not lines:
            continue
        stem = "gd_" + os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(out_i, stem + ".png"),           # CLAHE+resize to match base/inference
                    cv2.resize(io.clahe_unsharp(g), (size, size)))
        open(os.path.join(out_l, stem + ".txt"), "w").write("\n".join(lines))
        kept += 1
    print(f"GD-seed: added {kept} Grounding-DINO-labeled frames; retraining")
    model = YOLO(_detector(cfg)["name"] + ".pt")
    model.train(data=data_yaml, project=project, name="gdino", exist_ok=True, device=device,
                **train_kwargs(cfg))
    return os.path.join(project, "gdino", "weights", "best.pt")


def _pseudo_label_round(weights, cfg, project, data_yaml, unlabeled_dir=None, device=0):
    """Predict on unlabeled frames >= conf, write YOLO pseudo-labels into the train split, retrain.
    Frames are CLAHE+resized before predict AND on disk so the pseudo-labels match the base-train and
    inference preprocessing; the predicted class id is kept (not forced to 0) for multi-class tasks."""
    import cv2
    from ultralytics import YOLO
    from src.data_prep import io_utils as io
    conf = cfg["ssl"].get("conf", 0.4)
    size = (_detector(cfg) or {}).get("imgsz", 640)
    unlabeled_dir = unlabeled_dir or cfg.get("ssl", {}).get("unlabeled_dir", "data/raw/xcad")
    imgs = glob.glob(os.path.join(unlabeled_dir, "**", "*.png"), recursive=True)
    if not imgs:
        print("SSL: no unlabeled frames found; skipping")
        return weights
    model = YOLO(weights)
    proc = os.path.dirname(data_yaml)
    out_i = os.path.join(proc, "images/train")
    out_l = os.path.join(proc, "labels/train")
    os.makedirs(out_i, exist_ok=True); os.makedirs(out_l, exist_ok=True)
    kept = 0
    for ip in imgs:
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        frame = cv2.resize(io.clahe_unsharp(g), (size, size))    # match base/inference preprocessing
        b = model.predict(frame, conf=conf, verbose=False, device=device)[0].boxes
        if b is None or len(b) == 0:
            continue
        stem = "pl_" + os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(out_i, stem + ".png"), frame)
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in zip(b.cls.tolist(), b.xywhn.tolist())]
        open(os.path.join(out_l, stem + ".txt"), "w").write("\n".join(lines))
        kept += 1
    print(f"SSL: added {kept} pseudo-labeled frames; retraining")
    model = YOLO(_detector(cfg)["name"] + ".pt")
    model.train(data=data_yaml, project=project, name="ssl", exist_ok=True, device=device,
                **train_kwargs(cfg))
    return os.path.join(project, "ssl", "weights", "best.pt")


def main(cfg):
    return train(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
