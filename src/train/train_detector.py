"""Train YOLO11n stenosis detector (+ optional pseudo-label SSL). Importable; run on GPU.

Library entrypoint `train(cfg)` — call it from the Colab/Kaggle notebook. Heavy (GPU); do not
run on a laptop CPU. Saves best weights to runs/stenosis/ and reports F1/mAP.
"""
import argparse, glob, os, yaml


def _detector(cfg):
    return cfg.get("model") or cfg.get("detector")           # stenosis uses 'model', catheter 'detector'


def _data_yaml(cfg):
    task = cfg.get("task", "")
    sub = "catheter" if "catheter" in task else "stenosis"
    return f"data/processed/{sub}/data.yaml"


def train(cfg, project=None, data_yaml=None, device=0):
    # device=0 forces GPU (fails loud if none); pass 'cpu' to override.
    from ultralytics import YOLO
    m = _detector(cfg)
    tr = cfg.get("train", {})
    data_yaml = data_yaml or _data_yaml(cfg)
    project = project or f"runs/{'catheter' if 'catheter' in cfg.get('task','') else 'stenosis'}"
    model = YOLO(m["name"] + ".pt")
    model.train(data=data_yaml, imgsz=m.get("imgsz", 640),
                epochs=tr.get("epochs", 100), batch=tr.get("batch", 16),
                lr0=tr.get("lr", 1e-3), project=project, name="base", exist_ok=True, device=device)
    best = os.path.join(project, "base", "weights", "best.pt")

    if cfg.get("ssl", {}).get("pseudo_label"):
        best = _pseudo_label_round(best, cfg, project, data_yaml, device=device)

    val = YOLO(best).val(data=data_yaml, device=device)
    print("best:", best, "| mAP50:", round(float(val.box.map50), 4))
    return best


def _pseudo_label_round(weights, cfg, project, data_yaml, unlabeled_dir="data/raw/xcad", device=0):
    """Predict on unlabeled frames >= conf, write YOLO pseudo-labels into the train split, retrain."""
    from ultralytics import YOLO
    conf = cfg["ssl"].get("conf", 0.4)
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
    for res in model.predict(imgs, conf=conf, stream=True, verbose=False, device=device):
        b = res.boxes
        if b is None or len(b) == 0:
            continue
        stem = "pl_" + os.path.splitext(os.path.basename(res.path))[0]
        import shutil
        shutil.copy(res.path, os.path.join(out_i, stem + ".png"))
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in b.xywhn.tolist()]
        open(os.path.join(out_l, stem + ".txt"), "w").write("\n".join(lines))
        kept += 1
    print(f"SSL: added {kept} pseudo-labeled frames; retraining")
    m, tr = _detector(cfg), cfg.get("train", {})
    model = YOLO(m["name"] + ".pt")
    model.train(data=data_yaml, imgsz=m.get("imgsz", 640), epochs=tr.get("epochs", 100),
                batch=tr.get("batch", 16), lr0=tr.get("lr", 1e-3),
                project=project, name="ssl", exist_ok=True, device=device)
    return os.path.join(project, "ssl", "weights", "best.pt")


def main(cfg):
    return train(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
