"""CathAction -> YOLO (2 classes: catheter, guidewire). Feeds the catheter/guidewire detector.

CathAction ships per-frame instance masks (catheter + guidewire labelled separately) and, in some
releases, COCO json. We take the COCO path when present; otherwise derive boxes from per-class masks.
Confirm your CathAction download layout and adjust `_mask_dirs` if the folder names differ.
"""
import argparse, glob, os, yaml
import cv2
from src.data_prep import io_utils as io

OUT = "data/processed/catheter"
NAMES = ("catheter", "guidewire")


def _mask_dirs(root):
    """Return {yolo_idx: mask_dir} by matching class names in folder paths."""
    dirs = {}
    for idx, name in enumerate(NAMES):
        hit = glob.glob(os.path.join(root, "**", f"*{name}*"), recursive=True)
        hit = [h for h in hit if os.path.isdir(h)]
        if hit:
            dirs[idx] = hit[0]
    return dirs


def _from_masks(root, size):
    dirs = _mask_dirs(root)
    if not dirs:
        return 0
    # index masks by frame stem across classes
    frames = {}
    for idx, d in dirs.items():
        for mp in glob.glob(os.path.join(d, "*")):
            stem = os.path.splitext(os.path.basename(mp))[0]
            frames.setdefault(stem, []).append((idx, mp))
    n = 0
    for stem, items in frames.items():
        ip = io.resolve_image(root, stem + ".png") or io.resolve_image(root, stem + ".jpg")
        if not ip:
            continue
        class_masks = [(idx, cv2.imread(mp, cv2.IMREAD_GRAYSCALE)) for idx, mp in items]
        class_masks = [(i, m) for i, m in class_masks if m is not None]
        n += io.masks_to_yolo(ip, class_masks, OUT, size)
    return n


def main(cfg):
    root = cfg["datasets"]["cathaction"]["root"]
    size = cfg.get("detector", {}).get("imgsz", 640)
    cmap = {i + 1: i for i in range(len(NAMES))}                 # COCO cat ids 1..N -> 0..N-1
    n = io.coco_to_yolo(root, OUT, size=size, class_map=cmap)    # COCO path if present
    if n == 0:
        n = _from_masks(root, size)                             # mask fallback
    if n == 0:
        raise SystemExit(f"No CathAction annotations converted under {root!r}. Check layout.")
    yml = io.write_yolo_datayaml(OUT, names=NAMES)
    print(f"CathAction -> {OUT} : {n} frames ; data cfg {yml}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
