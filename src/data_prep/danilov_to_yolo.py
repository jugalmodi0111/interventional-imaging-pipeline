"""Stenosis detection prep -> YOLO format. Merges ARCADE task-2 (COCO) + Danilov boxes.

ARCADE task-2 is COCO -> auto-converted. Danilov (Mendeley ydrm75xywg) ships boxes in its own
layout; if it exports a COCO json it is picked up automatically, otherwise wire its native
annotation reader in `_danilov_native` below. Writes YOLO images/labels + data.yaml.
"""
import argparse, glob, os, xml.etree.ElementTree as ET, yaml
import cv2
from src.data_prep import io_utils as io

OUT = "data/processed/stenosis"

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")   # Danilov ships .bmp


def _index_images(root):
    """One os.walk -> {basename: path, stem: path} for O(1) image lookup.
    Avoids a per-annotation recursive glob (O(n*files)) that stalls on Danilov's flat dataset/ dir."""
    idx = {}
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in _IMG_EXTS:
                full = os.path.join(dp, f)
                idx.setdefault(f, full)
                idx.setdefault(os.path.splitext(f)[0], full)
    return idx


def _voc_box(obj, W, H):
    b = obj.find("bndbox")
    x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
    x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
    return f"0 {((x1 + x2) / 2) / W:.6f} {((y1 + y2) / 2) / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def _danilov_native(root, out_dir, size):
    """Danilov (Mendeley ydrm75xywg) ships boxes as Pascal-VOC XML or YOLO .txt. Map all
    severity classes (small/medium/large) -> single 'stenosis' class 0. Returns count."""
    n = 0
    imgidx = _index_images(root)   # build once; O(1) lookups below
    # (1) Pascal-VOC XML
    for xp in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        try:
            t = ET.parse(xp).getroot()
        except Exception:
            continue
        fn = t.findtext("filename")
        ip = (imgidx.get(os.path.basename(fn)) if fn else None) or \
             imgidx.get(os.path.splitext(os.path.basename(xp))[0])
        if not ip:
            continue
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H, W = g.shape
        lines = [_voc_box(o, W, H) for o in t.findall("object") if o.find("bndbox") is not None]
        stem = os.path.splitext(os.path.basename(ip))[0]
        sp = io.split_of(stem)
        io.ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
        cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"),
                    cv2.resize(io.clahe_unsharp(g), (size, size)))
        open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
        n += 1
    if n:
        return n
    # (2) already-YOLO: images with sibling .txt (force class 0, CLAHE the images)
    for tp in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        if os.path.basename(tp) in ("classes.txt", "data.txt"):
            continue
        stem = os.path.splitext(os.path.basename(tp))[0]
        ip = imgidx.get(stem)
        if not ip:
            continue
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        lines = [("0 " + " ".join(ln.split()[1:])) for ln in open(tp) if ln.strip()]
        sp = io.split_of(stem)
        io.ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
        cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"),
                    cv2.resize(io.clahe_unsharp(g), (size, size)))
        open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
        n += 1
    return n


def _danilov_stems(root):
    """True set of Danilov frame stems from raw — used to prove group_key actually collapsed them."""
    stems = set()
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in _IMG_EXTS:
                stems.add(os.path.splitext(f)[0])
    return stems


def main(cfg):
    ds = cfg["datasets"]
    size = cfg.get("model", {}).get("imgsz", 640)
    total = 0
    for key in ("arcade_stenosis", "danilov"):
        d = ds.get(key)
        if not d:
            continue
        dupes = io.duplicate_basenames_across_cocos(d["root"])       # ARCADE renumbers per split
        if dupes:
            print(f"[info] {key}: {len(dupes)} basenames repeat across COCO splits "
                  f"-> disambiguated by split prefix in coco_to_yolo (no data loss). "
                  f"e.g. {list(dupes)[:5]}")
        c = io.coco_to_yolo(d["root"], OUT, size=size, class_id=0)   # COCO path (both if present)
        if c == 0 and key == "danilov":
            c = _danilov_native(d["root"], OUT, size)
        print(f"{key}: {c} images")
        total += c
    if total == 0:
        raise SystemExit(f"No stenosis boxes converted. Check {[ds[k]['root'] for k in ds]}.")
    yml = io.write_yolo_datayaml(OUT, names=("stenosis",))
    # Honesty gate: RAISE before anyone trains on a split that could leak (patient/clip in both
    # train+val, or Danilov frames that group_key failed to collapse -> silent per-frame split).
    dstems = _danilov_stems(ds["danilov"]["root"]) if ds.get("danilov") else None
    rep = io.audit_split_leakage(OUT, danilov_stems=dstems)
    print(f"stenosis -> {OUT} : {total} images ; data cfg {yml} ; leakage-audit OK "
          f"(train {rep['train_imgs']}/{rep['train_groups']}g, val {rep['val_imgs']}/{rep['val_groups']}g)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
