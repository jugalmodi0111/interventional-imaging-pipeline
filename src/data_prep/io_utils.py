"""Shared data-prep IO: standardized (img,msk) PNG pairs, nnU-Net raw layout, COCO->YOLO.

Keeps every converter emitting the SAME on-disk shapes so datasets are interchangeable:
  - segmentation student:  data/processed/<task>/{img,msk}/<stem>.png   (msk 0/255)
  - nnU-Net teacher:       $nnUNet_raw/Dataset0XX_<name>/{imagesTr,labelsTr} + dataset.json
  - YOLO detector:         data/processed/<task>/{images,labels}/<split>/<stem>.{png,txt}
"""
import glob, hashlib, json, os, re
import cv2, numpy as np
from src.data_prep.preprocess import clahe_unsharp

# Video-derived frames are near-identical between consecutive frames, so a per-frame split leaks
# the same sequence into train AND val. Collapse every frame of one source sequence to a single
# group so it lands entirely on one side (honest holdout):
#   Danilov    <site>_<patient>_<seq>_<frame>  (e.g. 14_002_5_0016)          -> <site>_<patient>
#   CathAction <clip>_img-<seg>-<frame>        (e.g. JFQ_j3383201_img-00000-0042) -> <clip>
_PATIENT_RE = re.compile(r"^(\d+_\d+)_\d+_\d+$")
_CLIP_RE = re.compile(r"^(.+?)_img-\d+-\d+$")


def group_key(name):
    """Split-group key: collapse a source sequence's frames to one key; else the name itself."""
    m = _PATIENT_RE.match(name)
    if m:
        return m.group(1)
    m = _CLIP_RE.match(name)
    if m:
        return m.group(1)
    return name


def ensure(*ds):
    for d in ds:
        os.makedirs(d, exist_ok=True)


def split_of(name, val_frac=0.15):
    """Deterministic, patient-grouped train/val split (stable across runs/processes).
    Hashes group_key(name) so all frames of a Danilov patient share a split (no frame leakage)."""
    h = int(hashlib.md5(group_key(name).encode()).hexdigest(), 16) % 1000
    return "val" if h < val_frac * 1000 else "train"


def write_pair(img_gray, mask, stem, out_dir, size=512, clahe=True):
    ensure(os.path.join(out_dir, "img"), os.path.join(out_dir, "msk"))
    im = clahe_unsharp(img_gray) if clahe else img_gray
    im = cv2.resize(im, (size, size))
    mk = cv2.resize((mask > 0).astype(np.uint8) * 255, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "img", stem + ".png"), im)
    cv2.imwrite(os.path.join(out_dir, "msk", stem + ".png"), mk)


def write_nnunet_case(img_gray, mask, stem, raw_dir, size=512, clahe=True):
    imtr, latr = os.path.join(raw_dir, "imagesTr"), os.path.join(raw_dir, "labelsTr")
    ensure(imtr, latr)
    im = clahe_unsharp(img_gray) if clahe else img_gray
    cv2.imwrite(os.path.join(imtr, f"{stem}_0000.png"), cv2.resize(im, (size, size)))
    mk = cv2.resize((mask > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(latr, f"{stem}.png"), mk)          # label values 0/1


def write_nnunet_datasetjson(raw_dir, n, channel="XCA"):
    json.dump({"channel_names": {"0": channel},
               "labels": {"background": 0, "vessel": 1},
               "numTraining": n, "file_ending": ".png"},
              open(os.path.join(raw_dir, "dataset.json"), "w"), indent=2)


def resolve_image(root, file_name):
    fn = os.path.basename(file_name)
    # fast paths: COCO json usually sits in .../annotations, images in a sibling .../images
    for cand in (os.path.join(root, file_name), os.path.join(root, fn),
                 os.path.join(root, "..", "images", fn), os.path.join(root, "images", fn)):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(root, "**", fn), recursive=True)   # fallback (slow)
    return hits[0] if hits else None


def find_coco_jsons(root):
    """Yield paths of every COCO-shaped json under root."""
    for jp in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        if isinstance(d, dict) and "images" in d and "annotations" in d:
            yield jp


def coco_seg_to_pairs(root, out_dir, size=512, raw_dir=None):
    """ARCADE-style: union all polygon anns per image -> binary vessel mask. Returns count."""
    from pycocotools.coco import COCO
    n = 0
    for jp in find_coco_jsons(root):
        coco = COCO(jp); base = os.path.dirname(jp)
        for img in coco.loadImgs(coco.getImgIds()):
            ip = resolve_image(base, img["file_name"]) or resolve_image(root, img["file_name"])
            if not ip:
                continue
            g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            m = np.zeros((img["height"], img["width"]), np.uint8)
            for a in coco.loadAnns(coco.getAnnIds(imgIds=img["id"])):
                try:
                    m = np.maximum(m, coco.annToMask(a))
                except Exception:
                    pass
            stem = os.path.splitext(os.path.basename(ip))[0]
            write_pair(g, m, stem, out_dir, size)
            if raw_dir:
                write_nnunet_case(g, m, stem, raw_dir, size)
            n += 1
    return n


def coco_to_yolo(root, out_dir, size=512, class_id=0, class_map=None):
    """COCO bbox -> YOLO txt. class_map: {coco_cat_id: yolo_idx} for multi-class; else class_id."""
    from pycocotools.coco import COCO
    n = 0
    for jp in find_coco_jsons(root):
        coco = COCO(jp); base = os.path.dirname(jp)
        for img in coco.loadImgs(coco.getImgIds()):
            ip = resolve_image(base, img["file_name"]) or resolve_image(root, img["file_name"])
            if not ip:
                continue
            W, H = img["width"], img["height"]
            lines = []
            for a in coco.loadAnns(coco.getAnnIds(imgIds=img["id"])):
                cid = class_map.get(a["category_id"]) if class_map else class_id
                if cid is None:
                    continue
                x, y, w, h = a["bbox"]
                lines.append(f"{cid} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                             f"{w / W:.6f} {h / H:.6f}")
            stem = os.path.splitext(os.path.basename(ip))[0]
            sp = split_of(stem)
            ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
            g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"),
                        cv2.resize(clahe_unsharp(g), (size, size)))
            open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
            n += 1
    return n


def masks_to_yolo(img_path, class_masks, out_dir, size=512):
    """class_masks: list of (yolo_idx, binary_mask). Emit one YOLO box per connected component."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0
    H, W = g.shape
    lines = []
    for idx, m in class_masks:
        num, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype("uint8"))
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 8:
                continue
            lines.append(f"{idx} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    if not lines:
        return 0
    sp = split_of(stem)
    ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
    cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"), cv2.resize(clahe_unsharp(g), (size, size)))
    open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
    return 1


def write_yolo_datayaml(out_dir, names=("stenosis",)):
    p = os.path.join(out_dir, "data.yaml")
    with open(p, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(names)}\nnames: {list(names)}\n")
    return p


def _split_stems(out_dir, split):
    """Set of image stems in a YOLO split's images/<split> dir (strip extension)."""
    d = os.path.join(out_dir, "images", split)
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")}


def audit_split_leakage(out_dir, danilov_stems=None, max_ungrouped_frac=0.5):
    """Post-conversion honesty gate for a YOLO train/val split. Returns a report dict;
    RAISES AssertionError the moment the split could leak. Call it AFTER conversion and
    BEFORE training so a leaked run can never silently report an inflated metric.

    Two independent failure modes are checked:

    1. Group leakage — no ``group_key`` (and no bare stem) may appear in BOTH train and val.
       This is the direct guard: near-identical consecutive video frames of one patient must
       land entirely on one side, or val F1 is inflated (the 2026-07-12 F1 0.885 signature).

    2. Silent grouping no-op — ``group_key`` only collapses Danilov frames whose names match
       ``<site>_<patient>_<seq>_<frame>``. If the real files are named differently the collapse
       silently does nothing and the split degrades back to per-frame. Pass ``danilov_stems``
       (the true set of Danilov image stems, e.g. from walking data/raw/danilov) so this can be
       detected *independently of the regex*: if more than ``max_ungrouped_frac`` of them are
       ungrouped (``group_key(stem) == stem``), the grouping is untrustworthy -> raise, because
       we cannot prove the split is honest.
    """
    train, val = _split_stems(out_dir, "train"), _split_stems(out_dir, "val")
    assert train and val, (
        f"empty split (train={len(train)}, val={len(val)}) — conversion produced no data")

    # (1a) exact-stem overlap: the same image file must never be in both splits.
    stem_overlap = train & val
    assert not stem_overlap, (
        f"LEAKAGE: {len(stem_overlap)} identical stems in BOTH train and val, "
        f"e.g. {sorted(stem_overlap)[:5]}")

    # (1b) group overlap: no patient/clip sequence may straddle the split.
    gtrain, gval = {group_key(s) for s in train}, {group_key(s) for s in val}
    group_overlap = gtrain & gval
    assert not group_overlap, (
        f"LEAKAGE: {len(group_overlap)} patient/clip groups span BOTH train and val, "
        f"e.g. {sorted(group_overlap)[:5]} — frames of one patient leaked across the split")

    # (2) prove the Danilov video frames were actually collapsed (not a silent regex no-op).
    danilov_report = None
    if danilov_stems is not None:
        dset = set(danilov_stems)
        d_in_split = (train | val) & dset
        ungrouped = {s for s in d_in_split if group_key(s) == s}
        frac = len(ungrouped) / max(1, len(d_in_split))
        danilov_report = {"danilov_frames": len(d_in_split),
                          "ungrouped": len(ungrouped), "ungrouped_frac": round(frac, 3),
                          "patient_groups": len({group_key(s) for s in d_in_split})}
        assert d_in_split and frac <= max_ungrouped_frac, (
            f"UNGROUPED DANILOV: {len(ungrouped)}/{len(d_in_split)} "
            f"({frac:.0%}) Danilov frames were NOT collapsed by group_key — their filenames do "
            f"not match the assumed '<site>_<patient>_<seq>_<frame>' pattern, so the split is "
            f"per-frame and WILL leak. Inspect the names and update group_key()/_PATIENT_RE "
            f"before trusting any F1. Example ungrouped: {sorted(ungrouped)[:5]}")

    return {"train_imgs": len(train), "val_imgs": len(val),
            "train_groups": len(gtrain), "val_groups": len(gval),
            "val_frac_by_group": round(len(gval) / max(1, len(gtrain) + len(gval)), 3),
            "danilov": danilov_report}
