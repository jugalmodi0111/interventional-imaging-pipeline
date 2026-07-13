"""CADICA (2024) -> YOLO single class 'stenosis'. Patient-diverse coronary angiography w/ lesion boxes.

CADICA layout (verified against Mendeley p9bpx9ctcv):
    selectedVideos/pX/vY/input/<frame>.png            # ALL angiography frames (10 fps)
    selectedVideos/pX/vY/groundtruth/<frame>.txt      # one lesion box per line; keyframes only,
                                                      # and ONLY for lesion videos
NB the GT dir is lowercase ``groundtruth`` on Mendeley (some mirrors camel-case it) and Kaggle's
filesystem is case-SENSITIVE, so the sibling lookup below is case-insensitive.

Each groundtruth line is ``x y w h [label]`` where (x, y) is the ABSOLUTE-pixel TOP-LEFT corner,
(w, h) the box size in pixels, and the optional trailing ``label`` a severity string (e.g. ``p20_50``);
every severity collapses to the single YOLO class 0 ('stenosis'), and a 4-field line (no label) also
works. Patients are p1..p42; the split GROUP KEY is the PATIENT (``pX``) so all of a patient's frames
land on one side of the train/val split (no frame leakage), mirroring the Danilov converter. Only
keyframes carry a groundtruth .txt, so a frame is converted iff it has one (annotation-driven).

Heavy deps (cv2) are imported lazily inside the conversion functions, so the pure helpers below
(`cadica_boxes_to_yolo_lines`, `parse_cadica_gt`, `cadica_patient_of`) can be imported and
unit-tested without cv2 installed. Writes to the SHARED stenosis OUT dir like Danilov.
"""
import argparse, glob, os, re, yaml

OUT = "data/processed/stenosis"
NAMES = ("stenosis",)
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-testable without cv2)
# --------------------------------------------------------------------------------------------------
def cadica_boxes_to_yolo_lines(boxes_xywh_abs, W, H):
    """CADICA GT boxes (absolute top-left x,y,w,h in px) -> YOLO lines, all class 0 'stenosis'.

    Each box is normalized by image (``W``, ``H``) and converted from top-left to CENTER form. Every
    CADICA severity label collapses to the single class 0. Returns ``["0 cx cy w h", ...]`` (6-dp),
    one line per box (empty list -> a negative/background frame with an empty label file).
    """
    lines = []
    for box in boxes_xywh_abs:
        x, y, w, h = (float(v) for v in box[:4])
        lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    return lines


def parse_cadica_gt(text):
    """Parse a CADICA groundTruth file body -> ``[(x, y, w, h), ...]`` floats (label dropped).

    Each nonblank line is ``x y w h label``; lines with fewer than 4 leading numeric fields are
    skipped (robust to blank lines / malformed rows)."""
    boxes = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            x, y, w, h = (float(p) for p in parts[:4])
        except ValueError:
            continue
        boxes.append((x, y, w, h))
    return boxes


def cadica_patient_of(path):
    """Return the CADICA patient id (``pXX``) found in ``path``, else None.

    The patient is the train/val split GROUP KEY (all of ``pXX``'s frames land on one side)."""
    for part in os.path.normpath(path).split(os.sep):
        if re.fullmatch(r"p\d+", part):
            return part
    return None


def _sibling_dir(parent, name):
    """Case-insensitive lookup of subdir ``name`` under ``parent``. CADICA ships lowercase
    ``groundtruth`` but mirrors camel-case it, and Kaggle's filesystem is case-SENSITIVE, so a
    hardcoded name silently finds nothing there. Returns the matching path or None."""
    low = name.lower()
    try:
        for e in sorted(os.listdir(parent)):
            if e.lower() == low and os.path.isdir(os.path.join(parent, e)):
                return os.path.join(parent, e)
    except OSError:
        pass
    return None


def _out_stem(patient, video, frame_stem):
    """Globally-unique output stem so frames from different patients/videos never clobber one path.

    Real CADICA frames are already named ``pXX_vYY_NNNNN``; if the frame stem doesn't already start
    with the patient id, prefix ``<patient>_<video>_``."""
    if patient and frame_stem.startswith(patient + "_"):
        return frame_stem
    return "_".join([p for p in (patient, video) if p] + [frame_stem])


# --------------------------------------------------------------------------------------------------
# Conversion (needs cv2 / io_utils)
# --------------------------------------------------------------------------------------------------
def _iter_frames(root):
    """Yield ``(patient, video, img_path, gt_path)`` for every CADICA frame that has a GT .txt.

    Discovers every ``input`` dir under ``root`` (tolerating whether the ``selectedVideos`` wrapper
    is present) paired with a sibling ``groundTruth`` dir. Robust to missing dirs (skip, don't
    crash). Deterministic (sorted)."""
    for input_dir in sorted(glob.glob(os.path.join(root, "**", "input"), recursive=True)):
        if not os.path.isdir(input_dir):
            continue
        video_dir = os.path.dirname(input_dir)
        gt_dir = _sibling_dir(video_dir, "groundtruth")     # lowercase on Mendeley; case-insensitive for mirrors + Kaggle
        if not gt_dir:
            continue
        patient = cadica_patient_of(input_dir) or os.path.basename(video_dir)
        video = os.path.basename(video_dir)
        for ip in sorted(glob.glob(os.path.join(input_dir, "*"))):
            if os.path.splitext(ip)[1].lower() not in _IMG_EXTS:
                continue
            stem = os.path.splitext(os.path.basename(ip))[0]
            gp = os.path.join(gt_dir, stem + ".txt")
            if not os.path.isfile(gp):
                continue
            yield patient, video, ip, gp


def _convert(root, out_dir, size):
    """CLAHE+resize each CADICA frame, write YOLO images/labels/{train,val} split by PATIENT."""
    import cv2
    from src.data_prep import io_utils as io
    n = 0
    for patient, video, ip, gp in _iter_frames(root):
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H, W = g.shape
        with open(gp) as f:
            boxes = parse_cadica_gt(f.read())
        lines = cadica_boxes_to_yolo_lines(boxes, W, H)     # normalized by ORIGINAL W,H (pre-resize)
        stem = _out_stem(patient, video, os.path.splitext(os.path.basename(ip))[0])
        sp = io.split_of(patient)                           # PATIENT-grouped split -> no frame leak
        io.ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
        cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"),
                    cv2.resize(io.clahe_unsharp(g), (size, size)))
        open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(lines))
        n += 1
    return n


def main(cfg):
    from src.data_prep import io_utils as io
    d = cfg.get("datasets", {}).get("cadica")
    if not d or not d.get("root"):
        print("[cadica] no 'cadica' dataset configured; skipping.")
        return 0
    root = d["root"]
    size = cfg.get("model", {}).get("imgsz", 640)
    n = _convert(root, OUT, size)
    if n == 0:
        print(f"[cadica] WARNING: no CADICA frames converted under {root!r}; check layout.")
        return 0
    yml = io.write_yolo_datayaml(OUT, names=NAMES)
    print(f"CADICA -> {OUT} : {n} frames ; data cfg {yml}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
