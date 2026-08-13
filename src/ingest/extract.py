"""DICOM cine -> ordered, de-identified 8-bit PNG frames + a sidecar provenance record.

Canonical layout (locked):
    <clean_root>/<site>/frames/<stem_prefix>/f00000.png
    <clean_root>/<site>/sidecar/<stem_prefix>.json
Stem grammar:
    avf_<site>_<pid_hex10>_s<NN>_<FFFFF>   e.g. avf_inu_3f9c21b04e_s01_00012

Cath cine is MONOCHROME2 with 8-12 SIGNIFICANT bits in 16-bit words, and the diagnostic contrast
sits in a narrow slice of that range described by the header's VOI LUT. Ignoring that LUT is why
angio frames so often come out washed out, so ``to_8bit`` applies it before scaling.

B3 says model input is "cropped to the segment of interest". That crop is NOT automated here --
which part of a fistulography run matters (juxta-anastomotic, cannulation zone, outflow, central)
is a clinical decision that arrives with the labels (Task 13). Guessing would silently discard the
stenosis. Extraction writes FULL frames and records the deferral in the sidecar.

The sidecar never records the source filename: handover filenames routinely contain the patient
name, so only a content hash is kept.

Part of the handover arrives already-flattened as exported AVI/MP4 video with NO DICOM metadata at
all -- no patient ID, no UIDs, no PhotometricInterpretation, no WindowCenter, no BurnedInAnnotation.
``extract_video`` decodes those into the SAME ``frames/<stem_prefix>/f00000.png`` layout, converts
to greyscale, and writes a sidecar tagged ``provenance="video"`` / ``dicom_metadata=false`` so those
frames stay distinguishable from DICOM-derived ones. ``review_required`` is unconditionally True on
that path: there is no ``BurnedInAnnotation`` to consult, and an exported clip is the MOST likely
place to find a burned-in header. An unopenable or zero-frame source raises ``IOError`` with the
path in the message -- a silent zero-frame extraction would read downstream as "this study had no
images".

``cv2``/``numpy``/``pydicom`` are imported inside functions. Runs standalone:
``python -m src.ingest.extract <dicom-or-video> --out-root <clean_root>/<site>``.
"""
import hashlib
import os

FRAME_PATTERN = "f%05d.png"
SIDECAR_SCHEMA = "dialygo.ingest.sidecar/1"
DEID_METHOD = "tag:src.ingest.deid + pixel-screen:src.ingest.pixel_deid"
CROP_DEFERRAL = ("B3 segment-of-interest crop is a clinical/annotation decision and arrives with "
                 "the labels (Task 13); ingest writes full frames")
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mpg", ".mpeg", ".mkv", ".wmv", ".m4v")


def _as_float(v):
    """Best-effort float from a pydicom value (DSfloat, MultiValue, str). None when unusable."""
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _jsonable(v):
    """Coerce pydicom value types (DSfloat/IS/MultiValue/UID/PersonName) to plain JSON types."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _sha256_bytes(blob):
    h = hashlib.sha256()
    h.update(blob)
    return h.hexdigest()


def to_8bit(arr2d, ds):
    """VOI LUT -> MONOCHROME1 invert -> min-max to 0..255 uint8.

    A constant frame (dropped/blanked/lead-in) returns a flat mid-grey array instead of dividing by
    zero -- an obviously blank frame is honest, a NaN-poisoned one is not.
    """
    import numpy as np

    try:                                                   # pydicom >= 3
        from pydicom.pixels import apply_voi_lut
    except ImportError:                                    # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_voi_lut

    raw = np.asarray(arr2d)
    try:
        a = np.asarray(apply_voi_lut(raw, ds))
    except Exception:                                      # absent or broken LUT -> raw values
        a = raw
    a = a.astype(np.float32, copy=False)

    if str(getattr(ds, "PhotometricInterpretation", "") or "").strip().upper() == "MONOCHROME1":
        a = float(a.max()) - a                             # MONOCHROME1: minimum value is WHITE

    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.full(a.shape, 128, dtype=np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def stem_prefix(site, pseudo_patient, series_idx):
    """``avf_<site>_<pid_hex10>_s<NN>`` -- the per-series half of the locked stem grammar.

    ``deid.pseudo_id`` already returns ``"<site>_<hex10>"``, so the site is prepended only when it
    is not already there. Doubling it would break the AVF group-key regex (Task 12) and silently
    re-open the patient-leakage hole.
    """
    s = str(site).strip().lower()
    pid = str(pseudo_patient).strip().lower()
    if s and not pid.startswith(s + "_"):
        pid = f"{s}_{pid}"
    return f"avf_{pid}_s{int(series_idx):02d}"


def frame_stem(prefix, frame_idx):
    """``<prefix>_<FFFFF>`` -- the logical stem of one frame (what group_key/split_of see)."""
    return f"{prefix}_{int(frame_idx):05d}"


def write_sidecar(out_root, prefix, meta):
    """Write ``<out_root>/sidecar/<prefix>.json`` atomically. Returns the path."""
    from src.ingest.manifest import provenance, write_json_atomic

    d = os.path.join(str(out_root), "sidecar")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{prefix}.json")
    payload = {"schema": SIDECAR_SCHEMA,
               "stem_prefix": prefix,
               "deid_method": DEID_METHOD,
               "crop": {"applied": False, "reason": CROP_DEFERRAL}}
    payload.update(dict(meta or {}))
    payload.setdefault("tool", provenance("src.ingest.extract"))
    write_json_atomic(path, {k: _jsonable(v) for k, v in payload.items()})
    return path


def extract_series(ds, out_root, *, site, pseudo_patient, series_idx, mask_boxes=None):
    """Write every frame of ``ds`` as an ordered PNG under ``<out_root>/frames/<stem_prefix>/``.

    ``mask_boxes`` (from ``pixel_deid.detect_text_regions``) is applied to EVERY frame -- an overlay
    banner is burned into the whole cine, not one frame of it. Returns
    ``{"stem_prefix", "n_frames", "dir", "review_required"}``.
    """
    import cv2
    import numpy as np

    from src.ingest.pixel_deid import mask_regions, needs_review

    prefix = stem_prefix(site, pseudo_patient, series_idx)
    frames_dir = os.path.join(str(out_root), "frames", prefix)
    os.makedirs(frames_dir, exist_ok=True)

    # pydicom returns a 2-D array when NumberOfFrames == 1 (no leading frame axis) -- normalize to a
    # 3-D stack before iterating so single-frame series are not silently skipped or mis-shaped.
    px = ds.pixel_array
    frames = px[None, ...] if px.ndim == 2 else px
    stack = np.asarray(frames)
    boxes = [tuple(int(v) for v in b) for b in (mask_boxes or ())]

    n = 0
    for i in range(stack.shape[0]):
        frame = to_8bit(stack[i], ds)
        if boxes:
            frame = mask_regions(frame, boxes)
        dest = os.path.join(frames_dir, FRAME_PATTERN % i)
        if not cv2.imwrite(dest, frame):
            raise IOError(f"failed to write frame {i} of {prefix} to {dest}")
        n += 1

    review = bool(needs_review(ds, boxes))
    frame_time = _as_float(getattr(ds, "FrameTime", None))
    meta = {
        "provenance": "dicom",
        "dicom_metadata": True,
        "site": str(site),
        "pseudo_patient": str(pseudo_patient),
        "pseudo_study": _jsonable(getattr(ds, "StudyInstanceUID", None)),
        "pseudo_series": _jsonable(getattr(ds, "SeriesInstanceUID", None)),
        "pseudo_sop": _jsonable(getattr(ds, "SOPInstanceUID", None)),
        "series_idx": int(series_idx),
        "modality": _jsonable(getattr(ds, "Modality", None)),
        "manufacturer": _jsonable(getattr(ds, "Manufacturer", None)),
        "photometric_interpretation": _jsonable(getattr(ds, "PhotometricInterpretation", None)),
        "bits_stored": _jsonable(getattr(ds, "BitsStored", None)),
        "window_center": _as_float(getattr(ds, "WindowCenter", None)),
        "window_width": _as_float(getattr(ds, "WindowWidth", None)),
        "rows": int(stack.shape[1]),
        "columns": int(stack.shape[2]),
        "n_frames": n,
        "frame_time_ms": frame_time,
        "fps": round(1000.0 / frame_time, 3) if frame_time else None,
        "frame_pattern": FRAME_PATTERN,
        "frame_stem_pattern": prefix + "_%05d",
        "frames_dir": os.path.relpath(frames_dir, str(out_root)),
        "source_sha256": _sha256_bytes(np.ascontiguousarray(stack).tobytes()),
        "mask_boxes": [list(b) for b in boxes],
        "burned_in_tag": _jsonable(getattr(ds, "BurnedInAnnotation", None)),
        "review_required": review,
    }
    write_sidecar(out_root, prefix, meta)
    return {"stem_prefix": prefix, "n_frames": n, "dir": frames_dir, "review_required": review}


def extract_video(path, out_root, *, site, pseudo_patient, series_idx):
    """Decode an exported AVI/MP4 clip into the SAME frame layout as ``extract_series``.

    Part of the handover is already-flattened video with no DICOM metadata at all. Those frames go
    to the same ``frames/<stem_prefix>/f00000.png`` tree, but the sidecar is tagged
    ``provenance="video"`` / ``dicom_metadata=false`` so downstream code can tell them apart.

    ``review_required`` is unconditionally True: there is no ``BurnedInAnnotation`` to consult, and
    an exported clip is the MOST likely place to find a burned-in header.

    Raises ``IOError`` when the source cannot be opened or decodes to zero frames --
    ``cv2.VideoCapture`` reports a missing file/unknown codec only via ``isOpened()``, and a silent
    empty extraction would read downstream as "this study had no images".
    """
    import cv2

    from src.ingest.manifest import sha256_file

    src = str(path)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"cannot open video for extraction: {src}")

    prefix = stem_prefix(site, pseudo_patient, series_idx)
    frames_dir = os.path.join(str(out_root), "frames", prefix)
    os.makedirs(frames_dir, exist_ok=True)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    n, rows, cols = 0, None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            rows, cols = int(grey.shape[0]), int(grey.shape[1])
            dest = os.path.join(frames_dir, FRAME_PATTERN % n)
            if not cv2.imwrite(dest, grey):
                raise IOError(f"failed to write frame {n} of {prefix} to {dest}")
            n += 1
    finally:
        cap.release()

    if n == 0:
        raise IOError(f"video opened but decoded zero frames: {src}")

    meta = {
        "provenance": "video",
        "dicom_metadata": False,
        "site": str(site),
        "pseudo_patient": str(pseudo_patient),
        "pseudo_study": None,
        "pseudo_series": None,
        "pseudo_sop": None,
        "series_idx": int(series_idx),
        "modality": None,
        "manufacturer": None,
        "photometric_interpretation": None,
        "rows": rows,
        "columns": cols,
        "n_frames": n,
        "frame_time_ms": round(1000.0 / fps, 3) if fps > 0 else None,
        "fps": fps if fps > 0 else None,
        "frame_pattern": FRAME_PATTERN,
        "frame_stem_pattern": prefix + "_%05d",
        "frames_dir": os.path.relpath(frames_dir, str(out_root)),
        "source_sha256": sha256_file(src),
        "container": os.path.splitext(src)[1].lower().lstrip("."),
        "mask_boxes": [],
        "burned_in_tag": None,
        "review_required": True,
        "review_reason": ("exported video: no DICOM header to check for burned-in annotation; "
                          "overlay text is common on workstation exports"),
    }
    write_sidecar(out_root, prefix, meta)
    return {"stem_prefix": prefix, "n_frames": n, "dir": frames_dir, "review_required": True}


def main(argv=None):
    """CLI: extract one study file -- DICOM (de-identified + pixel-screened) or exported video."""
    import argparse
    import json

    from src.ingest.clearance import require_clearance
    from src.ingest.deid import load_or_create_salt
    from src.ingest.manifest import sha256_file

    ap = argparse.ArgumentParser(
        description="Extract de-identified PNG frames from one DICOM or exported video.")
    ap.add_argument("source", help="path to a DICOM file or an exported .avi/.mp4 clip")
    ap.add_argument("--out-root", required=True, help="output root: <clean_root>/<site>")
    ap.add_argument("--site", default="inu", help="site code used in the stem grammar")
    ap.add_argument("--salt", default="_keys/salt.bin", help="path to the de-id salt file")
    ap.add_argument("--series-idx", type=int, default=1)
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"],
                    help="Dialygo B5/B9 clearance mode -- real requires an executed agreement")
    ap.add_argument("--clearance", default=None, help="path to the signed clearance record")
    args = ap.parse_args(argv)
    require_clearance(args.mode, **({"clearance_path": args.clearance} if args.clearance else {}))

    salt = load_or_create_salt(args.salt)
    if os.path.splitext(args.source)[1].lower() in VIDEO_EXTS:
        from src.ingest.deid import pseudo_id
        # No header to hash: the clip's CONTENT hash is the only stable, PHI-free identifier.
        pid = pseudo_id(salt, sha256_file(args.source), site=args.site)
        res = extract_video(args.source, args.out_root, site=args.site, pseudo_patient=pid,
                            series_idx=args.series_idx)
    else:
        import numpy as np
        import pydicom

        from src.ingest.deid import deid_dataset
        from src.ingest.pixel_deid import detect_text_regions

        ds, ids = deid_dataset(pydicom.dcmread(args.source), salt, site=args.site)
        arr = np.asarray(ds.pixel_array)
        stack = arr if arr.ndim == 3 else arr[None, ...]
        probe = sorted({0, stack.shape[0] // 2, stack.shape[0] - 1})   # overlays are static; sample 3
        boxes = sorted({b for i in probe for b in detect_text_regions(to_8bit(stack[i], ds))})
        res = extract_series(ds, args.out_root, site=args.site,
                             pseudo_patient=ids["pseudo_patient"], series_idx=args.series_idx,
                             mask_boxes=boxes)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
