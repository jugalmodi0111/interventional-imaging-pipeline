"""Screen angiographic frames for BURNED-IN overlay text and mask it out.

Header de-identification (``src.ingest.deid``) cannot touch PHI that the cath-lab workstation
rendered into the pixel raster -- patient name, study date, accession, institution -- and the
``BurnedInAnnotation`` tag that is supposed to warn about it is unreliable on real exports (often
absent, often ``NO`` while a banner is plainly visible). So every frame is screened regardless of
the tag, and the tag only ever ADDS a reason to escalate.

The screen is intentionally OCR-free -- no new heavy dependency, and we never need to READ the
text, only locate it:
  1. look only at the top and bottom ``SCREEN_FRACTION`` bands (overlays live at the frame edges);
  2. threshold for near-saturated pixels (overlay glyphs are rendered at ~full white);
  3. morphological CLOSE with a wide, short kernel to join glyphs into word/line runs;
  4. keep connected components that are WIDER THAN TALL and above a minimum area.

Rule 4 is what rejects the contrast-filled vessel: a thin diagonal inside a ~10-row band produces a
component about as wide as it is tall, while a banner spans the full frame width.

``cv2``/``numpy`` are imported inside functions so this module stays torch-free AND cv2-free at
import time. Runs standalone: ``python -m src.ingest.pixel_deid <dicom> --mode {synthetic,real}
[--clearance-override-for-tests PATH]``. ``--mode`` is REQUIRED (audit P0.1 fix #1); there is no
default.
"""
import os
from pathlib import Path

#: Absolute, repo-root-anchored path to the B5/B9 clearance marker. Resolved from THIS file's own
#: location, never from the process cwd -- audit P0.1 fix #2: the old CLI silently fell back to
#: clearance.DEFAULT_CLEARANCE_PATH (a bare relative string) whenever --clearance was omitted, so
#: a two-line YAML dropped anywhere a real-mode run happened to be launched from could open the
#: gate. --clearance-override-for-tests is the only way to point a run at a different marker, and
#: it exists for tests only.
DEFAULT_CLEARANCE_PATH = Path(__file__).resolve().parents[2] / "configs" / "ingest_clearance.yaml"

SCREEN_FRACTION = 0.15          # top/bottom fraction of the frame that is screened for text
SATURATION_FRACTION = 0.90      # pixels at >=90% of full 8-bit scale count as "overlay bright"
MIN_BOX_AREA = 12               # px; below this it is speckle, not a glyph run
CLOSE_KERNEL = (9, 3)           # (w, h) rect: joins glyphs horizontally, never vertically


def _bands(h):
    """Return ``(top_end, bottom_start)`` row indices of the two screened bands."""
    band = max(1, int(round(h * SCREEN_FRACTION)))
    return band, max(band, h - band)


def detect_text_regions(arr):
    """Locate burned-in text runs in a 2-D uint8 frame.

    Returns a sorted list of ``(x, y, w, h)`` boxes in FULL-FRAME coordinates (``y`` is already
    offset back out of the band it was found in). Malformed input -> ``[]`` (never raises): the
    caller's ``needs_review`` is what keeps an unscreenable frame out of the clean store.
    """
    import cv2
    import numpy as np

    a = np.asarray(arr)
    if a.ndim != 2 or a.size == 0:
        return []
    a = a.astype(np.uint8, copy=False)
    h, w = a.shape
    top_end, bot_start = _bands(h)
    thresh = int(round(255 * SATURATION_FRACTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)

    boxes = []
    for y0, y1 in ((0, top_end), (bot_start, h)):
        if y1 <= y0:
            continue
        band = a[y0:y1]
        mask = (band >= thresh).astype(np.uint8) * 255
        if not mask.any():
            continue
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for i in range(1, n_labels):                       # label 0 is the background
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if bw <= bh or area < MIN_BOX_AREA:            # not a text run (vessel, speckle)
                continue
            boxes.append((bx, y0 + by, bw, bh))
    return sorted(boxes)


def mask_regions(arr, boxes):
    """Return a COPY of ``arr`` with every ``(x, y, w, h)`` box set to 0.

    Never mutates ``arr``. Boxes are clipped to the frame, so an over-wide or negative box blanks
    only the overlapping region instead of raising or wrapping around.
    """
    import numpy as np

    out = np.array(arr, copy=True)
    if out.ndim != 2 or out.size == 0:
        return out
    h, w = out.shape
    for box in boxes or ():
        try:
            bx, by, bw, bh = (int(v) for v in box)
        except (TypeError, ValueError):
            continue                                       # malformed box -> skip, never raise
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(w, max(0, bx + bw)), min(h, max(0, by + bh))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
    return out


def needs_review(ds, boxes):
    """True when this series must be looked at by a human before it enters the clean store.

    Fail-safe by construction: ``True`` if ``BurnedInAnnotation`` says YES, OR if the pixel screen
    found anything, OR if there is no header to consult. A false positive costs one human glance;
    a false negative leaks a patient's name into the training set and everything downstream of it.
    """
    if ds is None:
        return True
    tag = str(getattr(ds, "BurnedInAnnotation", "") or "").strip().upper()
    return tag == "YES" or bool(boxes)


def main(argv=None):
    """CLI: screen every frame of one DICOM and print the boxes that would be masked.

    --mode is REQUIRED, no default (audit P0.1 fix #1). The clearance marker is always resolved
    from DEFAULT_CLEARANCE_PATH (<repo-root>/configs/ingest_clearance.yaml) unless
    --clearance-override-for-tests names a different marker -- that flag exists for tests only and
    must never be used against a real drive (audit P0.1 fix #2).
    """
    import argparse
    import json

    import numpy as np
    import pydicom

    from src.ingest.clearance import (VALID_MODES, refuse_synthetic_against_mounted_drive,
                                      require_clearance)

    ap = argparse.ArgumentParser(
        description="Screen a DICOM's frames for burned-in overlay text.",
        allow_abbrev=False)          # a bare "--clearance ..." must NOT abbreviation-match
                                     # --clearance-override-for-tests (audit P0.1 fix #2)
    ap.add_argument("dicom", help="path to a DICOM file")
    ap.add_argument("--mode", required=True, choices=list(VALID_MODES),
                     help="REQUIRED, no default -- ingest clearance mode")
    ap.add_argument("--clearance-override-for-tests", dest="clearance", default=None,
                     help="TEST-ONLY: override the clearance marker path. Production runs must "
                          "never pass this -- the marker is always read from "
                          f"{DEFAULT_CLEARANCE_PATH}.")
    args = ap.parse_args(argv)
    clearance_path = args.clearance if args.clearance is not None else DEFAULT_CLEARANCE_PATH
    require_clearance(args.mode, clearance_path=clearance_path)
    # See extract.main: mode="synthetic" is an unverified caller claim, so corroborate it against
    # the path actually about to be opened (audit P0.1 item 4).
    refuse_synthetic_against_mounted_drive(args.mode, [args.dicom])

    ds = pydicom.dcmread(args.dicom)
    px = ds.pixel_array
    arr = np.asarray(px)
    stack = arr[None, ...] if arr.ndim == 2 else arr
    detections = []
    for i in range(stack.shape[0]):
        f = stack[i].astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        u8 = (np.full(f.shape, 128, np.uint8) if hi <= lo
              else np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8))
        boxes = [list(b) for b in detect_text_regions(u8)]
        detections.append({"frame": i, "boxes": boxes})
    any_boxes = any(d["boxes"] for d in detections)
    print(json.dumps({"source": os.path.basename(args.dicom),
                      "frames": int(stack.shape[0]),
                      "burned_in_tag": str(getattr(ds, "BurnedInAnnotation", "") or ""),
                      "review_required": needs_review(ds, [1] if any_boxes else []),
                      "detections": detections}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
