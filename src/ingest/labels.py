"""Clinician label adapters + the index<->label join.

Dialygo B7: label semantics belong to the clinical lead. Nothing in this module
maps a value onto a clinical threshold; labels travel through verbatim.
Dialygo B5: free-text reports are quarantined, never parsed - narrative prose is
the densest PHI carrier in a label export.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from src.ingest.clearance import require_clearance
from src.ingest.manifest import append_jsonl, provenance, read_jsonl

# Any column whose name contains one of these is narrative -> dropped, never parsed.
NARRATIVE_TOKENS = (
    "report", "impression", "narrative", "comment", "note", "history",
    "indication", "conclusion", "remark", "text", "description", "summary",
)
KEY_COLUMNS = (
    "key", "stem", "stem_prefix", "studyinstanceuid", "study_instance_uid",
    "study_uid", "seriesinstanceuid", "series_instance_uid", "series_uid",
    "accession", "accessionnumber", "patientid", "patient_id",
)
SEGMENT_COLUMNS = ("segment", "region", "location", "site_of_lesion")
LABEL_COLUMNS = ("label", "finding", "call", "assessment", "class", "grade")

_FRAME_RE = re.compile(r"^(?P<key>.+)_(?P<frame>\d{5})$")


def _norm_col(name) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def is_narrative_column(name) -> bool:
    """True when a spreadsheet column looks like prose rather than a coded field."""
    col = _norm_col(name)
    return any(tok in col for tok in NARRATIVE_TOKENS)


def normalize_label(value) -> str:
    """Verbatim passthrough, stripped and lowercased. No thresholding (B7)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def split_stem(stem) -> tuple[str, str]:
    """`avf_inu_3f9c21b04e_s01_00012` -> (`avf_inu_3f9c21b04e_s01`, `00012`)."""
    s = str(stem or "").strip()
    m = _FRAME_RE.match(s)
    if m:
        return m.group("key"), m.group("frame")
    return s, ""


def _pick(fieldnames, candidates):
    lowered = {_norm_col(c): c for c in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_csv_labels(path) -> list[dict]:
    """Load a clinician spreadsheet export. Narrative columns are quarantined."""
    path = Path(path)
    rows: list[dict] = []
    if not path.is_file():
        return rows
    try:
        # utf-8-sig: Excel exports carry a BOM.
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            quarantined = sorted(c for c in fieldnames if is_narrative_column(c))
            usable = [c for c in fieldnames if c not in quarantined]
            keycol = _pick(usable, KEY_COLUMNS)
            segcol = _pick(usable, SEGMENT_COLUMNS)
            labcol = _pick(usable, LABEL_COLUMNS)
            for raw in reader:
                rows.append({
                    "key": str(raw.get(keycol) or "").strip() if keycol else "",
                    "segment": normalize_label(raw.get(segcol)) if segcol else "",
                    "label": normalize_label(raw.get(labcol)) if labcol else "",
                    "source": str(path),
                    "quarantined": list(quarantined),
                })
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    return rows


def load_coco_labels(path) -> list[dict]:
    """Load an annotation-tool COCO JSON export."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(doc, dict):
        return []
    images = {img.get("id"): img.get("file_name", "")
              for img in doc.get("images", []) if isinstance(img, dict)}
    cats = {c.get("id"): c.get("name", "")
            for c in doc.get("categories", []) if isinstance(c, dict)}
    rows = []
    for ann in doc.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        stem = Path(str(images.get(ann.get("image_id"), ""))).stem
        key, frame = split_stem(stem)
        bbox = ann.get("bbox") or []
        rows.append({
            "key": key,
            "frame": frame,
            "bbox": [float(v) for v in bbox] if len(bbox) == 4 else [],
            "label": normalize_label(cats.get(ann.get("category_id"), "")),
            "source": str(path),
        })
    return rows


def load_mask_dir_labels(dirpath) -> list[dict]:
    """Load a directory of PNG masks named after our own frame stems."""
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        return []
    rows = []
    for mask in sorted(dirpath.rglob("*.png")):
        key, frame = split_stem(mask.stem)
        rows.append({
            "key": key,
            "frame": frame,
            "mask_path": str(mask),
            "source": str(dirpath),
        })
    return rows


def join_labels(index_rows, label_rows, *, key):
    """Join clinician labels to DICOM index rows on `key`.

    Returns (matched, unmatched_labels, unmatched_index). Nothing is dropped:
    a label row that hits no series and an index row no label covers are both
    returned. Callers MUST treat a non-empty `unmatched_labels` as blocking --
    a clinician labelled something that is not in the export.
    """
    index_rows = list(index_rows or [])
    label_rows = list(label_rows or [])

    by_key: dict[str, list[dict]] = {}
    for row in index_rows:
        k = str(row.get(key) or "").strip()
        if k:
            by_key.setdefault(k, []).append(row)

    matched: list[dict] = []
    unmatched_labels: list[dict] = []
    hit: set[str] = set()

    for lrow in label_rows:
        lk = str(lrow.get("key") or "").strip()
        usable = bool(lk) and (
            "label" not in lrow or normalize_label(lrow.get("label")) != "")
        if not usable or lk not in by_key:
            unmatched_labels.append(lrow)
            continue
        hit.add(lk)
        for irow in by_key[lk]:
            matched.append({"key": lk, "index_row": irow, "label_row": lrow})

    unmatched_index = [r for r in index_rows
                       if str(r.get(key) or "").strip() not in hit]
    return matched, unmatched_labels, unmatched_index


def write_labels_jsonl(path, matched) -> str:
    """Write matched rows one-per-line, each self-describing with provenance."""
    path = Path(path)
    matched = list(matched)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.touch()
    prov = provenance("src.ingest.labels", n_matched=len(matched))
    for entry in matched:
        append_jsonl(path, {**entry, "provenance": prov})
    return str(path)


LOADERS = {
    "csv": load_csv_labels,
    "coco": load_coco_labels,
    "mask_dir": load_mask_dir_labels,
}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.labels",
        description="Join clinician labels to the DICOM index (Dialygo B7).")
    ap.add_argument("--index", required=True, help="dicom_index.jsonl from index_dicom")
    ap.add_argument("--labels", required=True, help="CSV / COCO JSON / mask dir")
    ap.add_argument("--kind", choices=sorted(LOADERS), default="csv")
    ap.add_argument("--key", default="StudyInstanceUID",
                    help="index field the clinician key joins against")
    ap.add_argument("--out", required=True, help="labels.jsonl to write")
    ap.add_argument("--mode", default="synthetic",
                    help="synthetic until the B5 data-use agreement executes")
    args = ap.parse_args(argv)

    require_clearance(args.mode)

    index_rows = read_jsonl(args.index)
    label_rows = LOADERS[args.kind](args.labels)
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key=args.key)
    write_labels_jsonl(args.out, matched)

    quarantined = sorted({c for r in label_rows for c in r.get("quarantined", [])})
    if quarantined:
        print(f"[labels] quarantined narrative column(s), not parsed: "
              f"{', '.join(quarantined)}")
    print(f"[labels] matched={len(matched)} "
          f"unmatched_labels={len(unmatched_labels)} "
          f"unmatched_index={len(unmatched_index)} -> {args.out}")
    for row in unmatched_index[:10]:
        print(f"[labels]   unlabelled series: {row.get(args.key)}")
    if unmatched_labels:
        for row in unmatched_labels[:10]:
            print(f"[labels]   orphan label key={row.get('key')!r} "
                  f"label={row.get('label')!r} source={row.get('source')}")
        print(f"[labels] BLOCKING: {len(unmatched_labels)} label row(s) matched no "
              f"series. Resolve with the clinical lead before training -- do not "
              f"proceed on a silently smaller label set.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
