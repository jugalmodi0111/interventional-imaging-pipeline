"""Phase 2: parse DICOM headers into a flat index, dedupe by SOP, build the patient hierarchy.

Reads every row from Phase 1's `files.jsonl` that was tagged `kind="dicom"`, parses only the
header (`stop_before_pixels=True`) and emits one JSON row per unique SOP instance. Header-only
parsing is what makes indexing a whole handover drive cheap: an XA cine is ~4 KB of header in
front of ~120 MB of pixels, and Phase 2 never touches the pixels.

Dedupe by SOPInstanceUID is mandatory, not cosmetic. Vendor handovers routinely contain the same
series two or three times (a CD burn, a re-burn after a bad copy, a separate PACS pull). Counting
those duplicates inflates the apparent patient count, and letting them through would cause Phase 3
to write the same patient's frames twice -- under two different pseudonyms once de-identification
runs, which puts one person on both sides of a train/test split.

Fail-safe: `read_header` returns None for anything it cannot positively identify as a DICOM
instance. An unidentifiable file is dropped from the index rather than entered with guessed
identity. `pydicom` is imported inside functions so this module stays importable (and testable)
without the imaging stack loaded.
"""
import argparse
import json
from pathlib import Path

from src.ingest.clearance import require_clearance
from src.ingest.manifest import append_jsonl, provenance, write_json_atomic

#: DICOM keywords captured for every instance. Identifying tags are included on purpose: this
#: index lives on the cleared drive beside the source data (never in the repo) and is the input
#: to the PHI audit, which has to report what is actually present *before* anything is scrubbed.
INDEX_TAGS = (
    # identity / relationships
    "SOPInstanceUID", "SOPClassUID", "StudyInstanceUID", "SeriesInstanceUID",
    "PatientID", "PatientName", "PatientBirthDate", "PatientSex", "OtherPatientIDs",
    "AccessionNumber",
    # dates / times
    "StudyDate", "StudyTime", "SeriesDate", "AcquisitionDate",
    # free text that leaks names in practice
    "StudyDescription", "SeriesDescription",
    "InstitutionName", "InstitutionAddress",
    "ReferringPhysicianName", "PerformingPhysicianName",
    # acquisition -- kept through de-identification
    "Modality", "Manufacturer", "ManufacturerModelName",
    "KVP", "ExposureTime", "DistanceSourceToDetector", "DistanceSourceToPatient",
    "PositionerPrimaryAngle", "ImagerPixelSpacing", "CineRate", "FrameTime",
    # pixel geometry -- Phase 3 needs this to plan frame extraction
    "NumberOfFrames", "Rows", "Columns", "BitsAllocated", "BitsStored",
    "PhotometricInterpretation", "WindowCenter", "WindowWidth",
    "BurnedInAnnotation",
)


def _scalar(value):
    """Coerce a pydicom element value to something json.dumps can write.

    pydicom returns PersonName, DSfloat, IS and MultiValue objects. Multi-valued elements are
    joined with the DICOM value delimiter "\\". Raw bytes (OB/OW/UN) are dropped to None rather
    than guessed at -- an unparsed binary blob is exactly the kind of thing that hides PHI.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    # PersonName and other special pydicom types should be converted to string directly
    value_type_name = type(value).__name__
    if value_type_name in ('PersonName', 'DSfloat', 'IS'):
        return str(value)
    # For MultiValue sequences, join with backslash
    try:
        return "\\".join(str(v) for v in value)
    except TypeError:
        return str(value)


def read_header(path):
    """Parse the header of `path` and return a flat record, or None if it is not a DICOM instance.

    Uses `stop_before_pixels=True` (never decodes pixel data) and `force=True` so files missing the
    128-byte preamble and "DICM" magic -- common on older vendor CD burns -- still parse. Returns
    None when SOPInstanceUID is absent or parsing raises for any reason: a file we cannot positively
    identify is excluded from the index rather than recorded under a guessed identity.

    The returned dict has keys "path", every keyword in INDEX_TAGS, and "TransferSyntaxUID"
    (from file meta, or None when the file has no meta group). All values are JSON-serialisable.
    """
    import pydicom

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    sop = getattr(ds, "SOPInstanceUID", None)
    if sop is None or str(sop).strip() == "":
        return None

    rec = {"path": str(path)}
    for kw in INDEX_TAGS:
        rec[kw] = _scalar(getattr(ds, kw, None))
    meta = getattr(ds, "file_meta", None)
    rec["TransferSyntaxUID"] = _scalar(getattr(meta, "TransferSyntaxUID", None)) if meta else None
    return rec


def dedupe_by_sop(records):
    """Collapse records sharing a SOPInstanceUID; the lexicographically first path wins.

    Sorting by path before the first-wins pass makes the winner deterministic across runs,
    machines and filesystem walk order -- two indexing runs of the same drive must produce
    byte-identical output, otherwise downstream splits are not reproducible.

    Records with no SOPInstanceUID are dropped: they have no identity to deduplicate on.
    """
    seen = {}
    for rec in sorted(records, key=lambda r: str(r.get("path", ""))):
        sop = rec.get("SOPInstanceUID")
        if sop is None or str(sop).strip() == "":
            continue
        seen.setdefault(str(sop), rec)
    return sorted(seen.values(), key=lambda r: str(r.get("path", "")))


def build_hierarchy(records):
    """Group records into {PatientID: {StudyInstanceUID: {SeriesInstanceUID: [rec, ...]}}}.

    Records missing an identifier are bucketed under UNKNOWN_PATIENT / UNKNOWN_STUDY /
    UNKNOWN_SERIES rather than dropped, so the counts a reviewer sees always add up to the number
    of instances indexed and orphans are visible instead of invisible.

    Instances within a series preserve the input order (which build_index has already sorted
    by path), so the hierarchy is reproducible.
    """
    hier = {}
    for rec in records:
        pid = str(rec.get("PatientID") or "UNKNOWN_PATIENT")
        study = str(rec.get("StudyInstanceUID") or "UNKNOWN_STUDY")
        series = str(rec.get("SeriesInstanceUID") or "UNKNOWN_SERIES")
        hier.setdefault(pid, {}).setdefault(study, {}).setdefault(series, []).append(rec)
    return hier


def build_index(files_rows, out_dir, *, mode="synthetic",
                clearance_path="configs/ingest_clearance.yaml", site="unknown"):
    """Index every kind="dicom" row, dedupe by SOP, and write the Phase 2 artifacts.

    Writes <out_dir>/dicom_index.jsonl (one JSON object per unique SOP instance),
    <out_dir>/index_errors.jsonl (one row per row dropped from the index -- see below), and
    <out_dir>/index_summary.json (counts + provenance). Returns the counts dict with keys
    n_files_seen, n_dicom_rows_seen, n_dicom, n_unparsed, n_unique_sop, n_patients, n_studies,
    n_series.

    Three different failures used to `continue` past a dropped file identically and invisibly:
    non-DICOM, a missing SOPInstanceUID, and genuinely corrupt data all make read_header return
    None, and a lost SOP-dedupe race dropped a real record with no record of it at all. If every
    damaged instance of one patient's study happens to be dropped this way, that patient silently
    vanishes from the cohort with zero signal -- and Task 6's PHI audit, which reads this index,
    never sees it happen. index_errors.jsonl makes every drop visible:
      * reason="unparseable_or_missing_sop" -- read_header(path) returned None. Non-DICOM, a
        missing SOPInstanceUID, and an outright parse failure are indistinguishable from outside
        read_header, so they share one reason rather than a guessed diagnosis.
      * reason="sop_duplicate" -- a real, parseable instance that lost the SOP dedupe race;
        "kept_copy" names the path of the duplicate that was kept, which is the only signal left
        that a losing re-burn might be truncated relative to the copy that won.

    The clearance gate runs first, before any file is opened, so `mode="real"` cannot read a
    single byte of patient data until the institutional agreement is executed (Dialygo B5).

    Re-running overwrites rather than appends, so the index and its error log are idempotent:
    two runs of the same drive produce byte-identical output.
    """
    require_clearance(mode, clearance_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "dicom_index.jsonl"
    errors_path = out / "index_errors.jsonl"
    if index_path.exists():
        index_path.unlink()
    if errors_path.exists():
        errors_path.unlink()

    records = []
    error_rows = []
    n_dicom_rows_seen = 0
    n_unparsed = 0
    for row in files_rows:
        row = row or {}
        if row.get("kind") != "dicom":
            continue
        n_dicom_rows_seen += 1
        path = row.get("path")
        rec = read_header(path)
        if rec is None:                      # non-DICOM, missing SOP, or a parse failure
            n_unparsed += 1
            error_rows.append({"path": path, "reason": "unparseable_or_missing_sop"})
            continue
        rec["head_key"] = row.get("head_key")
        rec["size"] = row.get("size")
        records.append(rec)

    unique = dedupe_by_sop(records)
    for rec in unique:
        append_jsonl(str(index_path), rec)

    # Every record in `unique` is the lexicographically-first path for its SOP (dedupe_by_sop's
    # own rule); anything in `records` that isn't that winning path lost the dedupe race.
    kept_path_by_sop = {str(rec.get("SOPInstanceUID")): str(rec.get("path", "")) for rec in unique}
    for rec in sorted(records, key=lambda r: str(r.get("path", ""))):
        kept = kept_path_by_sop.get(str(rec.get("SOPInstanceUID")))
        if kept is not None and str(rec.get("path", "")) != kept:
            error_rows.append({"path": rec.get("path"), "reason": "sop_duplicate",
                               "kept_copy": kept})

    for err in error_rows:
        append_jsonl(str(errors_path), err)

    hier = build_hierarchy(unique)
    counts = {
        "n_files_seen": len(files_rows),
        "n_dicom_rows_seen": n_dicom_rows_seen,
        "n_dicom": len(records),
        "n_unparsed": n_unparsed,
        "n_unique_sop": len(unique),
        "n_patients": len(hier),
        "n_studies": sum(len(studies) for studies in hier.values()),
        "n_series": sum(len(series) for studies in hier.values() for series in studies.values()),
    }

    write_json_atomic(str(out / "index_summary.json"), {
        "counts": counts,
        "site": site,
        "mode": mode,
        "provenance": provenance("src.ingest.index_dicom", site=site, mode=mode),
    })
    return counts


def main():
    """CLI: python -m src.ingest.index_dicom --files files.jsonl --out out/ [--site inu]

    Prints the counts dict as JSON on stdout. Returns 2 (not a traceback) when the clearance
    gate refuses, so an operator sees a clear refusal rather than a crash.
    """
    import sys

    from src.ingest.clearance import ClearanceError
    from src.ingest.manifest import read_jsonl

    ap = argparse.ArgumentParser(description="Phase 2: index DICOM headers from a scan manifest.")
    ap.add_argument("--files", required=True, help="Phase 1 files.jsonl")
    ap.add_argument("--out", required=True, help="output directory for Phase 2 artifacts")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--clearance", default="configs/ingest_clearance.yaml")
    ap.add_argument("--site", default="unknown")
    args = ap.parse_args()

    try:
        counts = build_index(
            read_jsonl(args.files), args.out,
            mode=args.mode, clearance_path=args.clearance, site=args.site,
        )
    except ClearanceError as exc:
        print(f"refused: clearance gate rejected mode={args.mode!r}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
