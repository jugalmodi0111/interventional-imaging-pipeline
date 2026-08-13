"""Phase 1 of Dialygo ingest: a READ-ONLY inventory of an institutional handover drive.

Typing is by magic bytes, not by extension. Vendor CD dumps and PACS exports routinely store DICOM
with no extension (IM_0001, CD_IMG_0001, I0000001) or with vendor-specific ones, so an extension
match would miss most of a real handover -- and report "no DICOM found" on a drive full of DICOM,
which reads as a clean answer instead of a bug. is_dicom() reads the first 140 bytes and checks for
b"DICM" at offset 128 (Part-10), falling back to the headerless CD variant by interpreting the
first four bytes as a little-endian tag: group 0x0002 (file-meta group retained, preamble dropped)
or group 0x0008 (identification module) with a small element number.

This module never imports pydicom and never parses a dataset. A Phase 1 walk that calls dcmread on
200k files inherits every decoder failure and malformed private tag on the drive, each of which is
a chance to abort the walk; reading 140 bytes cannot fail that way.

Everything degrades instead of crashing: an unreadable file (permissions, dangling symlink, bad
sector) is recorded as kind="unreadable" with the error text and the walk continues. Progress is
checkpointed atomically after every completed directory, so a resumed run skips finished
directories and never duplicates a row.

Nothing here runs against real patient data until require_clearance passes -- it is the first
statement in scan_tree, before any walk, stat or mkdir, and `mode` defaults to "synthetic"
(Dialygo B5/B9).

CLI:  python -m src.ingest.scan --src /Volumes/HANDOVER --out .ingest --site site_a
"""
import argparse
import json
import os

from src.ingest import manifest
from src.ingest.clearance import DEFAULT_CLEARANCE_PATH, require_clearance

FILES_JSONL = "files.jsonl"
STATE_JSON = "scan_state.json"
PROVENANCE_JSON = "scan_provenance.json"

# OS/indexer droppings: macOS writes .DS_Store into every folder it displays. Counting them
# inflates the file totals the ingest plan is sized from.
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", ".Spotlight-V100", ".fseventsd",
              ".TemporaryItems", ".apDisk"}

DICOM_EXTS = {".dcm", ".dic", ".dicom", ".ima", ".img30", ".dcm30"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mpg", ".mpeg", ".mkv", ".wmv", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
# .txt is deliberately NOT a label extension: vendor drives are full of readme/log text files, and
# a hospital handover's tabular data arrives as csv/xlsx, never as YOLO-style .txt.
LABEL_EXTS = {".csv", ".xlsx", ".xls", ".json", ".xml"}

_HEAD_BYTES = 140          # 128-byte preamble + b"DICM" + a little slack for the first tag


def is_dicom(path):
    """True if the file looks like DICOM by content. Never raises; unreadable -> False."""
    try:
        with open(path, "rb") as f:
            head = f.read(_HEAD_BYTES)
    except OSError:
        return False
    if len(head) >= 132 and head[128:132] == b"DICM":
        return True
    return _looks_headerless_dicom(head)


def _looks_headerless_dicom(head):
    """Headerless (vendor CD) fallback: leading little-endian tag in group 0x0002 or 0x0008.

    A raw dataset written without a preamble starts at its lowest tag, which in practice is either
    the file-meta group (0002,0000..) when only the preamble was dropped, or the identification
    module (0008,0005 SpecificCharacterSet / 0008,0008 / 0008,0016 ...). Requiring a SMALL element
    number keeps arbitrary binaries from matching on a coincidental two bytes.
    """
    if len(head) < 8:
        return False
    group = int.from_bytes(head[0:2], "little")
    element = int.from_bytes(head[2:4], "little")
    if group not in (0x0002, 0x0008):
        return False
    return element <= 0x00FF


def classify(path):
    """Bucket a file: 'dicom' | 'video' | 'image' | 'label' | 'other'. Never raises.

    Content wins over extension. The extension check for DICOM is only a second chance for a file
    whose magic bytes we did not recognise but which announces itself (.dcm/.ima) -- Phase 1 is an
    inventory, so an over-inclusive 'dicom' bucket is cheaper than a missed study; the de-id phase
    rejects anything that will not parse.
    """
    if is_dicom(path):
        return "dicom"
    ext = os.path.splitext(path)[1].lower()
    if ext in DICOM_EXTS:
        return "dicom"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in LABEL_EXTS:
        return "label"
    return "other"


def _row(path):
    """One files.jsonl row. An unreadable file becomes a RECORD, not an exception."""
    ap = os.path.abspath(path)
    try:
        size = os.path.getsize(path)
        hk = manifest.head_key(path)
    except OSError as e:
        return {"path": ap, "kind": "unreadable", "size": -1, "head_key": None,
                "error": f"{type(e).__name__}: {e}"}
    return {"path": ap, "kind": classify(path), "size": size, "head_key": hk}


def summarize(rows):
    """Aggregate manifest rows -> {'counts': {kind: n}, 'bytes': {kind: n}, 'n_files': n}."""
    counts, byts = {}, {}
    for r in rows:
        kind = r.get("kind", "other")
        counts[kind] = counts.get(kind, 0) + 1
        size = r.get("size")
        byts[kind] = byts.get(kind, 0) + (size if isinstance(size, int) and size > 0 else 0)
    return {"counts": counts, "bytes": byts, "n_files": len(rows)}


def scan_tree(roots, out_dir, *, resume=True, mode="synthetic",
              clearance_path=DEFAULT_CLEARANCE_PATH, site="unknown"):
    """Walk `roots` read-only and write an inventory into `out_dir`. Returns a summary dict.

    Writes <out_dir>/files.jsonl (one row per file), <out_dir>/scan_state.json (resume checkpoint,
    written atomically after every completed directory) and <out_dir>/scan_provenance.json.

    resume=True skips directories already recorded in the checkpoint. resume=False starts a fresh
    files.jsonl so a re-scan cannot double-count. Symlinked directories are NOT followed (a drive
    with a loop would otherwise never terminate).

    out_dir must live OUTSIDE the scanned roots, or the scan will inventory its own output.
    """
    require_clearance(mode, clearance_path)          # Dialygo B5/B9 -- before anything touches disk

    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]
    roots = [str(r) for r in roots]

    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        raise ValueError(
            "scan_tree: root(s) not found or not a directory: " + ", ".join(missing) +
            " -- refusing to report a confident empty inventory for what may be a typo'd path.")

    os.makedirs(out_dir, exist_ok=True)
    files_path = os.path.join(out_dir, FILES_JSONL)
    state_path = os.path.join(out_dir, STATE_JSON)
    prov_path = os.path.join(out_dir, PROVENANCE_JSON)

    if resume:
        done = set(manifest.load_state(state_path).get("done_dirs", []))
        # Build set of already-recorded paths to avoid duplicating rows on mid-directory crash-resume
        already_recorded = set()
        if os.path.exists(files_path):
            try:
                for row in manifest.read_jsonl(files_path):
                    already_recorded.add(row.get("path"))
            except (OSError, ValueError):
                # If files.jsonl is corrupted, proceed without dedup to fail gracefully.
                # ValueError (not just json.JSONDecodeError, a subclass) also covers a torn
                # multi-byte UTF-8 tail surfacing as UnicodeDecodeError, in case read_jsonl's
                # own errors="replace" guard is ever bypassed by a future caller.
                pass
    else:
        done = set()
        already_recorded = set()
        if os.path.exists(files_path):
            os.remove(files_path)

    def _checkpoint():
        # Fsync files.jsonl BEFORE the atomic state write: the checkpoint is a promise that
        # every row up to this point is durable, and write_json_atomic already fsyncs the state
        # file itself. Fsyncing here, once per directory rather than once per append, is what
        # makes that promise true without paying an fsync per file on a 200k-file drive.
        if os.path.exists(files_path):
            manifest.fsync_file(files_path)
        manifest.save_state(state_path, {"schema_version": manifest.SCHEMA_VERSION,
                                         "site": site, "roots": roots,
                                         "done_dirs": sorted(done)})

    n_new = 0

    def _on_walk_error(err):
        # os.walk's default onerror=None silently drops a directory it cannot list (mode 000 is
        # routine on vendor NTFS/exFAT dumps): n_files would count only what was reachable and
        # exit 0 with no error row. Record the failure as a row instead of losing it.
        nonlocal n_new
        path = os.path.abspath(getattr(err, "filename", None) or "")
        row = {"path": path, "kind": "unreadable_dir", "error": f"{type(err).__name__}: {err}"}
        if path not in already_recorded:
            manifest.append_jsonl(files_path, row)
            already_recorded.add(path)
            n_new += 1

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, onerror=_on_walk_error):
            dirnames.sort()
            key = os.path.abspath(dirpath)
            if key in done:
                continue                              # already inventoried on a previous run
            for name in sorted(filenames):
                if name in SKIP_NAMES:
                    continue
                full_path = os.path.join(dirpath, name)
                abs_path = os.path.abspath(full_path)
                if abs_path not in already_recorded:
                    manifest.append_jsonl(files_path, _row(full_path))
                    already_recorded.add(abs_path)
                    n_new += 1
            done.add(key)
            _checkpoint()                             # atomic: a torn checkpoint would skip dirs
    _checkpoint()

    rows = manifest.read_jsonl(files_path)
    report = summarize(rows)
    report.update({
        "files_jsonl": files_path,
        "state_json": state_path,
        "new_rows": n_new,
        "site": site,
        "mode": mode,
        "roots": roots,
    })
    manifest.write_json_atomic(
        prov_path, manifest.provenance("ingest.scan", site=site, roots=roots, mode=mode,
                                       n_files=report["n_files"], counts=report["counts"]))
    return report


def main():
    """CLI entry point. --mode defaults to 'synthetic': real drives need an executed B5/B9 marker."""
    ap = argparse.ArgumentParser(
        description="Phase 1 read-only inventory of an institutional handover drive.")
    ap.add_argument("--src", nargs="+", required=True, help="one or more roots to walk")
    ap.add_argument("--out", default=".ingest", help="output dir (must be outside --src)")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--clearance", default=DEFAULT_CLEARANCE_PATH)
    ap.add_argument("--site", default="unknown", help="site tag for leave-one-site-out grouping")
    ap.add_argument("--no-resume", action="store_true", help="ignore any existing checkpoint")
    a = ap.parse_args()

    rep = scan_tree(a.src, a.out, resume=not a.no_resume, mode=a.mode,
                    clearance_path=a.clearance, site=a.site)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
