#!/usr/bin/env python3
"""End-to-end HDD ingest driver: sterilise (de-identify) and modularise a raw DICOM drive.

Runs the five ingest phases against a mounted institutional drive, keeping every byte of
patient data on that drive:

    1. scan     read-only inventory of the drive          -> <work>/files.jsonl
    2. index    header-only DICOM index + SOP dedupe      -> <work>/dicom_index.jsonl
    3. audit    PHI audit report for human review         -> <work>/phi_audit.md   (STOPS here
                on the first real run; re-run with --ack-phi-audit after reading it)
    4. deid     PS3.15 tag scrub + HMAC pseudonyms        -> <clean>/dicom/<pP>/<pStudy>/<pSeries>/<pSOP>.dcm
                + re-identification crosswalk             -> <clean>/_keys/crosswalk.csv  (0600, never in repo)
    5. extract  burned-in-text screen + PNG frame store   -> <clean>/frames/<stem>/f00000.png
                                                             <clean>/sidecar/<stem>.json

Every phase is resumable: re-running skips completed work. Per-file failures are logged to
<work>/*_errors.jsonl and never abort the run.

Usage (from the worktree root):

    python scripts/ingest_hdd.py --src "/Volumes/INU" --site inu --mode real
    # read <work>/phi_audit.md, then:
    python scripts/ingest_hdd.py --src "/Volumes/INU" --site inu --mode real --ack-phi-audit

The Dialygo B5/B9 legal gate runs before any disk access. --mode real refuses until BOTH flags
in configs/ingest_clearance.yaml are true; flipping them asserts the signed agreements exist.
There is no override flag by design.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingest import deid, extract, index_dicom, pixel_deid, scan  # noqa: E402
from src.ingest.clearance import ClearanceError, require_clearance  # noqa: E402
from src.ingest.manifest import append_jsonl, read_jsonl, write_json_atomic  # noqa: E402

CLEARANCE_PATH = str(REPO_ROOT / "configs" / "ingest_clearance.yaml")

# Top-level drive entries that must never be scanned or written into.
RESERVED_TOP = {
    "intv-img-clean", "intv-img-work",
    "$RECYCLE.BIN", "System Volume Information", "FOUND.000",
    ".Spotlight-V100", ".fseventsd", ".Trashes", ".TemporaryItems",
    "Autorun.inf",
}


def die(msg):
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(2)


def log(msg):
    print(msg, flush=True)


def append_jsonl_0600(path, row):
    """append_jsonl, but the file is owner-only — for anything that re-identifies the cohort.

    `manifest.append_jsonl` opens with plain `open(..., "a")`, so the file lands at whatever the
    umask allows (0644 on a typical mount). That is right for an inventory and wrong for the
    crosswalk journal, which is a real-ID -> pseudonym mapping in plaintext and therefore needs
    the same 0600 that `deid.write_crosswalk` gives the CSV it feeds.

    The mode argument to os.open only applies when the file is created, so the explicit chmod is
    what repairs a file an earlier, unhardened run already left world-readable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.chmod(str(path), 0o600)


# --------------------------------------------------------------------------- gate

def check_paths(args, src, work, clean):
    """P0.1 corroboration: the declared mode must match where the paths actually point."""
    repo = REPO_ROOT.resolve()
    if args.mode == "synthetic":
        if str(src.resolve()).startswith("/Volumes/"):
            die("--mode synthetic with --src on /Volumes/. Declaring synthetic while pointing "
                "at a mounted drive is exactly the bypass the audit flagged. Use --mode real.")
    if args.mode == "real":
        for name, p in (("--work", work), ("--clean-root", clean)):
            try:
                p.resolve().relative_to(repo)
            except ValueError:
                continue
            die(f"{name}={p} is inside the repository. Real-mode PHI artifacts (index, audit, "
                f"crosswalk) must live on the cleared drive, never in the repo tree.")


def pick_roots(src):
    """Scan roots = top-level directories of the drive, minus our own output and OS litter."""
    roots = []
    skipped = []
    for entry in sorted(os.listdir(src)):
        p = src / entry
        if entry in RESERVED_TOP or entry in scan.SKIP_NAMES:
            skipped.append(entry)
            continue
        if p.is_dir():
            roots.append(str(p))
        else:
            skipped.append(entry)          # loose top-level files (DICOMDIR etc.) — reported, not walked
    if not roots:
        die(f"no scannable directories found under {src}")
    return roots, skipped


# --------------------------------------------------------------------------- phase 2b: index errors

def summarize_index_errors(work):
    """P0.6: build_index writes index_errors.jsonl itself; roll its rows up for the audit."""
    rows = read_jsonl(work / "index_errors.jsonl")
    n_unparsed = sum(1 for r in rows if r.get("reason") == "unparseable_or_missing_sop")
    n_dup = sum(1 for r in rows if r.get("reason") == "sop_duplicate")
    return {"n_unparsed": n_unparsed, "n_sop_duplicates": n_dup}


# --------------------------------------------------------------------------- phase 3: PHI audit

def _present(rows, key):
    return sum(1 for r in rows if str(r.get(key) or "").strip())


#: Keywords whose density the audit reports. Their DISPOSITION is never written here -- it is
#: derived from deid's own lists by _disposition(), so the sentence the human signs off on cannot
#: drift away from what phase 4 actually does. The header used to assert every one of these was
#: "ALL scrubbed in phase 4"; SeriesDescription was in KEEP_TAGS and PatientID is pseudonymised
#: rather than emptied, so the gate was being told something false in two places.
AUDITED_KEYWORDS = (
    "PatientID", "PatientName", "PatientBirthDate", "OtherPatientIDs",
    "AccessionNumber", "ReferringPhysicianName", "PerformingPhysicianName",
    "InstitutionName", "InstitutionAddress", "StudyDescription", "SeriesDescription",
)


def _disposition(keyword):
    """What phase 4 will do to `keyword`, read out of deid at report time.

    Anything not named by deid's scrub lists is reported as KEPT in capitals: an identifier the
    pipeline is about to carry through to the clean drive is exactly what the reviewer is being
    asked to notice, so it must never be able to hide inside a blanket reassurance.
    """
    if keyword == "PatientID":
        return "pseudonymised (HMAC, phase 4)"
    if keyword in deid.REMOVE_TAGS:
        return "emptied (phase 4)"
    if keyword in deid.DATE_TAGS:
        return "date-shifted (phase 4)"
    return "**KEPT — survives the scrub, reaches the clean drive**"


def write_phi_audit(work, err_counts):
    rows = read_jsonl(work / "dicom_index.jsonl")
    lines = ["# PHI audit — read this before de-identification runs", ""]
    if not rows:
        lines.append("**Index is empty — nothing to audit.**")
        (work / "phi_audit.md").write_text("\n".join(lines))
        return

    n = len(rows)
    patients = {str(r.get("PatientID") or "UNKNOWN") for r in rows}
    studies = {str(r.get("StudyInstanceUID") or "") for r in rows}
    series = {str(r.get("SeriesInstanceUID") or "") for r in rows}
    dates = sorted(str(r.get("StudyDate")) for r in rows if r.get("StudyDate"))

    lines += [
        f"- unique SOP instances: **{n}**",
        f"- patients: **{len(patients)}** · studies: {len(studies)} · series: {len(series)}",
        f"- study dates: {dates[0]}..{dates[-1]}" if dates else "- study dates: none recorded",
        f"- unparseable/dropped DICOM files: **{err_counts['n_unparsed']}** "
        f"(see index_errors.jsonl) · SOP duplicates collapsed: {err_counts['n_sop_duplicates']}",
        "",
        "## Identifier density (rows carrying a non-empty value, and what phase 4 does to it)",
        "",
        "Dispositions below are read from `src/ingest/deid.py` at report time, not written by "
        "hand: `emptied` = the keyword is in REMOVE_TAGS, `date-shifted` = DATE_TAGS, "
        "`pseudonymised` = replaced by the HMAC pseudonym. Anything marked **KEPT** survives "
        "the scrub and reaches the clean drive — read those rows carefully before acknowledging.",
        "",
    ]
    for key in AUDITED_KEYWORDS:
        lines.append(f"| {key} | {_present(rows, key)}/{n} | {_disposition(key)} |")
    lines += ["", "## Burned-in annotation flags (drives phase-5 pixel screening)", ""]
    for val, c in Counter(str(r.get("BurnedInAnnotation")) for r in rows).most_common():
        lines.append(f"| BurnedInAnnotation={val} | {c} |")
    lines += ["", "## Equipment", ""]
    for val, c in Counter(f"{r.get('Manufacturer')} / {r.get('ManufacturerModelName')}"
                          for r in rows).most_common():
        lines.append(f"| {val} | {c} |")
    lines += ["", "## Modality / transfer syntax", ""]
    for val, c in Counter(f"{r.get('Modality')} · {r.get('TransferSyntaxUID')}"
                          for r in rows).most_common():
        lines.append(f"| {val} | {c} |")
    frames = [int(r["NumberOfFrames"]) for r in rows
              if str(r.get("NumberOfFrames") or "").lstrip("-").isdigit()]
    if frames:
        lines += ["", f"Frames per instance: min {min(frames)} · max {max(frames)} · "
                      f"total {sum(frames)}"]
    (work / "phi_audit.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- phase 4: deid

def run_deid(work, clean, site, salt, limit):
    """Phase 4. Two containment invariants hold over everything this writes.

    **The crosswalk exists in one protected place.** `crosswalk_rows.jsonl` is a real-ID ->
    pseudonym mapping in plaintext, i.e. the crosswalk, so it is created 0600 exactly like
    `_keys/crosswalk.csv` and is *deleted* once its rows have been folded into that CSV. It is a
    per-instance durability journal for one run, not a permanent second copy: resume reloads the
    accumulated mapping from the 0600 CSV (`deid.read_crosswalk`), and any rows a killed run left
    behind are merged on the next run before the journal is removed. Nothing is lost either way,
    and in the steady state exactly one file on the drive re-identifies the cohort.

    **`deid_done.jsonl` pairs a pseudonym with nothing real.** It is the only artifact that
    carries a pseudonym next to anything else, so every value beside it is the post-scrub one:
    the shifted StudyDate and the blanked StudyTime read off the scrubbed `ds`, and the remapped
    study/series/SOP UIDs. The source path and the source UIDs are deliberately absent -- a path
    like `.../REDDY_SURESH/IM001.dcm` is plaintext PHI, and a source UID is a direct PACS lookup
    key, so either one would re-identify a pseudonym without the salt or the crosswalk. The
    source-to-output join is still fully recoverable by anyone holding the salt, by re-deriving
    `remap_uid(salt, SOPInstanceUID)` from `dicom_index.jsonl` -- which is the point: the join
    requires the key.
    """
    rows = read_jsonl(work / "dicom_index.jsonl")
    done_path = work / "deid_done.jsonl"
    done = {r["sop"] for r in read_jsonl(done_path)}
    err_path = work / "deid_errors.jsonl"
    quarantine_path = work / "deid_quarantine.jsonl"
    xwalk_path = clean / "_keys" / "crosswalk.csv"
    xwalk_rows_path = work / "crosswalk_rows.jsonl"

    import pydicom

    n_new = n_err = n_quar = 0
    for row in rows:
        sop = str(row.get("SOPInstanceUID") or "")
        real_pid = str(row.get("PatientID") or "UNKNOWN")
        try:
            # Resume key is the remapped SOP UID, so the done log stays free of source UIDs.
            # Derived here rather than after the read: an unusable UID is a row-level error.
            done_key = deid.remap_uid(salt, sop)
        except ValueError as e:
            append_jsonl(err_path, {"path": row.get("path"), "sop": sop,
                                    "error": f"{type(e).__name__}: {e}"})
            n_err += 1
            continue
        if done_key in done:
            continue
        if limit is not None and n_new >= limit:
            log(f"  --limit {limit} reached; stopping deid early (resumable).")
            break
        try:
            ds = pydicom.dcmread(row["path"], force=True)
            ds, ids = deid.deid_dataset(ds, salt, site=site)
            leftovers = deid.residual_phi(ds)
            if leftovers:
                append_jsonl(quarantine_path, {"path": row["path"], "sop": sop,
                                               "residual": sorted(map(str, leftovers))})
                n_quar += 1
                continue                     # never write a file that failed the residual gate
            pp = ids["pseudo_patient"]
            dst = (clean / "dicom" / pp / ids["pseudo_study"]
                   / ids["pseudo_series"] / f"{ids['pseudo_sop']}.dcm")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".dcm.part")
            ds.save_as(str(tmp), enforce_file_format=True)
            os.replace(tmp, dst)
            append_jsonl_0600(xwalk_rows_path, {"real": real_pid, "pseudo": pp})
            append_jsonl(done_path, {
                "sop": ids["pseudo_sop"], "dst": str(dst), "pseudo_patient": pp,
                "pseudo_study": ids["pseudo_study"],
                "pseudo_series": ids["pseudo_series"],
                # post-scrub values off `ds`: the per-patient shifted date and the emptied time.
                "study_date": str(getattr(ds, "StudyDate", "") or ""),
                "study_time": str(getattr(ds, "StudyTime", "") or ""),
            })
            n_new += 1
        except Exception as e:  # noqa: BLE001 — one bad burn must not kill the batch
            append_jsonl(err_path, {"path": row["path"], "sop": sop,
                                    "error": f"{type(e).__name__}: {e}"})
            n_err += 1

    # Crosswalk: the 0600 CSV is the accumulated store, so seed from it, fold in this run's
    # journal (plus anything a previously killed run left there), rewrite, then drop the journal.
    mapping = deid.read_crosswalk(xwalk_path)
    for r in read_jsonl(xwalk_rows_path):
        mapping[r["real"]] = r["pseudo"]
    if mapping:
        deid.write_crosswalk(xwalk_path, mapping)
        # Only after the protected copy is on disk — never widen the window with no crosswalk.
        xwalk_rows_path.unlink(missing_ok=True)
    return {"n_deid_new": n_new, "n_deid_total": len(read_jsonl(done_path)),
            "n_deid_errors": n_err, "n_quarantined": n_quar, "n_crosswalk": len(mapping)}


# --------------------------------------------------------------------------- phase 5: extract

def _frames_of(ds):
    px = ds.pixel_array
    if px.ndim == 2:
        return px[None, ...]
    if px.ndim == 3 and px.shape[-1] in (3, 4) and int(getattr(ds, "NumberOfFrames", 1) or 1) == 1:
        return px[None, ...]
    return px


def _screen_boxes(ds):
    """Union of text boxes across first/middle/last frame — overlays are static, 3 frames suffice."""
    frames = _frames_of(ds)
    idxs = sorted({0, len(frames) // 2, len(frames) - 1})
    boxes = []
    for i in idxs:
        boxes.extend(pixel_deid.detect_text_regions(extract.to_8bit(frames[i], ds)))
    return boxes


def allocator_sort_key(r):
    """Stable per-instance order for the T17 stem allocator, over pseudonymous fields only.

    `study_date` is the shifted date and `study_time` is now always empty, but neither costs the
    allocator anything: the shift is a constant per patient and the allocator enumerates within a
    patient, so shifted dates sort exactly as the real ones did, and the remapped study/series
    UIDs plus the remapped SOP UID are a deterministic total order underneath. What the allocator
    needs is stability across runs, not wall-clock chronology.
    """
    return (r.get("study_date", ""), r.get("study_time", ""), r.get("pseudo_study", ""),
            r.get("pseudo_series", ""), r["sop"])


def run_extract(work, clean, site, limit):
    rows = read_jsonl(work / "deid_done.jsonl")
    done_path = work / "extract_done.jsonl"
    done = {r["sop"] for r in read_jsonl(done_path)}
    err_path = work / "extract_errors.jsonl"
    qa_path = work / "qa_review.jsonl"

    # T17 series-index allocator: one stem per SOP instance (each XA instance is one cine run),
    # enumerated per patient in a stable order — no two instances can share a stem.
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["pseudo_patient"]].append(r)
    alloc = {}
    for pp, rs in by_patient.items():
        rs.sort(key=allocator_sort_key)
        for i, r in enumerate(rs, start=1):
            alloc[r["sop"]] = i

    import pydicom

    n_new = n_err = n_qa = 0
    for r in rows:
        if r["sop"] in done:
            continue
        if limit is not None and n_new >= limit:
            log(f"  --limit {limit} reached; stopping extract early (resumable).")
            break
        try:
            ds = pydicom.dcmread(r["dst"], force=True)
            boxes = _screen_boxes(ds)
            prefix = extract.stem_prefix(site, r["pseudo_patient"], alloc[r["sop"]])
            if pixel_deid.needs_review(ds, boxes):
                append_jsonl(qa_path, {"stem_prefix": prefix, "dicom": r["dst"],
                                       "n_boxes": len(boxes),
                                       "burned_in": str(getattr(ds, "BurnedInAnnotation", ""))})
                n_qa += 1
            extract.extract_series(ds, str(clean), site=site,
                                   pseudo_patient=r["pseudo_patient"],
                                   series_idx=alloc[r["sop"]], mask_boxes=boxes or None)
            append_jsonl(done_path, {"sop": r["sop"], "stem_prefix": prefix,
                                     "masked_boxes": len(boxes)})
            n_new += 1
        except Exception as e:  # noqa: BLE001
            append_jsonl(err_path, {"dicom": r.get("dst"), "sop": r["sop"],
                                    "error": f"{type(e).__name__}: {e}"})
            n_err += 1
    return {"n_extract_new": n_new, "n_extract_total": len(read_jsonl(done_path)),
            "n_extract_errors": n_err, "n_flagged_for_review": n_qa}


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="drive root, e.g. /Volumes/INU")
    ap.add_argument("--site", required=True, help="lowercase site slug, e.g. inu")
    ap.add_argument("--mode", required=True, choices=("synthetic", "real"),
                    help="REQUIRED, no default — 'real' needs the B5/B9 clearance flags")
    ap.add_argument("--work", default=None,
                    help="working-state dir (default <src>/intv-img-work/<site>)")
    ap.add_argument("--clean-root", default=None,
                    help="de-identified output root (default <src>/intv-img-clean/<site>)")
    ap.add_argument("--ack-phi-audit", action="store_true",
                    help="confirm phi_audit.md has been read; unlocks phases 4-5 in real mode")
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot cap: process at most N new instances in deid and extract")
    ap.add_argument("--no-resume", action="store_true", help="fresh scan (phase 1 only)")
    args = ap.parse_args(argv)

    # Gate FIRST — before a single byte of --src is touched. Marker path is fixed; no override.
    try:
        require_clearance(args.mode, CLEARANCE_PATH)
    except ClearanceError as e:
        die(str(e))

    src = Path(args.src)
    work = Path(args.work) if args.work else src / "intv-img-work" / args.site
    clean = Path(args.clean_root) if args.clean_root else src / "intv-img-clean" / args.site
    check_paths(args, src, work, clean)
    if not src.is_dir():
        die(f"--src {src} is not a mounted directory")
    work.mkdir(parents=True, exist_ok=True)
    clean.mkdir(parents=True, exist_ok=True)

    roots, skipped = pick_roots(src)
    log(f"[gate ] mode={args.mode} clearance OK · {len(roots)} scan roots "
        f"({len(skipped)} top-level entries skipped)")

    log("[scan ] phase 1: read-only inventory ...")
    s = scan.scan_tree(roots, str(work), resume=not args.no_resume, mode=args.mode,
                       clearance_path=CLEARANCE_PATH, site=args.site)
    log(f"[scan ] {json.dumps(s.get('counts', s), default=str)}")

    log("[index] phase 2: header-only DICOM index ...")
    idx = index_dicom.build_index(read_jsonl(work / scan.FILES_JSONL), str(work),
                                  mode=args.mode, clearance_path=CLEARANCE_PATH, site=args.site)
    errs = summarize_index_errors(work)
    log(f"[index] {json.dumps(idx, default=str)}")
    log(f"[index] drop accounting: {json.dumps(errs)}")

    log("[audit] phase 3: PHI audit report ...")
    write_phi_audit(work, errs)
    log(f"[audit] written: {work / 'phi_audit.md'}")

    if args.mode == "real" and not args.ack_phi_audit:
        log("\nSTOP — human review checkpoint.")
        log(f"Read {work / 'phi_audit.md'} (identifier density, burned-in flags, drop log).")
        log("Then re-run the same command with --ack-phi-audit to run de-identification.")
        return 0

    log("[deid ] phase 4: PS3.15 tag scrub + HMAC pseudonyms ...")
    salt = deid.load_or_create_salt(clean / "_keys" / "salt.bin")
    d = run_deid(work, clean, args.site, salt, args.limit)
    log(f"[deid ] {json.dumps(d)}")

    log("[extr ] phase 5: pixel screen + PNG frame store ...")
    x = run_extract(work, clean, args.site, args.limit)
    log(f"[extr ] {json.dumps(x)}")

    report = {"src": str(src), "site": args.site, "mode": args.mode,
              "scan": s.get("counts", s), "index": idx, "index_drops": errs, **d, **x,
              "skipped_top_level": skipped}
    write_json_atomic(work / "ingest_report.json", report)

    log("\nDONE. Outputs:")
    log(f"  clean DICOM   {clean / 'dicom'}")
    log(f"  frames        {clean / 'frames'}")
    log(f"  sidecars      {clean / 'sidecar'}")
    log(f"  crosswalk     {clean / '_keys' / 'crosswalk.csv'}  <- guard this file; it re-identifies")
    log(f"  review queue  {work / 'qa_review.jsonl'} ({x['n_flagged_for_review']} flagged)")
    log(f"  quarantine    {work / 'deid_quarantine.jsonl'} ({d['n_quarantined']} held back)")
    log(f"  full report   {work / 'ingest_report.json'}")
    log("\nBefore ANY train/val split on these frames: land the Task 12 group_key AVF patch")
    log("(io_utils.py) or the split will leak frames per-patient and certify itself clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
