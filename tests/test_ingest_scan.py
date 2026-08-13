"""Phase 1 read-only inventory: magic-byte typing, resume, and the B5 gate."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.ingest.clearance import ClearanceError
from src.ingest.scan import classify, is_dicom, main, scan_tree, summarize
from tests.fixtures.synthetic_dicom import make_xa_dataset, write_dataset, write_headerless

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def drive(tmp_path):
    """A miniature scattered handover drive: nested DICOM, a headerless CD file, and noise."""
    root = tmp_path / "DRIVE"
    (root / "STUDY_A" / "SER1").mkdir(parents=True)
    (root / "VENDOR_CD").mkdir(parents=True)
    (root / "misc").mkdir(parents=True)

    write_dataset(make_xa_dataset("INU-00417", n_frames=4, rows=32, cols=32),
                  root / "STUDY_A" / "SER1" / "IM000001.dcm")
    write_dataset(make_xa_dataset("INU-00418", n_frames=4, rows=32, cols=32),
                  root / "STUDY_A" / "SER1" / "IM000002.dcm")
    # Vendor CD style: NO extension, NO preamble -- extension matching would miss this entirely.
    write_headerless(make_xa_dataset("INU-00419", n_frames=2, rows=32, cols=32),
                     root / "VENDOR_CD" / "CD_IMG_0001")

    (root / "misc" / "cine_run3.avi").write_bytes(b"RIFF\x24\x00\x00\x00AVI LIST" + b"\x00" * 64)
    (root / "misc" / "snapshot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    (root / "misc" / "labels.csv").write_text("patient_id,access_type\nINU-00417,AVF\n")
    (root / "misc" / "readme.txt").write_text("Vendor export, 2024-05-17.\n")
    (root / "misc" / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1" + b"\x00" * 32)
    return root


def _by_name(rows):
    return {os.path.basename(r["path"]): r for r in rows}


def _all_files(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        out.extend(os.path.join(dirpath, n) for n in filenames)
    return sorted(out)


def test_is_dicom_part10(drive):
    assert is_dicom(str(drive / "STUDY_A" / "SER1" / "IM000001.dcm")) is True


def test_is_dicom_headerless(drive):
    """No preamble, no extension -- must still be recognised via the leading group number."""
    assert is_dicom(str(drive / "VENDOR_CD" / "CD_IMG_0001")) is True


def test_is_dicom_rejects_png(drive):
    assert is_dicom(str(drive / "misc" / "snapshot.png")) is False
    assert is_dicom(str(drive / "misc" / "cine_run3.avi")) is False
    assert is_dicom(str(drive / "misc" / "readme.txt")) is False


def test_is_dicom_rejects_short_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert is_dicom(str(p)) is False
    q = tmp_path / "tiny.bin"
    q.write_bytes(b"\x08\x00")
    assert is_dicom(str(q)) is False


def test_is_dicom_missing_path_returns_false(tmp_path):
    assert is_dicom(str(tmp_path / "nope.dcm")) is False


@pytest.mark.parametrize("name,kind", [
    ("STUDY_A/SER1/IM000001.dcm", "dicom"),
    ("STUDY_A/SER1/IM000002.dcm", "dicom"),
    ("VENDOR_CD/CD_IMG_0001", "dicom"),
    ("misc/cine_run3.avi", "video"),
    ("misc/snapshot.png", "image"),
    ("misc/labels.csv", "label"),
    ("misc/readme.txt", "other"),
])
def test_classify_per_file(drive, name, kind):
    assert classify(str(drive / name)) == kind


def test_scan_records_one_row_per_file_with_expected_keys(drive, tmp_path):
    out = tmp_path / ".ingest"
    rep = scan_tree([str(drive)], str(out))
    from src.ingest.manifest import read_jsonl

    rows = read_jsonl(str(out / "files.jsonl"))
    assert len(rows) == 7
    assert rep["n_files"] == 7
    assert rep["counts"] == {"dicom": 3, "video": 1, "image": 1, "label": 1, "other": 1}
    assert rep["bytes"]["dicom"] > 0
    for r in rows:
        assert {"path", "kind", "size", "head_key"} <= set(r)
        assert os.path.isabs(r["path"])
        assert r["size"] > 0
        assert re.fullmatch(r"\d+:[0-9a-f]{64}", r["head_key"])
    assert _by_name(rows)["CD_IMG_0001"]["kind"] == "dicom"


def test_scan_skips_ds_store(drive, tmp_path):
    from src.ingest.manifest import read_jsonl

    out = tmp_path / ".ingest"
    scan_tree([str(drive)], str(out))
    names = set(_by_name(read_jsonl(str(out / "files.jsonl"))))
    assert ".DS_Store" not in names


def test_scan_resume_does_not_duplicate_rows(drive, tmp_path):
    from src.ingest.manifest import read_jsonl

    out = tmp_path / ".ingest"
    first = scan_tree([str(drive)], str(out))
    second = scan_tree([str(drive)], str(out))
    assert first["new_rows"] == 7
    assert second["new_rows"] == 0
    rows = read_jsonl(str(out / "files.jsonl"))
    assert len(rows) == 7
    assert len({r["path"] for r in rows}) == 7
    assert second["n_files"] == 7


def test_scan_resume_does_not_duplicate_after_mid_directory_crash(drive, tmp_path):
    """Simulate a crash mid-directory: files already appended, directory not marked done.

    On resume, the walker re-enters the directory. Without deduplication, it would re-append
    every file already recorded, creating duplicates in files.jsonl. This test verifies that
    the scan correctly deduplicates on resume.
    """
    from src.ingest.manifest import read_jsonl, write_json_atomic
    import json

    out = tmp_path / ".ingest"
    os.makedirs(out, exist_ok=True)

    # Simulate a first run that crashes mid-directory STUDY_A/SER1.
    # Two files from SER1 have been appended, but the directory is not yet marked done.
    ser1_path = str(drive / "STUDY_A" / "SER1")
    im1_path = str(drive / "STUDY_A" / "SER1" / "IM000001.dcm")
    im2_path = str(drive / "STUDY_A" / "SER1" / "IM000002.dcm")

    # Write pre-crash state: files.jsonl with two files from SER1
    from src.ingest.scan import classify
    rows = [
        {"path": im1_path, "kind": classify(im1_path), "size": os.path.getsize(im1_path),
         "head_key": "dummy:abc123"},
        {"path": im2_path, "kind": classify(im2_path), "size": os.path.getsize(im2_path),
         "head_key": "dummy:def456"},
    ]
    for row in rows:
        from src.ingest.manifest import append_jsonl
        append_jsonl(str(out / "files.jsonl"), row)

    # Write pre-crash state: scan_state.json WITHOUT SER1 or its parent directories marked done
    # (simulating that _checkpoint never ran after the crash)
    write_json_atomic(
        str(out / "scan_state.json"),
        {"schema_version": "1.0", "site": "unknown", "roots": [str(drive)], "done_dirs": []}
    )

    # Now resume: should NOT duplicate the two files already in files.jsonl
    rep = scan_tree([str(drive)], str(out), resume=True)

    # Verify: 7 total rows (2 from pre-crash + 5 new), but 7 unique paths
    final_rows = read_jsonl(str(out / "files.jsonl"))
    assert len(final_rows) == 7, f"Expected 7 total rows, got {len(final_rows)}"
    unique_paths = {r["path"] for r in final_rows}
    assert len(unique_paths) == 7, f"Expected 7 unique paths, got {len(unique_paths)}"
    # The two files from SER1 should appear exactly once each in the final rows
    assert sum(1 for r in final_rows if r["path"] == im1_path) == 1, "IM000001.dcm duplicated"
    assert sum(1 for r in final_rows if r["path"] == im2_path) == 1, "IM000002.dcm duplicated"


def test_scan_no_resume_rewrites_instead_of_appending(drive, tmp_path):
    from src.ingest.manifest import read_jsonl

    out = tmp_path / ".ingest"
    scan_tree([str(drive)], str(out))
    rep = scan_tree([str(drive)], str(out), resume=False)
    assert rep["new_rows"] == 7
    assert len(read_jsonl(str(out / "files.jsonl"))) == 7


def test_scan_leaves_originals_untouched(drive, tmp_path):
    """Phase 1 is READ-ONLY. Prove it: no original mtime moves, no file appears or disappears."""
    before = {p: os.stat(p).st_mtime_ns for p in _all_files(drive)}
    scan_tree([str(drive)], str(tmp_path / ".ingest"))
    after = {p: os.stat(p).st_mtime_ns for p in _all_files(drive)}
    assert before == after


def test_scan_real_mode_raises_before_writing_anything(drive, tmp_path):
    marker = tmp_path / "clearance.yaml"
    marker.write_text("data_agreement_executed: false\nip_agreement_executed: false\n")
    out = tmp_path / ".ingest"
    with pytest.raises(ClearanceError) as ei:
        scan_tree([str(drive)], str(out), mode="real", clearance_path=str(marker))
    assert "B5" in str(ei.value) and "B9" in str(ei.value)
    assert not out.exists()          # gate runs before the output dir is even created


def test_scan_records_unreadable_file_without_crashing(drive, tmp_path):
    from src.ingest.manifest import read_jsonl

    os.symlink(str(drive / "does_not_exist.dcm"), str(drive / "STUDY_A" / "broken.dcm"))
    out = tmp_path / ".ingest"
    rep = scan_tree([str(drive)], str(out))
    rows = _by_name(read_jsonl(str(out / "files.jsonl")))
    assert rep["n_files"] == 8
    bad = rows["broken.dcm"]
    assert bad["kind"] == "unreadable"
    assert bad["head_key"] is None
    assert "FileNotFoundError" in bad["error"]
    assert rep["counts"]["unreadable"] == 1
    assert rep["counts"]["dicom"] == 3      # the walk continued past the failure


_CAN_DENY_PERMISSIONS = (
    os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0
)


@pytest.mark.skipif(not _CAN_DENY_PERMISSIONS,
                    reason="chmod 000 does not restrict access on Windows or as root")
def test_scan_records_unreadable_directory_without_crashing(tmp_path):
    """Mode-000 subdirectory (routine on vendor NTFS/exFAT dumps) must not vanish silently.

    Before this fix, os.walk's default onerror=None swallowed the PermissionError: n_files
    counted only what was reachable, the scan exited 0, and nothing in files.jsonl or the
    summary hinted that a whole directory had been skipped.
    """
    from src.ingest.manifest import read_jsonl

    root = tmp_path / "DRIVE"
    locked = root / "LOCKED_DIR"
    locked.mkdir(parents=True)
    (locked / "hidden.dcm").write_bytes(b"\x00" * 200)   # never seen -- dir is unreadable
    (root / "visible.txt").write_text("ok\n")
    os.chmod(str(locked), 0o000)
    try:
        out = tmp_path / ".ingest"
        rep = scan_tree([str(root)], str(out))          # must not raise
        rows = read_jsonl(str(out / "files.jsonl"))
        bad = [r for r in rows if r["kind"] == "unreadable_dir"]
        assert len(bad) == 1
        assert bad[0]["path"] == os.path.abspath(str(locked))
        assert bad[0]["error"]
        assert rep["counts"]["unreadable_dir"] == 1
        assert rep["counts"]["other"] == 1                # visible.txt still recorded
    finally:
        os.chmod(str(locked), 0o755)                     # let pytest clean up tmp_path


def test_scan_tree_raises_on_missing_root(tmp_path):
    """A typo'd drive path must refuse loudly, not report a confident empty inventory."""
    out = tmp_path / ".ingest"
    bad = tmp_path / "TYPO_DRIVE"
    with pytest.raises(ValueError) as ei:
        scan_tree([str(bad)], str(out))
    assert str(bad) in str(ei.value)
    assert not out.exists()                              # refuses before any output is written


def test_scan_tree_raises_naming_every_missing_root(drive, tmp_path):
    """Multiple bad roots: every one must be named, not just the first."""
    out = tmp_path / ".ingest"
    bad1 = tmp_path / "NOPE_ONE"
    bad2 = tmp_path / "NOPE_TWO"
    with pytest.raises(ValueError) as ei:
        scan_tree([str(drive), str(bad1), str(bad2)], str(out))
    msg = str(ei.value)
    assert str(bad1) in msg
    assert str(bad2) in msg
    assert not out.exists()


def test_scan_fsyncs_files_jsonl_before_checkpoint_not_on_every_append(tmp_path, monkeypatch):
    """Durability fix: files.jsonl is fsynced once per checkpoint, not once per append.

    Before this fix, only the state checkpoint was fsynced (write_json_atomic); files.jsonl
    relied on the OS to eventually flush its buffers. After a power loss, scan_state.json could
    vouch for rows that were never actually written to disk, and resume would then skip that
    directory forever, believing it complete.
    """
    from src.ingest import manifest as manifest_mod

    root = tmp_path / "DRIVE"
    root.mkdir()
    for i in range(20):
        (root / f"f{i:03d}.txt").write_text("x")

    calls = []
    orig = manifest_mod.fsync_file

    def spy(path):
        calls.append(path)
        return orig(path)

    monkeypatch.setattr(manifest_mod, "fsync_file", spy)
    out = tmp_path / ".ingest"
    scan_tree([str(root)], str(out))

    assert calls                                          # fsync_file was actually used
    assert len(calls) < 20                                # NOT once per append
    assert all(c == str(out / "files.jsonl") for c in calls)


def test_scan_survives_torn_multibyte_tail_in_files_jsonl_on_resume(drive, tmp_path):
    """A torn append landing mid multi-byte UTF-8 character (a non-ASCII patient name is
    near-certain on an institutional drive) must not crash resume's dedup read or the final
    summarize() read -- both call manifest.read_jsonl on files.jsonl.
    """
    out = tmp_path / ".ingest"
    scan_tree([str(drive)], str(out))
    files_path = out / "files.jsonl"
    with open(files_path, "ab") as f:
        # First byte of a 2-byte UTF-8 sequence, no continuation byte -- exactly what a drive
        # yanked mid-write leaves behind.
        f.write(b'{"path": "/x", "name": "R\xc3')

    rep = scan_tree([str(drive)], str(out), resume=True)   # must not raise
    assert rep["n_files"] == 7                              # torn tail dropped, nothing new


def test_scan_writes_state_and_provenance(drive, tmp_path):
    import json

    from src.ingest.manifest import SCHEMA_VERSION, load_state

    out = tmp_path / ".ingest"
    scan_tree([str(drive)], str(out), site="site_a")
    st = load_state(str(out / "scan_state.json"))
    assert st["schema_version"] == SCHEMA_VERSION
    assert st["site"] == "site_a"
    # DRIVE, DRIVE/STUDY_A, DRIVE/STUDY_A/SER1, DRIVE/VENDOR_CD, DRIVE/misc
    assert len(st["done_dirs"]) == 5
    prov = json.loads((out / "scan_provenance.json").read_text())
    assert prov["tool"] == "ingest.scan"
    assert prov["site"] == "site_a"


def test_summarize_counts_and_bytes():
    rows = [
        {"path": "/a", "kind": "dicom", "size": 10, "head_key": "10:x"},
        {"path": "/b", "kind": "dicom", "size": 5, "head_key": "5:y"},
        {"path": "/c", "kind": "image", "size": 3, "head_key": "3:z"},
        {"path": "/d", "kind": "unreadable", "size": -1, "head_key": None, "error": "boom"},
    ]
    s = summarize(rows)
    assert s["n_files"] == 4
    assert s["counts"] == {"dicom": 2, "image": 1, "unreadable": 1}
    assert s["bytes"] == {"dicom": 15, "image": 3, "unreadable": 0}


def test_summarize_empty():
    assert summarize([]) == {"counts": {}, "bytes": {}, "n_files": 0}


def test_main_smoke(drive, tmp_path, monkeypatch, capsys):
    out = tmp_path / ".ingest"
    monkeypatch.setattr("sys.argv", ["scan", "--src", str(drive), "--out", str(out)])
    assert main() == 0
    printed = capsys.readouterr().out
    assert '"n_files": 7' in printed
    assert (out / "files.jsonl").exists()


def test_module_import_is_torch_cv2_and_pydicom_free():
    """Repo convention: heavy imports live inside functions, not at module scope."""
    code = (
        "import sys, src.ingest.scan;"
        "bad=[k for k in ('torch','cv2','pydicom') if k in sys.modules];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stdout + r.stderr
