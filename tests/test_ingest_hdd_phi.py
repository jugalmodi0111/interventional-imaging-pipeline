"""PHI-containment invariants for the HDD ingest driver (scripts/ingest_hdd.py).

Dialygo B5: this suite never reads real patient data. Every DICOM comes from
tests/fixtures/synthetic_dicom.py and every identifier here is an invented string.

Three defects are pinned down here, all found before the first real-drive run:

  1. `work/crosswalk_rows.jsonl` is the real-ID -> pseudonym mapping in plaintext. It was
     written at default umask while the artifact it feeds (`clean/_keys/crosswalk.csv`) was
     hardened to 0600. The scratch file must get the same protection, and must not outlive the
     crosswalk it was accumulated for.
  2. `work/deid_done.jsonl` is the ONLY file in the pipeline that pairs a pseudonym with
     anything else. It therefore must pair it with nothing real: no unshifted date, no
     wall-clock time, no source UID, no source path. Re-identification must require the salt.
  3. `work/phi_audit.md` is what a human reads before typing --ack-phi-audit. Every disposition
     claim it makes has to be derived from the scrub lists in src/ingest/deid.py, so the two
     cannot drift apart again.
"""
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

from src.ingest import deid
from src.ingest.manifest import append_jsonl, read_jsonl
from tests.fixtures.synthetic_dicom import make_xa_dataset, write_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SALT = b"S" * 32
REAL_PID = "INU-00417"
REAL_DATE = "20240517"
REAL_TIME = "101500"


def _load_driver():
    """Import scripts/ingest_hdd.py by path -- it is a script, not an installed module."""
    path = REPO_ROOT / "scripts" / "ingest_hdd.py"
    spec = importlib.util.spec_from_file_location("ingest_hdd_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hdd = _load_driver()


def _mode(path):
    return stat.S_IMODE(Path(path).stat().st_mode)


def _index_row(ds, path):
    """The subset of a dicom_index.jsonl row that run_deid actually reads."""
    return {
        "path": str(path),
        "SOPInstanceUID": ds.SOPInstanceUID,
        "StudyInstanceUID": ds.StudyInstanceUID,
        "SeriesInstanceUID": ds.SeriesInstanceUID,
        "PatientID": ds.PatientID,
        "StudyDate": ds.StudyDate,
        "StudyTime": ds.StudyTime,
        "SeriesDescription": ds.SeriesDescription,
    }


def _build_drive(tmp_path, patient_ids=(REAL_PID,), *, name="drive"):
    """Write one synthetic instance per patient and the dicom_index.jsonl that indexes them.

    Returns (work, clean, rows) where rows are the index rows as written.
    """
    src_dir = tmp_path / name
    src_dir.mkdir(parents=True, exist_ok=True)
    work = tmp_path / "work"
    clean = tmp_path / "clean"
    work.mkdir(parents=True, exist_ok=True)
    clean.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, pid in enumerate(patient_ids):
        ds = make_xa_dataset(pid, n_frames=2, rows=16, cols=16, study_date=REAL_DATE)
        ds.StudyTime = REAL_TIME
        path = write_dataset(ds, src_dir / f"{name}-{i}.dcm")
        row = _index_row(ds, path)
        rows.append(row)
        append_jsonl(work / "dicom_index.jsonl", row)
    return work, clean, rows


# ---------------------------------------------------------------------------
# Defect 1: the crosswalk scratch file is the crosswalk, and must be protected as one
# ---------------------------------------------------------------------------


def test_append_jsonl_0600_creates_the_file_owner_only(tmp_path):
    path = tmp_path / "scratch.jsonl"

    hdd.append_jsonl_0600(path, {"real": REAL_PID, "pseudo": "inu_deadbeef01"})

    assert _mode(path) == 0o600
    assert read_jsonl(path) == [{"real": REAL_PID, "pseudo": "inu_deadbeef01"}]


def test_append_jsonl_0600_rehardens_a_file_an_older_run_left_world_readable(tmp_path):
    """A run before this fix left crosswalk_rows.jsonl at 0644; appending must repair it."""
    path = tmp_path / "scratch.jsonl"
    path.write_text('{"real": "OLD", "pseudo": "inu_0000000000"}\n')
    path.chmod(0o644)

    hdd.append_jsonl_0600(path, {"real": REAL_PID, "pseudo": "inu_deadbeef01"})

    assert _mode(path) == 0o600
    assert len(read_jsonl(path)) == 2, "re-hardening must not truncate what was already there"


def test_crosswalk_rows_scratch_is_removed_once_the_crosswalk_is_written(tmp_path):
    work, clean, _ = _build_drive(tmp_path)

    stats = hdd.run_deid(work, clean, "inu", SALT, None)

    xwalk = clean / "_keys" / "crosswalk.csv"
    assert xwalk.exists() and _mode(xwalk) == 0o600
    assert not (work / "crosswalk_rows.jsonl").exists(), (
        "the plaintext scratch copy must not outlive the 0600 crosswalk it feeds"
    )
    assert stats["n_crosswalk"] == 1


def test_resume_keeps_crosswalk_entries_from_earlier_runs(tmp_path):
    """The scratch file is deleted each run, so the 0600 crosswalk must be what resume reads."""
    work, clean, _ = _build_drive(tmp_path, [REAL_PID], name="batch1")
    hdd.run_deid(work, clean, "inu", SALT, None)

    # second handover batch: a new patient appended to the same index
    ds = make_xa_dataset("INU-00999", n_frames=2, rows=16, cols=16, study_date=REAL_DATE)
    path = write_dataset(ds, tmp_path / "batch2.dcm")
    append_jsonl(work / "dicom_index.jsonl", _index_row(ds, path))

    hdd.run_deid(work, clean, "inu", SALT, None)

    mapping = deid.read_crosswalk(clean / "_keys" / "crosswalk.csv")
    assert set(mapping) == {REAL_PID, "INU-00999"}, "the first batch must survive the second run"
    assert mapping[REAL_PID] == deid.pseudo_id(SALT, REAL_PID, site="inu")


def test_rows_left_by_a_crashed_run_are_merged_into_the_crosswalk_then_removed(tmp_path):
    """A run killed mid-loop leaves scratch rows; the next run must absorb, not discard, them."""
    work, clean, _ = _build_drive(tmp_path)
    append_jsonl(work / "crosswalk_rows.jsonl",
                 {"real": "INU-CRASHED", "pseudo": "inu_abcdef0123"})

    hdd.run_deid(work, clean, "inu", SALT, None)

    mapping = deid.read_crosswalk(clean / "_keys" / "crosswalk.csv")
    assert mapping["INU-CRASHED"] == "inu_abcdef0123"
    assert not (work / "crosswalk_rows.jsonl").exists()


def test_read_crosswalk_round_trips_write_crosswalk(tmp_path):
    path = tmp_path / "_keys" / "crosswalk.csv"
    mapping = {"INU-2": "inu_2222222222", "INU-1": "inu_1111111111"}

    deid.write_crosswalk(path, mapping)

    assert deid.read_crosswalk(path) == mapping


def test_read_crosswalk_of_a_missing_file_is_empty(tmp_path):
    assert deid.read_crosswalk(tmp_path / "nope.csv") == {}


# ---------------------------------------------------------------------------
# Defect 2: deid_done.jsonl pairs a pseudonym with pseudonymous values only
# ---------------------------------------------------------------------------


def test_deid_done_records_the_shifted_study_date_not_the_real_one(tmp_path):
    work, clean, _ = _build_drive(tmp_path)
    offset = deid.day_offset(SALT, REAL_PID)
    assert offset != 0, "precondition: a zero shift would make this test vacuous"

    hdd.run_deid(work, clean, "inu", SALT, None)

    (done,) = read_jsonl(work / "deid_done.jsonl")
    assert done["study_date"] == deid.shift_date(REAL_DATE, offset)
    assert done["study_date"] != REAL_DATE


def test_deid_done_records_the_blanked_study_time(tmp_path):
    """StudyTime is in REMOVE_TAGS; the row must reflect the scrubbed value, not the real one."""
    work, clean, _ = _build_drive(tmp_path)

    hdd.run_deid(work, clean, "inu", SALT, None)

    (done,) = read_jsonl(work / "deid_done.jsonl")
    assert done["study_time"] == ""


def test_deid_done_carries_remapped_uids_not_source_uids(tmp_path):
    work, clean, rows = _build_drive(tmp_path)
    (row,) = rows

    hdd.run_deid(work, clean, "inu", SALT, None)

    (done,) = read_jsonl(work / "deid_done.jsonl")
    assert done["pseudo_study"] == deid.remap_uid(SALT, row["StudyInstanceUID"])
    assert done["pseudo_series"] == deid.remap_uid(SALT, row["SeriesInstanceUID"])
    assert done["sop"] == deid.remap_uid(SALT, row["SOPInstanceUID"])


def test_deid_done_contains_no_real_identifier_anywhere(tmp_path):
    """The whole-row invariant: re-identifying from this file must require the salt.

    Date, time, UIDs and the source path are each a lookup key against a scheduling log, a
    claim record, PACS, or a folder name typed at handover -- none needs the crosswalk.
    """
    work, clean, rows = _build_drive(tmp_path)
    (row,) = rows
    assert deid.day_offset(SALT, REAL_PID) != 0, "precondition: shift must actually move the date"

    hdd.run_deid(work, clean, "inu", SALT, None)

    blob = json.dumps(read_jsonl(work / "deid_done.jsonl"))
    for forbidden in (REAL_PID, REAL_DATE, REAL_TIME, row["StudyInstanceUID"],
                      row["SeriesInstanceUID"], row["SOPInstanceUID"], row["path"]):
        assert forbidden not in blob, f"deid_done.jsonl leaks {forbidden!r}"


def test_deid_resume_skips_instances_already_done(tmp_path):
    """Changing the resume key must not break resume: a second run must add nothing."""
    work, clean, _ = _build_drive(tmp_path)

    first = hdd.run_deid(work, clean, "inu", SALT, None)
    second = hdd.run_deid(work, clean, "inu", SALT, None)

    assert first["n_deid_new"] == 1
    assert second["n_deid_new"] == 0
    assert second["n_deid_total"] == 1, "a re-run must not append a duplicate done row"


def test_extract_allocator_still_orders_instances_deterministically(tmp_path):
    """run_extract sorts done-rows by the recorded fields; they must still be a total order."""
    work, clean, _ = _build_drive(tmp_path, [REAL_PID, REAL_PID, REAL_PID])

    hdd.run_deid(work, clean, "inu", SALT, None)
    rows = read_jsonl(work / "deid_done.jsonl")

    keys = [hdd.allocator_sort_key(r) for r in rows]
    assert len(set(keys)) == len(rows), "sort key must be unique per instance"
    assert sorted(keys) == sorted(keys)


# ---------------------------------------------------------------------------
# Defect 3: the sign-off gate must state the disposition the code actually applies
# ---------------------------------------------------------------------------


def _audit_table(work):
    """Parse phi_audit.md's identifier-density rows into {keyword: disposition text}."""
    table = {}
    for line in (work / "phi_audit.md").read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and "/" in cells[1]:
            table[cells[0]] = cells[2]
    return table


def _write_audit(tmp_path):
    work, _, _ = _build_drive(tmp_path)
    hdd.write_phi_audit(work, {"n_unparsed": 0, "n_sop_duplicates": 0})
    return work


def test_phi_audit_never_claims_a_kept_tag_is_scrubbed(tmp_path):
    work = _write_audit(tmp_path)
    text = (work / "phi_audit.md").read_text()

    assert "ALL scrubbed in phase 4" not in text, (
        "the blanket claim was false for SeriesDescription and for PatientID"
    )
    for key, disposition in _audit_table(work).items():
        if key in deid.REMOVE_TAGS:
            assert "emptied" in disposition, f"{key} is emptied but the audit says {disposition!r}"
        elif key == "PatientID":
            assert "pseudonymised" in disposition
        else:
            assert "KEPT" in disposition, f"{key} survives the scrub but the audit hides it"


def test_phi_audit_reports_series_description_as_scrubbed(tmp_path):
    work = _write_audit(tmp_path)

    assert "emptied" in _audit_table(work)["SeriesDescription"]


def test_phi_audit_disposition_follows_deid_when_a_tag_moves(monkeypatch, tmp_path):
    """The audit text is generated from REMOVE_TAGS, so the two can never drift apart again."""
    monkeypatch.setattr(deid, "REMOVE_TAGS",
                        tuple(t for t in deid.REMOVE_TAGS if t != "StudyDescription"))

    work = _write_audit(tmp_path)

    assert "KEPT" in _audit_table(work)["StudyDescription"]


@pytest.mark.parametrize("keyword", ["PatientName", "SeriesDescription", "StudyDescription"])
def test_audited_free_text_keywords_are_all_in_remove_tags(keyword):
    assert keyword in deid.REMOVE_TAGS
