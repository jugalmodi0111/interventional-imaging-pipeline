"""Phase 2 DICOM header index: read_header, SOP dedupe, hierarchy, counts.

Dialygo B5: every test here builds its DICOM from tests/fixtures/synthetic_dicom.py.
No real patient data is read by this suite, and none may be until the institutional
agreement executes.
"""
import json

import pytest

from src.ingest.clearance import ClearanceError
from src.ingest.index_dicom import (
    INDEX_TAGS,
    build_hierarchy,
    build_index,
    dedupe_by_sop,
    read_header,
)
from tests.fixtures.synthetic_dicom import make_xa_dataset, write_dataset, write_headerless

# Deterministic UIDs so dedupe/hierarchy assertions never depend on random generation.
P1_STUDY = "1.2.826.0.1.3680043.8.498.1001"
P1_SER_A = "1.2.826.0.1.3680043.8.498.2001"
P1_SER_B = "1.2.826.0.1.3680043.8.498.2002"
P2_STUDY = "1.2.826.0.1.3680043.8.498.1002"
P2_SER_A = "1.2.826.0.1.3680043.8.498.2003"


def _row(path, kind="dicom"):
    """A scan.py files.jsonl row for `path`."""
    from pathlib import Path
    p = Path(path)
    return {"path": str(path), "kind": kind, "size": p.stat().st_size, "head_key": "hk"}


def test_read_header_parses_part10_and_is_json_serialisable(tmp_path):
    ds = make_xa_dataset("INU-00417", n_frames=8, sop_uid="1.2.826.0.1.3680043.8.498.3001",
                         PatientName="REDDY^SURESH^^Mr", ManufacturerModelName="Artis Zee")
    p = write_dataset(ds, tmp_path / "case.dcm")

    rec = read_header(p)

    assert rec is not None
    assert rec["path"] == str(p)
    assert rec["SOPInstanceUID"] == "1.2.826.0.1.3680043.8.498.3001"
    assert rec["PatientID"] == "INU-00417"
    assert rec["Modality"] == "XA"
    # B6: vendor identity is deliberately indexed and later retained.
    assert rec["Manufacturer"] == "Siemens"
    assert rec["ManufacturerModelName"] == "Artis Zee"
    assert rec["PatientName"] == "REDDY^SURESH^^Mr"      # PersonName coerced to str
    assert rec["KVP"] == 70.0                            # DSfloat coerced to float
    assert rec["NumberOfFrames"] == 8                    # IS coerced to int
    assert set(INDEX_TAGS).issubset(rec.keys())
    json.dumps(rec)                                      # must not raise


def test_read_header_accepts_headerless_vendor_burn(tmp_path):
    """No preamble / no DICM magic still indexes -- this is why force=True is used."""
    ds = make_xa_dataset("INU-00417", sop_uid="1.2.826.0.1.3680043.8.498.3002")
    p = write_headerless(ds, tmp_path / "IM_0001")

    rec = read_header(p)

    assert rec is not None
    assert rec["SOPInstanceUID"] == "1.2.826.0.1.3680043.8.498.3002"
    assert rec["TransferSyntaxUID"] is None      # raw dataset carries no file meta


def test_read_header_returns_none_for_non_dicom(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("patient list, not a dicom file")

    assert read_header(p) is None


def test_read_header_returns_none_for_missing_sop_uid(tmp_path):
    """A parseable dataset with no SOPInstanceUID has no identity -- drop it, do not guess."""
    ds = make_xa_dataset("INU-00417", sop_uid="1.2.826.0.1.3680043.8.498.3003")
    p = write_dataset(ds, tmp_path / "nosop.dcm")
    import pydicom

    parsed = pydicom.dcmread(str(p), force=True)
    del parsed.SOPInstanceUID
    parsed.save_as(str(p), enforce_file_format=False)

    assert read_header(p) is None


def test_read_header_returns_none_for_missing_file(tmp_path):
    assert read_header(tmp_path / "does_not_exist.dcm") is None


def test_dedupe_by_sop_keeps_first_path_and_is_order_independent():
    """The same SOP under a burn folder and a PACS pull must collapse to one record.

    "0_burn/..." sorts before "a_pacs/...", so the burn copy is the winner regardless of the
    order the records were discovered in.
    """
    dup_a = {"path": "/drive/a_pacs/im1.dcm", "SOPInstanceUID": "S1", "PatientID": "P1"}
    dup_b = {"path": "/drive/0_burn/im1.dcm", "SOPInstanceUID": "S1", "PatientID": "P1"}
    other = {"path": "/drive/a_pacs/im2.dcm", "SOPInstanceUID": "S2", "PatientID": "P1"}

    forward = dedupe_by_sop([dup_a, dup_b, other])
    reverse = dedupe_by_sop([other, dup_b, dup_a])

    assert len(forward) == 2
    assert forward == reverse                                  # deterministic
    assert [r["path"] for r in forward] == ["/drive/0_burn/im1.dcm", "/drive/a_pacs/im2.dcm"]


def test_dedupe_by_sop_drops_records_without_a_sop_uid():
    kept = {"path": "/d/a.dcm", "SOPInstanceUID": "S1"}
    junk = {"path": "/d/b.dcm", "SOPInstanceUID": None}

    assert dedupe_by_sop([kept, junk]) == [kept]


def _rec(path, patient, study, series, sop):
    return {
        "path": path, "PatientID": patient, "StudyInstanceUID": study,
        "SeriesInstanceUID": series, "SOPInstanceUID": sop,
    }


def test_build_hierarchy_shape():
    recs = [
        _rec("/d/1.dcm", "INU-00417", P1_STUDY, P1_SER_A, "S1"),
        _rec("/d/2.dcm", "INU-00417", P1_STUDY, P1_SER_A, "S2"),
        _rec("/d/3.dcm", "INU-00417", P1_STUDY, P1_SER_B, "S3"),
        _rec("/d/4.dcm", "INU-00902", P2_STUDY, P2_SER_A, "S4"),
    ]

    hier = build_hierarchy(recs)

    assert set(hier) == {"INU-00417", "INU-00902"}
    assert set(hier["INU-00417"]) == {P1_STUDY}
    assert set(hier["INU-00417"][P1_STUDY]) == {P1_SER_A, P1_SER_B}
    assert len(hier["INU-00417"][P1_STUDY][P1_SER_A]) == 2
    assert len(hier["INU-00417"][P1_STUDY][P1_SER_B]) == 1
    assert len(hier["INU-00902"][P2_STUDY][P2_SER_A]) == 1


def test_build_hierarchy_buckets_missing_ids_instead_of_dropping():
    """A record with no PatientID is quarantined under a sentinel, never silently discarded."""
    hier = build_hierarchy([_rec("/d/x.dcm", None, None, None, "S9")])

    assert hier["UNKNOWN_PATIENT"]["UNKNOWN_STUDY"]["UNKNOWN_SERIES"][0]["SOPInstanceUID"] == "S9"


@pytest.fixture
def handover(tmp_path):
    """A miniature vendor handover drive.

    Patient 1 (INU-00417): one study, two series (2 instances + 1 instance).
    Patient 2 (INU-00902): one study, one series, one instance.
    Plus a re-burn of patient 1's first instance under a folder that sorts earlier,
    a text file mis-tagged kind="dicom" by Phase 1, and a non-DICOM row Phase 2 must ignore.
    """
    burn = tmp_path / "0_burn"
    pacs = tmp_path / "a_pacs"
    burn.mkdir()
    pacs.mkdir()

    def xa(pid, study, series, sop, study_date="20240517"):
        return make_xa_dataset(pid, study_uid=study, series_uid=series, sop_uid=sop,
                               study_date=study_date)

    p1_a1 = write_dataset(xa("INU-00417", P1_STUDY, P1_SER_A, "SOP-A1"), pacs / "a1.dcm")
    p1_a2 = write_dataset(xa("INU-00417", P1_STUDY, P1_SER_A, "SOP-A2"), pacs / "a2.dcm")
    p1_b1 = write_dataset(xa("INU-00417", P1_STUDY, P1_SER_B, "SOP-B1"), pacs / "b1.dcm")
    p2_c1 = write_dataset(xa("INU-00902", P2_STUDY, P2_SER_A, "SOP-C1"), pacs / "c1.dcm")
    dup = write_dataset(xa("INU-00417", P1_STUDY, P1_SER_A, "SOP-A1"), burn / "a1.dcm")

    bogus = tmp_path / "PATLIST.TXT"
    bogus.write_text("this is not dicom")
    other = tmp_path / "readme.pdf"
    other.write_bytes(b"%PDF-1.4\n")

    rows = [
        _row(p1_a1), _row(p1_a2), _row(p1_b1), _row(p2_c1), _row(dup),
        _row(bogus),                       # Phase 1 guessed dicom, header parse says otherwise
        _row(other, kind="other"),         # never opened by Phase 2
    ]
    return rows, dup


def _read_jsonl_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_build_index_counts_and_artifacts(tmp_path, handover):
    rows, dup = handover
    out = tmp_path / "index_out"

    counts = build_index(rows, out, mode="synthetic", site="inu")

    assert counts == {
        "n_files_seen": 7,           # every row handed to us, including the pdf
        "n_dicom_rows_seen": 6,      # rows Phase 1 tagged kind="dicom" (5 real + the .TXT)
        "n_dicom": 5,                # headers that actually parsed (the .TXT did not)
        "n_unparsed": 1,             # the .TXT: read_header returned None
        "n_unique_sop": 4,           # the re-burn collapsed away
        "n_patients": 2,
        "n_studies": 2,
        "n_series": 3,
    }

    indexed = [json.loads(line) for line in
               (out / "dicom_index.jsonl").read_text().splitlines() if line.strip()]
    assert len(indexed) == 4
    assert {r["SOPInstanceUID"] for r in indexed} == {"SOP-A1", "SOP-A2", "SOP-B1", "SOP-C1"}
    # first path wins: the 0_burn copy of SOP-A1 sorts before the a_pacs copy
    a1 = next(r for r in indexed if r["SOPInstanceUID"] == "SOP-A1")
    assert a1["path"] == str(dup)
    # scan rows are carried through for traceability back to Phase 1
    assert a1["head_key"] == "hk"

    summary = json.loads((out / "index_summary.json").read_text())
    assert summary["counts"] == counts
    assert summary["site"] == "inu"
    assert summary["mode"] == "synthetic"
    assert summary["provenance"]["tool"] == "src.ingest.index_dicom"

    # Every file dropped from the index now leaves a visible trace instead of a silent `continue`.
    errors = _read_jsonl_rows(out / "index_errors.jsonl")
    assert len(errors) == 2
    bogus_path = rows[5]["path"]                     # PATLIST.TXT, tagged dicom by Phase 1
    p1_a1_path = rows[0]["path"]                      # the a_pacs copy that lost the dedupe race
    assert {"path": bogus_path, "reason": "unparseable_or_missing_sop"} in errors
    dup_row = next(e for e in errors if e["reason"] == "sop_duplicate")
    assert dup_row["path"] == p1_a1_path
    assert dup_row["kept_copy"] == str(dup)


def test_build_index_is_idempotent(tmp_path, handover):
    rows, _ = handover
    out = tmp_path / "index_out"

    first = build_index(rows, out, mode="synthetic", site="inu")
    text_first = (out / "dicom_index.jsonl").read_text()
    errors_first = (out / "index_errors.jsonl").read_text()
    second = build_index(rows, out, mode="synthetic", site="inu")

    assert first == second
    assert (out / "dicom_index.jsonl").read_text() == text_first    # no append-on-rerun
    assert (out / "index_errors.jsonl").read_text() == errors_first  # error log truncates too


def test_build_index_writes_error_row_for_corrupt_file(tmp_path):
    """A genuinely corrupt file (not just non-DICOM) must also leave a visible trace.

    read_header cannot distinguish non-DICOM, missing-SOPInstanceUID and outright-corrupt --
    all three return None -- so all three share one reason string.
    """
    junk = tmp_path / "corrupt.dcm"
    junk.write_bytes(b"\x00\x01\x02\x03" * 64)          # not empty, not a DICOM instance either
    out = tmp_path / "index_out"
    rows = [_row(junk)]

    counts = build_index(rows, out, mode="synthetic", site="inu")

    assert counts["n_dicom_rows_seen"] == 1
    assert counts["n_unparsed"] == 1
    assert counts["n_dicom"] == 0
    errors = _read_jsonl_rows(out / "index_errors.jsonl")
    assert errors == [{"path": str(junk), "reason": "unparseable_or_missing_sop"}]


def test_build_index_writes_sop_duplicate_row_with_kept_copy(tmp_path, handover):
    """The losing copy of a SOP dedupe race must be named, along with what beat it.

    Losing dedupe silently used to discard the loser's size/head_key with no trace at all --
    the only signal that a re-burn might be truncated relative to the copy that was kept.
    """
    rows, dup = handover
    out = tmp_path / "index_out"

    build_index(rows, out, mode="synthetic", site="inu")

    errors = _read_jsonl_rows(out / "index_errors.jsonl")
    dup_rows = [e for e in errors if e["reason"] == "sop_duplicate"]
    assert len(dup_rows) == 1
    assert dup_rows[0]["path"] == rows[0]["path"]        # the a_pacs copy of SOP-A1
    assert dup_rows[0]["kept_copy"] == str(dup)          # the 0_burn copy won lexicographically


def test_build_index_refuses_synthetic_mode_against_a_volumes_row(tmp_path):
    """P0.1 fix #3: build_index corroborates the synthetic claim against files_rows' own paths --
    a fabricated /Volumes row is refused without touching a real mounted drive."""
    out = tmp_path / "index_out"
    rows = [{"path": "/Volumes/CATHLAB_HANDOVER/STUDY_A/im1.dcm", "kind": "dicom",
            "size": 100, "head_key": "hk"}]

    with pytest.raises(ClearanceError) as ei:
        build_index(rows, out, mode="synthetic", site="inu")

    msg = str(ei.value)
    assert "synthetic" in msg.lower() and "/Volumes" in msg
    assert not out.exists()


def test_build_index_refuses_real_mode_before_the_agreement(tmp_path, handover):
    """Dialygo B5: the clearance gate is checked before a single header is opened."""
    rows, _ = handover
    out = tmp_path / "index_out"

    with pytest.raises(ClearanceError):
        build_index(rows, out, mode="real", site="inu")

    assert not out.exists()


def test_main_indexes_from_a_files_jsonl(tmp_path, handover, monkeypatch, capsys):
    from src.ingest import index_dicom

    rows, _ = handover
    files_jsonl = tmp_path / "files.jsonl"
    files_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "cli_out"

    monkeypatch.setattr(
        "sys.argv",
        ["index_dicom", "--files", str(files_jsonl), "--out", str(out),
         "--mode", "synthetic", "--site", "inu"],
    )
    rc = index_dicom.main()

    assert rc == 0
    assert (out / "dicom_index.jsonl").exists()
    assert json.loads(capsys.readouterr().out)["n_unique_sop"] == 4


def test_main_returns_nonzero_when_clearance_refuses(tmp_path, handover, monkeypatch, capsys):
    from src.ingest import index_dicom

    rows, _ = handover
    files_jsonl = tmp_path / "files.jsonl"
    files_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(
        "sys.argv",
        ["index_dicom", "--files", str(files_jsonl), "--out", str(tmp_path / "no"),
         "--mode", "real", "--site", "inu"],
    )
    rc = index_dicom.main()

    assert rc == 2
    assert "clearance" in capsys.readouterr().err.lower()


# --- P0.1: module-level clearance-gate hardening ------------------------------------------------


def test_main_requires_mode(tmp_path, handover, monkeypatch, capsys):
    """Fix #1: --mode has no default -- omitting it must fail argument parsing."""
    from src.ingest import index_dicom

    rows, _ = handover
    files_jsonl = tmp_path / "files.jsonl"
    files_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(
        "sys.argv",
        ["index_dicom", "--files", str(files_jsonl), "--out", str(tmp_path / "out"),
         "--site", "inu"],
    )
    with pytest.raises(SystemExit) as ei:
        index_dicom.main()
    assert ei.value.code == 2
    assert "--mode" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_clearance_override_for_tests_flag_is_honoured(tmp_path, handover, monkeypatch):
    """Fix #2: the renamed flag still works, so tests can point real-mode runs at a fixture."""
    from src.ingest import index_dicom

    rows, _ = handover
    files_jsonl = tmp_path / "files.jsonl"
    files_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    marker = tmp_path / "clearance.yaml"
    marker.write_text("data_agreement_executed: true\nip_agreement_executed: true\n")
    out = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        ["index_dicom", "--files", str(files_jsonl), "--out", str(out), "--mode", "real",
         "--site", "inu", "--clearance-override-for-tests", str(marker)],
    )
    rc = index_dicom.main()

    assert rc == 0
    assert out.exists()


def test_main_rejects_the_old_bare_clearance_flag(tmp_path, handover, monkeypatch):
    """The old --clearance flag name must be REJECTED outright, not silently accepted as an
    argparse abbreviation of --clearance-override-for-tests."""
    from src.ingest import index_dicom

    rows, _ = handover
    files_jsonl = tmp_path / "files.jsonl"
    files_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    marker = tmp_path / "clearance.yaml"
    marker.write_text("data_agreement_executed: true\nip_agreement_executed: true\n")

    monkeypatch.setattr(
        "sys.argv",
        ["index_dicom", "--files", str(files_jsonl), "--out", str(tmp_path / "out"),
         "--mode", "real", "--site", "inu", "--clearance", str(marker)],
    )
    with pytest.raises(SystemExit) as ei:
        index_dicom.main()
    assert ei.value.code == 2


def test_main_real_mode_ignores_a_cwd_relative_clearance_marker(tmp_path, handover, monkeypatch,
                                                                capsys):
    """Fix #2: without the override flag, the marker must resolve from the repo root, never from
    a marker that happens to sit at a cwd-relative 'configs/ingest_clearance.yaml'."""
    from src.ingest import index_dicom

    rows, _ = handover
    files_jsonl = tmp_path / "files.jsonl"
    files_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))

    fake_cwd = tmp_path / "fake_cwd"
    (fake_cwd / "configs").mkdir(parents=True)
    (fake_cwd / "configs" / "ingest_clearance.yaml").write_text(
        "data_agreement_executed: true\nip_agreement_executed: true\n")
    monkeypatch.chdir(fake_cwd)

    out = fake_cwd / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["index_dicom", "--files", str(files_jsonl), "--out", str(out), "--mode", "real",
         "--site", "inu"],
    )
    rc = index_dicom.main()

    assert rc == 2
    assert "clearance" in capsys.readouterr().err.lower()
    assert not out.exists()
