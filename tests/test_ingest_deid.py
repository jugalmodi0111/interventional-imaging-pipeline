"""De-identification key material: salt, pseudonyms, date shifts, UID remapping, crosswalk.

Dialygo B5: this suite never reads real patient data. Identifiers here are invented strings
and every DICOM comes from tests/fixtures/synthetic_dicom.py.

Transcribed from docs/superpowers/plans/2026-08-02-ingest-dicom-pipeline.md Task 7 (pseudonymization
key material) and Task 8 (the tag scrub), with three reconciliations against the implementation as
actually built:

  1. DEIDENTIFICATION_METHOD is a tuple of short strings (multi-valued LO, VR-length safe), not the
     plan's single long string. `ds.DeidentificationMethod` round-trips as a pydicom MultiValue, so
     comparisons go through `list(...)`.
  2. Fixture-literal assertions (ManufacturerModelName="Artis Zee", DistanceSourceToDetector=1200.0)
     are only true when the dataset is built with those values passed as `make_xa_dataset`
     **overrides -- the fixture's own defaults are "<Manufacturer> CathLab Model-1" and 1000.0.
  3. Two of the plan's on-disk-round-trip byte assertions checked literals the fixture never writes
     (b"REDDY", b"Institute of Nephro-Urology"); they are replaced with the PHI the fixture's
     defaults actually put in the file (PatientName, InstitutionName, AccessionNumber, MRN).
"""
import csv
import re
import stat
from datetime import datetime

import pytest

from src.ingest import deid
from src.ingest.deid import (
    DEIDENTIFICATION_METHOD,
    KEEP_TAGS,
    REMOVE_TAGS,
    day_offset,
    deid_dataset,
    load_or_create_salt,
    pseudo_id,
    remap_uid,
    residual_phi,
    shift_date,
    write_crosswalk,
)
from tests.fixtures.synthetic_dicom import make_xa_dataset, write_dataset

SALT_A = b"A" * 32
SALT_B = b"B" * 32
REAL_UID = "1.2.826.0.1.3680043.8.498.1001"


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def _read_back(ds, tmp_path, name="case.dcm"):
    """Round-trip through Part-10 so file_meta exists exactly as it would on the drive."""
    import pydicom

    path = write_dataset(ds, tmp_path / name)
    return pydicom.dcmread(str(path))


# ---------------------------------------------------------------------------
# Task 7, Step 1: load_or_create_salt -- created once, 0600, reused thereafter
# ---------------------------------------------------------------------------


def test_load_or_create_salt_creates_once_and_reuses(tmp_path):
    path = tmp_path / "salt.bin"

    first = load_or_create_salt(path)
    second = load_or_create_salt(path)

    assert isinstance(first, bytes) and len(first) == 32
    assert first == second, "re-running must reuse the salt, never mint a new one"


def test_load_or_create_salt_is_owner_only(tmp_path):
    """0600: the salt plus the crosswalk is what re-identifies the cohort."""
    path = tmp_path / "salt.bin"

    load_or_create_salt(path)

    assert _mode(path) == 0o600


def test_load_or_create_salt_creates_parent_directories(tmp_path):
    path = tmp_path / "keys" / "nested" / "salt.bin"

    salt = load_or_create_salt(path, n_bytes=16)

    assert len(salt) == 16
    assert path.exists()


def test_load_or_create_salt_refuses_a_truncated_salt(tmp_path):
    """A short/empty salt file means something went wrong -- fail loudly, never silently reseed.

    Silently regenerating would produce a *different* pseudonym namespace for the same cohort,
    which is the one failure mode that quietly splits a patient in two.
    """
    path = tmp_path / "salt.bin"
    path.write_bytes(b"\x00\x01")

    with pytest.raises(ValueError, match="salt"):
        load_or_create_salt(path, n_bytes=32)


# ---------------------------------------------------------------------------
# Task 7, Step 2: pseudo_id -- deterministic, site-scoped, salt-sensitive
# ---------------------------------------------------------------------------


def test_pseudo_id_shape_and_determinism():
    first = pseudo_id(SALT_A, "INU-00417", site="inu")
    second = pseudo_id(SALT_A, "INU-00417", site="inu")

    assert first == second, "re-runs and later batches must reproduce the same pseudonym"
    assert re.fullmatch(r"inu_[0-9a-f]{10}", first), first


def test_pseudo_id_differs_across_patients_sites_salts_and_domains():
    base = pseudo_id(SALT_A, "INU-00417", site="inu")

    assert pseudo_id(SALT_A, "INU-00902", site="inu") != base      # different patient
    assert pseudo_id(SALT_A, "INU-00417", site="nims") != base     # different site
    assert pseudo_id(SALT_B, "INU-00417", site="inu") != base      # different salt
    # domain separation: the same text in a different namespace must not collide
    assert pseudo_id(SALT_A, "INU-00417", site="inu", kind="U") != base


def test_pseudo_id_does_not_leak_the_real_id():
    out = pseudo_id(SALT_A, "INU-00417", site="inu")

    assert "00417" not in out
    assert "INU-00417" not in out


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_pseudo_id_refuses_an_empty_identifier(bad):
    """An empty MRN must not silently become one shared bucket that merges distinct patients."""
    with pytest.raises(ValueError, match="real_id"):
        pseudo_id(SALT_A, bad, site="inu")


# ---------------------------------------------------------------------------
# Task 7, Step 3: day_offset -- stable per patient, inside [-364, 0]
# ---------------------------------------------------------------------------


def test_day_offset_is_stable_per_patient():
    assert day_offset(SALT_A, "INU-00417") == day_offset(SALT_A, "INU-00417")


def test_day_offset_is_in_range_for_many_patients():
    """Bounded shift: enough to destroy the calendar date, small enough to keep studies plausible."""
    for i in range(500):
        off = day_offset(SALT_A, f"INU-{i:05d}")
        assert -364 <= off <= 0
        assert isinstance(off, int)


def test_day_offset_varies_across_patients_and_salts():
    offsets = {day_offset(SALT_A, f"INU-{i:05d}") for i in range(200)}

    assert len(offsets) > 100, "offsets must spread, not cluster on a few values"
    # the salt must change the offset for the overwhelming majority of patients
    differs = sum(day_offset(SALT_A, f"INU-{i:05d}") != day_offset(SALT_B, f"INU-{i:05d}")
                  for i in range(200))
    assert differs > 150


# ---------------------------------------------------------------------------
# Task 7, Step 4: shift_date -- intervals preserved, malformed input degrades to ""
# ---------------------------------------------------------------------------


def test_shift_date_preserves_a_31_day_interval():
    """The whole point of shifting rather than deleting: intervals must survive exactly."""
    first = shift_date("20240517", -100)
    follow_up = shift_date("20240617", -100)     # 31 days later

    assert first == "20240207"
    assert follow_up == "20240309"
    delta = datetime.strptime(follow_up, "%Y%m%d") - datetime.strptime(first, "%Y%m%d")
    assert delta.days == 31


def test_shift_date_zero_offset_is_identity():
    assert shift_date("20240517", 0) == "20240517"


def test_shift_date_crosses_a_year_boundary():
    assert shift_date("20240101", -1) == "20231231"


@pytest.mark.parametrize("bad", ["", None, "   ", "2024", "2024-05-17", "20240532",
                                 "notadate", "202405177", "00000000"])
def test_shift_date_returns_empty_string_for_malformed_input(bad):
    """Fail-safe: an unparseable date yields an empty element, never a confident wrong date."""
    assert shift_date(bad, -100) == ""


# ---------------------------------------------------------------------------
# Task 7, Step 5: remap_uid -- deterministic, collision-free, DICOM-legal
# ---------------------------------------------------------------------------


def test_remap_uid_is_deterministic_and_dicom_legal():
    out = remap_uid(SALT_A, REAL_UID)

    assert out == remap_uid(SALT_A, REAL_UID)
    assert out.startswith("2.25."), "2.25.<int> is the DICOM-registered derived-UID root"
    assert len(out) <= 64, "UI has a 64-byte limit"
    assert re.fullmatch(r"[0-9.]+", out), out
    assert not out.endswith(".")
    assert out != REAL_UID


def test_remap_uid_is_collision_free_over_a_realistic_cohort():
    uids = {remap_uid(SALT_A, f"1.2.826.0.1.3680043.8.498.{i}") for i in range(5000)}

    assert len(uids) == 5000


def test_remap_uid_depends_on_salt_and_is_domain_separated():
    assert remap_uid(SALT_B, REAL_UID) != remap_uid(SALT_A, REAL_UID)
    # a UID and a patient id with identical text must not derive related values
    assert remap_uid(SALT_A, "INU-00417") != pseudo_id(SALT_A, "INU-00417", site="inu")


@pytest.mark.parametrize("bad", [None, "", "  "])
def test_remap_uid_refuses_an_empty_uid(bad):
    with pytest.raises(ValueError, match="uid"):
        remap_uid(SALT_A, bad)


# ---------------------------------------------------------------------------
# Task 7, Step 6: write_crosswalk -- 0600, round-trips, never in the repo
# ---------------------------------------------------------------------------


def test_write_crosswalk_is_owner_only_and_round_trips(tmp_path):
    path = tmp_path / "crosswalk.csv"
    mapping = {
        "INU-00417": pseudo_id(SALT_A, "INU-00417", site="inu"),
        "INU-00902": pseudo_id(SALT_A, "INU-00902", site="inu"),
    }

    written = write_crosswalk(path, mapping)

    assert written == path
    assert _mode(path) == 0o600, "salt + crosswalk together re-identify the cohort"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["real_id", "pseudo_id"]
    assert dict(rows[1:]) == mapping


def test_write_crosswalk_is_sorted_and_rewritable(tmp_path):
    path = tmp_path / "crosswalk.csv"

    write_crosswalk(path, {"B": "inu_2", "A": "inu_1"})
    write_crosswalk(path, {"A": "inu_1", "B": "inu_2", "C": "inu_3"})

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert [r[0] for r in rows] == ["real_id", "A", "B", "C"], "deterministic, no duplicate rows"
    assert _mode(path) == 0o600, "rewriting must not widen the mode"


# ---------------------------------------------------------------------------
# Task 7, Step 7: main() -- provision key material as `python -m src.ingest.deid`
#
# Ruling 5: --salt defaults to "_keys/salt.bin" rather than being required=True as the plan wrote.
# Both tests here pass --salt explicitly, so that CLI reconciliation does not change their behavior.
# ---------------------------------------------------------------------------


def test_main_provisions_the_salt_and_reports_a_fingerprint(tmp_path, monkeypatch, capsys):
    import json

    salt_path = tmp_path / "keys" / "salt.bin"
    monkeypatch.setattr("sys.argv", ["deid", "--salt", str(salt_path), "--site", "inu"])

    rc = deid.main()

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert salt_path.exists() and _mode(salt_path) == 0o600
    assert out["salt_path"] == str(salt_path)
    assert re.fullmatch(r"[0-9a-f]{16}", out["salt_fingerprint"])
    assert out["created"] is True
    # the fingerprint identifies the key without revealing any of it
    assert out["salt_fingerprint"] not in salt_path.read_bytes().hex()


def test_main_is_idempotent_and_reports_the_same_fingerprint(tmp_path, monkeypatch, capsys):
    import json

    salt_path = tmp_path / "salt.bin"
    monkeypatch.setattr("sys.argv", ["deid", "--salt", str(salt_path), "--site", "inu"])
    deid.main()
    first = json.loads(capsys.readouterr().out)
    deid.main()
    second = json.loads(capsys.readouterr().out)

    assert first["salt_fingerprint"] == second["salt_fingerprint"]
    assert second["created"] is False


# ---------------------------------------------------------------------------
# Task 8, Step 1: REMOVE_TAGS / KEEP_TAGS and residual_phi flags a dirty dataset
# ---------------------------------------------------------------------------


def test_remove_and_keep_tags_do_not_overlap():
    assert not (set(REMOVE_TAGS) & set(KEEP_TAGS))
    # B6: vendor identity is retained, never scrubbed.
    assert "Manufacturer" in KEEP_TAGS and "Manufacturer" not in REMOVE_TAGS
    assert "ManufacturerModelName" in KEEP_TAGS
    # PatientID is replaced with the pseudonym, not emptied, so it is in neither list.
    assert "PatientID" not in REMOVE_TAGS and "PatientID" not in KEEP_TAGS


def test_series_description_is_scrubbed_not_kept():
    """Free text typed at the console leaks names exactly the way StudyDescription does.

    The PHI audit that gates the human --ack-phi-audit sign-off lists SeriesDescription among
    the identifiers it promises are scrubbed. Resolved in favour of the stronger claim: the
    tag is emptied, so the promise the reviewer signs off on is true.
    """
    assert "SeriesDescription" in REMOVE_TAGS
    assert "SeriesDescription" not in KEEP_TAGS


def test_residual_phi_flags_a_series_description(tmp_path):
    """In REMOVE_TAGS means the pre-write residual gate quarantines it if it ever survives."""
    ds = make_xa_dataset("INU-00417", SeriesDescription="AVF RUN 3 REDDY")

    assert "SeriesDescription" in residual_phi(ds)


def test_series_description_is_emptied_by_the_scrub(tmp_path):
    ds = _read_back(make_xa_dataset("INU-00417", SeriesDescription="AVF RUN 3 REDDY^SURESH"),
                    tmp_path)

    clean, _ = deid_dataset(ds, SALT_A, site="inu")

    assert str(clean.SeriesDescription) == ""
    assert residual_phi(clean) == []


def test_series_description_text_is_gone_from_the_file_on_disk(tmp_path):
    """The byte-level check: nothing typed into the series label reaches the clean drive."""
    import pydicom

    ds = _read_back(make_xa_dataset("INU-00417", SeriesDescription="AVF RUN 3 REDDY"), tmp_path)

    clean, _ = deid_dataset(ds, SALT_A, site="inu")
    out = tmp_path / "clean.dcm"
    clean.save_as(str(out), enforce_file_format=True)

    assert b"REDDY" not in out.read_bytes()
    assert str(pydicom.dcmread(str(out)).SeriesDescription) == ""


def test_residual_phi_flags_an_untouched_dataset():
    ds = make_xa_dataset("INU-00417")

    dirty = residual_phi(ds)

    assert "PatientName" in dirty
    assert "InstitutionName" in dirty
    assert "ReferringPhysicianName" in dirty
    assert "AccessionNumber" in dirty
    assert "StudyDescription" in dirty


def test_residual_phi_ignores_empty_and_whitespace_values():
    ds = make_xa_dataset("INU-00417")
    ds.PatientName = ""
    ds.InstitutionName = "   "

    dirty = residual_phi(ds)

    assert "PatientName" not in dirty
    assert "InstitutionName" not in dirty


# ---------------------------------------------------------------------------
# Task 8, Step 2: deid_dataset empties every identifying element and reports clean
# ---------------------------------------------------------------------------


def test_deid_dataset_empties_every_identifying_element(tmp_path):
    from pydicom.datadict import tag_for_keyword

    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)
    assert residual_phi(ds), "precondition: the fixture is dirty"

    clean, ids = deid_dataset(ds, SALT_A, site="inu")

    for kw in REMOVE_TAGS:
        tag = tag_for_keyword(kw)
        if tag in clean:
            assert str(clean[tag].value or "").strip() == "", f"{kw} survived the scrub"
    assert residual_phi(clean) == []
    assert clean.PatientIdentityRemoved == "YES"
    # Ruling 3: DEIDENTIFICATION_METHOD is a tuple of short strings (multi-valued LO), not the
    # plan's single 248-char string -- pydicom returns a MultiValue, so compare through list().
    assert list(clean.DeidentificationMethod) == list(DEIDENTIFICATION_METHOD)
    assert set(ids) == {"real_patient", "pseudo_patient", "pseudo_study",
                        "pseudo_series", "pseudo_sop"}


def test_deid_dataset_refuses_an_instance_with_no_patient_id(tmp_path):
    """Fail-safe: no PatientID means no stable pseudonym and no stable date offset."""
    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)
    ds.PatientID = ""

    with pytest.raises(ValueError, match="PatientID"):
        deid_dataset(ds, SALT_A, site="inu")


# ---------------------------------------------------------------------------
# Task 8, Step 3: PatientID becomes the pseudonym and ids matches the key material
# ---------------------------------------------------------------------------


def test_patient_id_is_replaced_with_the_pseudonym(tmp_path):
    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)

    clean, ids = deid_dataset(ds, SALT_A, site="inu")

    expected = pseudo_id(SALT_A, "INU-00417", site="inu")
    assert clean.PatientID == expected
    assert ids["real_patient"] == "INU-00417"
    assert ids["pseudo_patient"] == expected
    assert "INU-00417" not in str(clean.PatientID)


def test_the_same_patient_pseudonymises_identically_in_a_later_batch(tmp_path):
    """Second handover, separate run: the pseudonym must match so the batches merge."""
    first, ids_1 = deid_dataset(_read_back(make_xa_dataset("INU-00417"), tmp_path, "a.dcm"),
                                SALT_A, site="inu")
    second, ids_2 = deid_dataset(_read_back(make_xa_dataset("INU-00417"), tmp_path, "b.dcm"),
                                 SALT_A, site="inu")

    assert first.PatientID == second.PatientID
    assert ids_1["pseudo_patient"] == ids_2["pseudo_patient"]


# ---------------------------------------------------------------------------
# Task 8, Step 4: clinically required tags survive -- including vendor identity (B6)
#
# Ruling 1: ManufacturerModelName="Artis Zee" and DistanceSourceToDetector=1200.0 are not the
# fixture's defaults ("<Manufacturer> CathLab Model-1" and 1000.0) -- passed as overrides so the
# plan's literal assertions stay verbatim.
# ---------------------------------------------------------------------------


def test_clinically_required_tags_survive_the_scrub(tmp_path):
    ds = _read_back(make_xa_dataset("INU-00417", ManufacturerModelName="Artis Zee",
                                    DistanceSourceToDetector=1200.0), tmp_path)

    clean, _ = deid_dataset(ds, SALT_A, site="inu")

    # Dialygo B6: Manufacturer and ManufacturerModelName are RETAINED on purpose. Vendor identity
    # is not patient identity, and leave-one-site-out external validation is impossible without
    # knowing which C-arm produced each study.
    assert clean.Manufacturer == "Siemens"
    assert clean.ManufacturerModelName == "Artis Zee"

    assert clean.Modality == "XA"
    assert float(clean.KVP) == 70.0
    assert float(clean.DistanceSourceToDetector) == 1200.0
    assert int(clean.CineRate) == 15
    assert int(clean.WindowCenter) == 2048
    assert int(clean.WindowWidth) == 4096
    assert int(clean.BitsAllocated) == 16
    assert int(clean.BitsStored) == 12
    assert clean.PhotometricInterpretation == "MONOCHROME2"


def test_pixel_data_is_untouched_by_the_tag_scrub(tmp_path):
    """The scrub is tag-level only. Pixel-domain screening is a separate, mandatory phase."""
    ds = _read_back(make_xa_dataset("INU-00417", n_frames=4, rows=64, cols=64), tmp_path)
    before = bytes(ds.PixelData)

    clean, _ = deid_dataset(ds, SALT_A, site="inu")

    assert bytes(clean.PixelData) == before
    assert int(clean.Rows) == 64 and int(clean.Columns) == 64


# ---------------------------------------------------------------------------
# Task 8, Step 5: UID remapping preserves study/series structure
# ---------------------------------------------------------------------------


def test_uid_remap_preserves_study_and_series_relationships(tmp_path):
    """Two instances of one series must still share a study and a series after remapping."""
    common = {"study_uid": "1.2.826.0.1.3680043.8.498.1001",
              "series_uid": "1.2.826.0.1.3680043.8.498.2001"}
    ds_1 = _read_back(make_xa_dataset("INU-00417", sop_uid="1.2.3.4.5.1", **common),
                      tmp_path, "i1.dcm")
    ds_2 = _read_back(make_xa_dataset("INU-00417", sop_uid="1.2.3.4.5.2", **common),
                      tmp_path, "i2.dcm")

    clean_1, ids_1 = deid_dataset(ds_1, SALT_A, site="inu")
    clean_2, ids_2 = deid_dataset(ds_2, SALT_A, site="inu")

    assert clean_1.StudyInstanceUID == clean_2.StudyInstanceUID
    assert clean_1.SeriesInstanceUID == clean_2.SeriesInstanceUID
    assert clean_1.SOPInstanceUID != clean_2.SOPInstanceUID
    assert ids_1["pseudo_study"] == ids_2["pseudo_study"]
    assert ids_1["pseudo_series"] == ids_2["pseudo_series"]
    assert ids_1["pseudo_sop"] != ids_2["pseudo_sop"]


def test_remapped_uids_differ_from_the_originals(tmp_path):
    ds = _read_back(make_xa_dataset("INU-00417",
                                    study_uid="1.2.826.0.1.3680043.8.498.1001",
                                    series_uid="1.2.826.0.1.3680043.8.498.2001",
                                    sop_uid="1.2.3.4.5.1"), tmp_path)

    clean, ids = deid_dataset(ds, SALT_A, site="inu")

    assert str(clean.StudyInstanceUID) != "1.2.826.0.1.3680043.8.498.1001"
    assert str(clean.SeriesInstanceUID) != "1.2.826.0.1.3680043.8.498.2001"
    assert str(clean.SOPInstanceUID) != "1.2.3.4.5.1"
    for key in ("pseudo_study", "pseudo_series", "pseudo_sop"):
        assert ids[key].startswith("2.25.") and len(ids[key]) <= 64


# ---------------------------------------------------------------------------
# Task 8, Step 6: dates shifted, the 31-day interval preserved, the original gone
# ---------------------------------------------------------------------------


def test_dates_are_shifted_with_intervals_preserved(tmp_path):
    """Index study and a 31-day follow-up for one patient: the interval must survive exactly."""
    index_ds = _read_back(make_xa_dataset("INU-00417", study_date="20240517",
                                          study_uid="1.2.826.0.1.3680043.8.498.1001",
                                          sop_uid="1.2.3.4.5.1"), tmp_path, "idx.dcm")
    follow_ds = _read_back(make_xa_dataset("INU-00417", study_date="20240617",
                                           study_uid="1.2.826.0.1.3680043.8.498.1002",
                                           sop_uid="1.2.3.4.5.2"), tmp_path, "fup.dcm")

    clean_index, _ = deid_dataset(index_ds, SALT_A, site="inu")
    clean_follow, _ = deid_dataset(follow_ds, SALT_A, site="inu")

    assert str(clean_index.StudyDate) != "20240517", "the absolute date must not survive"
    assert str(clean_follow.StudyDate) != "20240617"

    delta = (datetime.strptime(str(clean_follow.StudyDate), "%Y%m%d")
             - datetime.strptime(str(clean_index.StudyDate), "%Y%m%d"))
    assert delta.days == 31, "three months vs three years is the clinical question"

    offset = day_offset(SALT_A, "INU-00417")
    assert str(clean_index.StudyDate) == shift_date("20240517", offset)
    assert str(clean_index.SeriesDate) == shift_date("20240517", offset)
    assert str(clean_index.AcquisitionDate) == shift_date("20240517", offset)


def test_times_are_emptied_and_birth_date_removed(tmp_path):
    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)
    assert str(ds.StudyTime).strip() != ""
    assert str(ds.PatientBirthDate) == "19631104"

    clean, _ = deid_dataset(ds, SALT_A, site="inu")

    assert str(clean.StudyTime).strip() == ""
    assert str(clean.PatientBirthDate).strip() == ""


def test_two_patients_get_different_shifts(tmp_path):
    """Per-patient offsets stop anyone aligning timelines to recover a calendar date."""
    a = _read_back(make_xa_dataset("INU-00417", study_date="20240517"), tmp_path, "a.dcm")
    b = _read_back(make_xa_dataset("INU-00902", study_date="20240517"), tmp_path, "b.dcm")

    clean_a, _ = deid_dataset(a, SALT_A, site="inu")
    clean_b, _ = deid_dataset(b, SALT_A, site="inu")

    assert str(clean_a.StudyDate) != str(clean_b.StudyDate)


# ---------------------------------------------------------------------------
# Task 8, Step 7: private tags and overlay planes are destroyed
# ---------------------------------------------------------------------------


def test_private_tag_and_overlay_plane_are_removed(tmp_path):
    """Private elements can hold anything; an overlay is a bitmap a tag scrub would never reach."""
    from pydicom.tag import Tag

    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)
    ds.add_new(Tag(0x0009, 0x0010), "LO", "ACME PRIVATE CREATOR")
    ds.add_new(Tag(0x6000, 0x3000), "OW", b"\x00\x01\x02\x03")
    assert Tag(0x0009, 0x0010) in ds and Tag(0x6000, 0x3000) in ds

    clean, _ = deid_dataset(ds, SALT_A, site="inu")

    assert Tag(0x0009, 0x0010) not in clean
    assert Tag(0x6000, 0x3000) not in clean


def test_all_overlay_groups_are_swept_not_just_6000(tmp_path):
    from pydicom.tag import Tag

    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)
    for group in (0x6000, 0x6002, 0x601E):
        ds.add_new(Tag(group, 0x3000), "OW", b"\x00\x01")
        ds.add_new(Tag(group, 0x0010), "US", 64)

    clean, _ = deid_dataset(ds, SALT_A, site="inu")

    assert not [t for t in clean.keys() if 0x6000 <= t.group <= 0x601F]


# ---------------------------------------------------------------------------
# Task 8, Step 8: file_meta SOP UID matches the dataset, and the file survives a round-trip
#
# Ruling 1: ManufacturerModelName="Artis Zee" is an override, not the fixture default.
# Ruling 2 (audit P1): the plan's b"REDDY" / b"Institute of Nephro-Urology" asserts checked byte
# strings the fixture never writes. Replaced with the PHI literals the fixture's own defaults put
# into the file: PatientName ("SYNTHETIC^INU-00417"), InstitutionName ("Synthetic Regional Dialysis
# Centre"), AccessionNumber ("ACC-2024-0517-33") and OtherPatientIDs ("MRN-88213").
# ---------------------------------------------------------------------------


def test_file_meta_sop_uid_is_updated_to_the_remapped_uid(tmp_path):
    """A stale MediaStorageSOPInstanceUID leaks the original UID and desynchronises the file."""
    ds = _read_back(make_xa_dataset("INU-00417", sop_uid="1.2.3.4.5.1"), tmp_path)
    assert str(ds.file_meta.MediaStorageSOPInstanceUID) == "1.2.3.4.5.1"

    clean, ids = deid_dataset(ds, SALT_A, site="inu")

    assert str(clean.file_meta.MediaStorageSOPInstanceUID) == str(clean.SOPInstanceUID)
    assert str(clean.file_meta.MediaStorageSOPInstanceUID) == ids["pseudo_sop"]
    assert str(clean.file_meta.MediaStorageSOPInstanceUID) != "1.2.3.4.5.1"


def test_deidentified_file_round_trips_clean_on_disk(tmp_path):
    """End-to-end: what actually lands on the drive must read back with no residual PHI."""
    import pydicom

    ds = _read_back(make_xa_dataset("INU-00417", sop_uid="1.2.3.4.5.1",
                                    ManufacturerModelName="Artis Zee"), tmp_path)
    clean, ids = deid_dataset(ds, SALT_A, site="inu")

    out = tmp_path / "deid.dcm"
    clean.save_as(str(out))
    reloaded = pydicom.dcmread(str(out))

    assert residual_phi(reloaded) == []
    assert reloaded.PatientID == ids["pseudo_patient"]
    assert reloaded.PatientIdentityRemoved == "YES"
    assert str(reloaded.file_meta.MediaStorageSOPInstanceUID) == str(reloaded.SOPInstanceUID)
    # B6: vendor identity is still there after the round-trip.
    assert reloaded.Manufacturer == "Siemens"
    assert reloaded.ManufacturerModelName == "Artis Zee"
    # nothing from the original identity remains anywhere in the file bytes -- these are the
    # fixture's OWN default PHI literals (audit P1: the plan checked for byte strings the fixture
    # never wrote, which made the assertion vacuous).
    raw = out.read_bytes()
    assert b"SYNTHETIC^INU-00417" not in raw     # original PatientName
    assert b"INU-00417" not in raw               # original PatientID
    assert b"Synthetic Regional Dialysis Centre" not in raw   # original InstitutionName
    assert b"ACC-2024-0517-33" not in raw        # original AccessionNumber
    assert b"MRN-88213" not in raw               # original OtherPatientIDs
