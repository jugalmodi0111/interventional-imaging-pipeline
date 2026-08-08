"""The synthetic DICOM fixture IS the dataset under Dialygo B5 -- pin its shape and its PHI.

If these tests drift, every downstream ingest test is asserting against something other than what
a cath-lab XA study actually looks like.
"""
import numpy as np

from tests.fixtures.synthetic_dicom import (
    XA_SOP_CLASS_UID,
    make_xa_dataset,
    write_dataset,
    write_headerless,
)

PHI_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "OtherPatientIDs", "AccessionNumber",
    "InstitutionName", "InstitutionAddress", "ReferringPhysicianName",
    "PerformingPhysicianName", "StudyDescription", "StudyDate", "SeriesDate", "AcquisitionDate",
]

# Must survive de-identification: acquisition geometry + windowing drive the model, and B6 keeps
# vendor identity for leave-one-site-out external validation.
CLINICAL_TAGS = [
    "Modality", "Manufacturer", "ManufacturerModelName", "KVP", "ExposureTime",
    "DistanceSourceToDetector", "DistanceSourceToPatient", "PositionerPrimaryAngle",
    "PositionerSecondaryAngle", "ImagerPixelSpacing", "CineRate", "FrameTime",
    "WindowCenter", "WindowWidth",
]


def test_multiframe_shape_and_phi_present(tmp_path):
    import pydicom

    p = write_dataset(make_xa_dataset(), tmp_path / "IM000001.dcm")
    ds = pydicom.dcmread(p)
    px = ds.pixel_array
    assert px.shape == (8, 64, 64)
    assert px.dtype == np.uint16
    assert int(ds.NumberOfFrames) == 8
    assert ds.SOPClassUID == XA_SOP_CLASS_UID
    for tag in PHI_TAGS:
        assert tag in ds, f"fixture is missing PHI tag {tag} -- de-id tests would pass vacuously"
        assert str(ds[tag].value).strip() != "", f"PHI tag {tag} is empty"
    assert ds.PatientID == "INU-00417"
    assert "INU-00417" in str(ds.PatientName)   # ID embedded in the name, as real exports do


def test_clinical_tags_survive_for_leave_one_site_out(tmp_path):
    import pydicom

    p = write_dataset(make_xa_dataset(manufacturer="Philips"), tmp_path / "IM000002.dcm")
    ds = pydicom.dcmread(p)
    for tag in CLINICAL_TAGS:
        assert tag in ds, f"fixture is missing clinically-required tag {tag}"
    assert ds.Modality == "XA"
    assert ds.Manufacturer == "Philips"                     # Dialygo B6: vendor identity retained
    assert str(ds.ManufacturerModelName).startswith("Philips")
    assert ds.PhotometricInterpretation == "MONOCHROME2"
    assert ds.BitsAllocated == 16 and ds.BitsStored == 12 and ds.HighBit == 11
    assert [float(v) for v in ds.ImagerPixelSpacing] == [0.194, 0.194]


def test_part10_file_has_dicm_magic_at_offset_128(tmp_path):
    p = write_dataset(make_xa_dataset(), tmp_path / "part10.dcm")
    raw = open(p, "rb").read(200)
    assert raw[128:132] == b"DICM"
    assert raw[:128] == b"\x00" * 128


def test_headerless_file_has_no_dicm_magic(tmp_path):
    p = write_headerless(make_xa_dataset(), tmp_path / "CD_IMG_0001")
    raw = open(p, "rb").read(200)
    assert raw[128:132] != b"DICM"
    assert b"DICM" not in raw[:132]
    # First element is (0008,0005) SpecificCharacterSet -> group 0x0008, small element number.
    assert int.from_bytes(raw[0:2], "little") == 0x0008
    assert int.from_bytes(raw[2:4], "little") == 0x0005


def test_headerless_file_still_parses_with_force(tmp_path):
    import pydicom

    p = write_headerless(make_xa_dataset("INU-00419"), tmp_path / "CD_IMG_0002")
    ds = pydicom.dcmread(p, force=True)
    assert ds.PatientID == "INU-00419"
    assert ds.Modality == "XA"


def test_burned_in_paints_band_and_sets_tag(tmp_path):
    import pydicom

    p = write_dataset(make_xa_dataset(burned_in=True), tmp_path / "burned.dcm")
    ds = pydicom.dcmread(p)
    assert ds.BurnedInAnnotation == "YES"
    px = ds.pixel_array
    assert px[:, :8, :].min() >= 3500          # banner painted on EVERY frame
    assert px[:, 8:, :].max() < 3500           # and nowhere else


def test_not_burned_in_has_clean_top_rows(tmp_path):
    import pydicom

    p = write_dataset(make_xa_dataset(burned_in=False), tmp_path / "clean.dcm")
    ds = pydicom.dcmread(p)
    assert ds.BurnedInAnnotation == "NO"
    assert ds.pixel_array[:, :8, :].max() < 3500


def test_supplied_uids_are_honoured(tmp_path):
    import pydicom

    ds = make_xa_dataset(study_uid="1.2.3.4.5", series_uid="1.2.3.4.6", sop_uid="1.2.3.4.7")
    p = write_dataset(ds, tmp_path / "uids.dcm")
    back = pydicom.dcmread(p)
    assert back.StudyInstanceUID == "1.2.3.4.5"
    assert back.SeriesInstanceUID == "1.2.3.4.6"
    assert back.SOPInstanceUID == "1.2.3.4.7"
    assert back.file_meta.MediaStorageSOPInstanceUID == "1.2.3.4.7"


def test_omitted_uids_are_unique_per_call():
    a, b = make_xa_dataset(), make_xa_dataset()
    assert a.StudyInstanceUID != b.StudyInstanceUID
    assert a.SeriesInstanceUID != b.SeriesInstanceUID
    assert a.SOPInstanceUID != b.SOPInstanceUID


def test_pixels_have_vessel_structure(tmp_path):
    import pydicom

    p = write_dataset(make_xa_dataset(n_frames=4, rows=32, cols=32), tmp_path / "vessel.dcm")
    px = pydicom.dcmread(p).pixel_array
    assert px.shape == (4, 32, 32)
    assert px.min() == 200 and px.max() == 3000       # background vs vessel -> windowing has work
    assert px.max() < 4096                            # fits BitsStored=12
    assert not np.array_equal(px[0], px[1])           # vessel moves between frames
