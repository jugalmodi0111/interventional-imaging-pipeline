"""Synthetic multi-frame XA (fistulography) DICOM builder.

Dialygo B5: no real patient data may be processed until the institutional agreement executes, so
this module is the ONLY source of DICOM for the ingest test suite. It deliberately reproduces the
three things that break real ingests:

  1. Part-10 vs headerless. Vendor CD dumps frequently write the raw dataset with NO 128-byte
     preamble and no (0002,....) file-meta group, so extension- or preamble-based detection misses
     them. write_dataset() emits Part-10; write_headerless() emits the CD-style variant.
  2. Burned-in annotation. Cath labs stamp patient name/date across the top of the image. That is
     PHI in the PIXELS, which tag-level de-identification cannot touch. burned_in=True paints an
     8-row band at max intensity on every frame and sets BurnedInAnnotation="YES".
  3. 12-bit data in 16-bit words (BitsStored=12, BitsAllocated=16, MONOCHROME2), which is what
     makes WindowCenter/WindowWidth load-bearing.

PHI tags and clinically-required tags are both populated so de-identification tests can assert
BOTH directions: PHI must be gone, acquisition geometry must remain. Per Dialygo B6, Manufacturer
and ManufacturerModelName are clinical, not identifying -- leave-one-site-out external validation
needs vendor identity.

Pixel encoding is Explicit VR Little Endian (uncompressed) on purpose: the suite must never depend
on a compiled JPEG plugin being present.
"""
import numpy as np

XA_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.12.1"          # X-Ray Angiographic Image Storage
IMPLEMENTATION_CLASS_UID = "1.2.826.0.1.3680043.8.498.1"   # synthetic implementation identity

_BACKGROUND = 200      # dim myocardium/soft tissue
_VESSEL = 3000         # contrast-filled fistula
_VESSEL_EDGE = 2600    # partial-volume edge so thresholds are not trivially bimodal
_BANNER = 4000         # burned-in annotation, near the 12-bit ceiling (4095)
_BANNER_ROWS = 8


def _pixels(n_frames, rows, cols, burned_in):
    """(n_frames, rows, cols) uint16 with a bright diagonal 'vessel' that shifts per frame."""
    arr = np.full((n_frames, rows, cols), _BACKGROUND, dtype=np.uint16)
    for f in range(n_frames):
        for i in range(rows):
            j = (i + 2 * f) % cols
            arr[f, i, j] = _VESSEL
            arr[f, i, (j + 1) % cols] = _VESSEL_EDGE
    if burned_in:
        arr[:, :_BANNER_ROWS, :] = _BANNER
    return arr


def make_xa_dataset(patient_id="INU-00417", *, n_frames=8, rows=64, cols=64, burned_in=False,
                    study_uid=None, series_uid=None, sop_uid=None, modality="XA",
                    manufacturer="Siemens", study_date="20240517", **overrides):
    """Build an in-memory multi-frame XA dataset with PHI + clinical tags + 12-bit pixels.

    UIDs default to freshly generated unique values; pass study_uid/series_uid/sop_uid to force a
    grouping (e.g. two instances that must land in the same series).

    **overrides: after the dataset is fully built (all default tags + PixelData set), apply
    arbitrary DICOM keyword attributes. For example: PatientName="REDDY^SURESH^^Mr",
    ManufacturerModelName="Artis Zee", DistanceSourceToDetector=1200.0.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    sop_uid = sop_uid or generate_uid()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = XA_SOP_CLASS_UID
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID

    ds = Dataset()
    ds.file_meta = meta
    ds.preamble = b"\x00" * 128

    # --- lowest tag in the dataset; the headerless detector keys off group 0x0008 ---
    ds.SpecificCharacterSet = "ISO_IR 100"

    # --- identity / SOP ---
    ds.SOPClassUID = XA_SOP_CLASS_UID
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    # --- PHI: every one of these must be gone after de-identification ---
    ds.PatientName = f"SYNTHETIC^{patient_id}"
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19631104"
    ds.PatientSex = "M"
    ds.OtherPatientIDs = "MRN-88213"
    ds.AccessionNumber = "ACC-2024-0517-33"
    ds.InstitutionName = "Synthetic Regional Dialysis Centre"
    ds.InstitutionAddress = "12 Nowhere Road, Testville"
    ds.ReferringPhysicianName = "REF^PHYSICIAN"
    ds.PerformingPhysicianName = "PERF^PHYSICIAN"
    ds.StudyDescription = "FISTULOGRAM LEFT BRACHIOCEPHALIC AVF"
    ds.SeriesDescription = "AVF RUN 3"
    ds.StudyDate = study_date
    ds.SeriesDate = study_date
    ds.AcquisitionDate = study_date
    ds.StudyTime = "101500"
    ds.SeriesTime = "101532"

    # --- clinical: must SURVIVE de-identification (B6 keeps vendor identity) ---
    ds.Modality = modality
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = f"{manufacturer} CathLab Model-1"
    ds.KVP = 70.0
    ds.ExposureTime = 8
    ds.DistanceSourceToDetector = 1000.0
    ds.DistanceSourceToPatient = 750.0
    ds.PositionerPrimaryAngle = -30.0
    ds.PositionerSecondaryAngle = 15.0
    ds.ImagerPixelSpacing = [0.194, 0.194]
    ds.CineRate = 15
    ds.FrameTime = 66.67
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096

    # --- pixels: 12-bit stored in 16-bit words, MONOCHROME2 ---
    arr = _pixels(n_frames, rows, cols, burned_in)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = n_frames
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.BurnedInAnnotation = "YES" if burned_in else "NO"
    ds.PixelData = arr.astype("<u2").tobytes()

    # --- apply overrides ---
    for k, v in overrides.items():
        setattr(ds, k, v)

    return ds


def write_dataset(ds, path):
    """Write a conformant Part-10 file: 128-byte preamble, b'DICM' at offset 128, file-meta group."""
    p = str(path)
    ds.save_as(p, enforce_file_format=True)
    return p


def write_headerless(ds, path):
    """Write the RAW dataset the way vendor CD exports do: no preamble, no (0002,....) group.

    Copies the main dataset elements into a bare Dataset (file_meta is a separate attribute and is
    therefore dropped), then writes Explicit VR Little Endian with no wrapper. The resulting file
    has no b'DICM' magic anywhere, so extension- and preamble-based detection both miss it.
    """
    from pydicom.dataset import Dataset

    bare = Dataset()
    for elem in ds:                      # tag order; file_meta is NOT part of this iteration
        bare.add(elem)
    bare.preamble = None
    p = str(path)
    bare.save_as(p, enforce_file_format=False, implicit_vr=False, little_endian=True)
    return p
