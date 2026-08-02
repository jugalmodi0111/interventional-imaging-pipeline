# DICOM Ingest Pipeline (`src/ingest/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/ingest/` — a resumable, phase-separated pipeline that turns a scattered external drive of institutional fistulography (raw DICOM, vendor CD dumps, exported video/images) into a de-identified, pseudonymized, patient-grouped PNG frame store symlinked into `data/raw/avf_fistulography/`.

**Architecture:** Five sequential phases, each its own module with a `python -m src.ingest.<phase>` CLI, each idempotent and resumable, each consuming the previous phase's on-disk artifact: **scan** (read-only walk + magic-byte typing) → **index** (DICOM headers + PHI audit) → **deid** (tag scrub, UID remap, date shift, pixel-overlay masking) → **extract** (VOI-LUT windowed PNG frames + sidecars) → **link** (symlink clean tree into the repo). Supporting modules: `manifest` (hashing/provenance/resume state), `clearance` (B5/B9 legal gate enforced in code), `labels` (pluggable clinician-label adapters), `doctor` (health check). Originals on the drive are never modified.

**Tech Stack:** Python 3.12, pydicom (headers + pixel data), pylibjpeg (compressed XA transfer syntaxes), numpy, OpenCV (video decode, morphology, PNG write), PyYAML, pytest.

**Source spec:** [`Dialygo_Orientation_and_Requirements.md`](../../Dialygo_Orientation_and_Requirements.md) · **Plan delta:** [`2026-08-01-dialygo-realignment.md`](2026-08-01-dialygo-realignment.md) task **T1.7**.

## Global Constraints

These apply to **every** task. Copied verbatim from the Dialygo requirements and the repo's own conventions.

- **B5 (hard gate):** No patient data — identifiable *or* de-identified — may be processed outside the institutional agreement. Until it is executed, development uses **public, synthetic, or the engineer's own non-patient data only**. Every task here is built and tested against **synthetic DICOM generated in-process**. No real fistulography touches this machine.
- **B9 (hard gate):** The IP/engagement agreement must be executed before development on real data begins.
- **B5 (split rule):** Data is split **by patient, never by image**. No patient may appear in both training and testing.
- **B3:** Input is a single still angiographic frame (PNG), de-identified, cropped to the segment of interest. Output must carry a calibrated confidence and default to uncertain/refer when low.
- **B6:** External validation at ≥1 non-source site is a first-class deliverable — so vendor/model DICOM tags are **retained**. Vendor identity is not patient identity.
- **B7:** The engineer does not define what counts as "abnormal." Label semantics come from the clinical lead.
- **Repo convention:** Modules import **torch-free and cv2-free at module level**; heavy imports go inside functions (mirrors `src/data_prep/autolabel_gdino.py`, `src/serve/orchestrator.py`).
- **Repo convention:** Pure helpers are unit-tested; every module exposes `main()` and runs as `python -m src.ingest.<module>`.
- **Repo convention:** Fail-safe defaults — malformed or missing input degrades to "defer / refuse", never to a confident-looking success (mirrors `src/serve/registry.py:69-76`, `src/serve/diagnosis.py`).
- **Test command:** `python -m pytest tests/ -q` from the repo root. The suite must stay green (374 passing as of commit `f1eb255`).
- **Commit convention:** Conventional Commits — `feat(ingest): …`, `test(ingest): …`, `chore(deps): …`.

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `src/ingest/__init__.py` | Package marker. |
| `src/ingest/clearance.py` | B5/B9 legal gate. Refuses real-data runs without an executed-agreement marker. |
| `src/ingest/manifest.py` | Content hashing, provenance records, atomic JSONL, resume state. |
| `src/ingest/scan.py` | Phase 1. Read-only drive walk, magic-byte file typing, resumable checkpoints. |
| `src/ingest/index_dicom.py` | Phase 2. Header extraction, patient→study→series hierarchy, SOP dedupe, PHI audit. |
| `src/ingest/deid.py` | Phase 3a. Pseudonymization, tag scrub, UID remap, date shift, crosswalk. |
| `src/ingest/pixel_deid.py` | Phase 3b. Burned-in overlay text detection + masking, QA flagging. |
| `src/ingest/extract.py` | Phase 4. VOI-LUT windowing, multi-frame → PNG, sidecar JSON, video path. |
| `src/ingest/labels.py` | Clinician-label adapters (CSV / COCO / mask dir) + join with unmatched reporting. |
| `src/ingest/link.py` | Phase 5. Symlink clean tree into `data/raw/`. |
| `src/ingest/doctor.py` | Health check: mount, links, manifest, PHI-leak scan. |
| `configs/ingest_sites.yaml` | Per-site config: site code, drive roots, clean-tree root. |
| `configs/ingest_clearance.yaml` | B5/B9 marker (both flags `false` until the agreements execute). |
| `tests/fixtures/synthetic_dicom.py` | Synthetic PHI-bearing multi-frame XA DICOM factory — the only "data" this plan touches. |
| `tests/test_ingest_*.py` | One test module per source module. |

**Modify:**

| File | Change |
|---|---|
| `requirements.txt` | Add `pydicom>=3.0`, `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-rle`, `openpyxl`. |
| `src/data_prep/io_utils.py:18-34` | Add AVF stem regex to `group_key` — **the leakage guard** (Task 12). |
| `.gitignore` | Add `.ingest/`, `*.crosswalk.csv`, `*.salt`, `salt.bin`. |
| `Makefile` | Add `ingest-scan`, `ingest-index`, `ingest-deid`, `ingest-extract`, `ingest-link`, `ingest-doctor`. |
| `docs/PROJECT_TRACKER.md` | Tick T1.7, add changelog entry. |

**Canonical on-disk shapes** — every task emits exactly these:

```
<drive>/intv-img-clean/<site>/
  dicom/<pseudo_patient>/<pseudo_study>/<pseudo_series>/<pseudo_sop>.dcm
  frames/<stem_prefix>/f00000.png          # stem_prefix = avf_<site>_<pid>_s<NN>
  sidecar/<stem_prefix>.json
  _keys/salt.bin                            # 0600, NEVER in the repo
  _keys/crosswalk.csv                       # 0600, NEVER in the repo
  _manifest/{scan,index,deid,extract}.jsonl

<repo>/.ingest/<site>/                      # gitignored working state
  files.jsonl  scan_state.json  dicom_index.jsonl  phi_audit.md  qa_review.jsonl

<repo>/data/raw/avf_fistulography -> <drive>/intv-img-clean/<site>/frames   # symlink
```

**Frame stem grammar** (locked; Task 12 enforces it in `io_utils.group_key`):

```
avf_<site>_<pseudo_patient_hex10>_s<series:02d>_<frame:05d>
e.g.  avf_inu_3f9c21b04e_s01_00012     ->  group key: avf_inu_3f9c21b04e
```

---
### Task 1: Dependencies, `src/ingest/` skeleton, and the synthetic DICOM fixture

Everything downstream of this task is tested against one object: a multi-frame XA DICOM built in
memory by `tests/fixtures/synthetic_dicom.py`. Under **Dialygo B5** no real patient study may be
opened until the institutional agreement executes, so this fixture *is* the dataset for the whole
ingest pipeline — not a stand-in for it, not a smoke test before the real files arrive. Every
assertion in Tasks 2–4 (and every ingest task after them) reads bytes that this module wrote. That
means the fixture has to be honest about the things that actually break real ingests: DICOM stored
without a Part-10 preamble (vendor CD dumps), burned-in patient banners across the top of the
frame, 12-bit pixels stored in 16-bit words, and a mix of PHI tags next to acquisition-geometry
tags that must *survive* de-identification.

Per **Dialygo B6**, `Manufacturer` and `ManufacturerModelName` are deliberately populated and will
be deliberately retained downstream: leave-one-site-out external validation needs to know which
vendor's C-arm produced a frame. Vendor identity is not patient identity.

`pylibjpeg` and its two decoder plugins are added now rather than later because cath-lab cine is
almost never stored uncompressed — JPEG-Lossless (1.2.840.10008.1.2.4.57/70) and JPEG-LS are the
common transfer syntaxes, and pydicom cannot decode either without the plugins. The synthetic
fixture writes Explicit VR Little Endian (no plugin needed) so the test suite never depends on a
compiled decoder, but the runtime dependency must be declared before Phase 2 pixel extraction.

**Files:**
- Modify: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/requirements.txt`
- Modify: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/.gitignore`
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/__init__.py`
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/fixtures/__init__.py`
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/fixtures/synthetic_dicom.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_fixture.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `tests.fixtures.synthetic_dicom.make_xa_dataset(patient_id="INU-00417", *, n_frames=8, rows=64, cols=64, burned_in=False, study_uid=None, series_uid=None, sop_uid=None, modality="XA", manufacturer="Siemens", study_date="20240517") -> pydicom.dataset.Dataset`
  - `tests.fixtures.synthetic_dicom.write_dataset(ds, path) -> str` (Part-10: 128-byte preamble + `b"DICM"` at offset 128)
  - `tests.fixtures.synthetic_dicom.write_headerless(ds, path) -> str` (raw dataset, no preamble, no file-meta group)
  - `tests.fixtures.synthetic_dicom.XA_SOP_CLASS_UID` (str)

- [ ] **Step 1: Declare the new dependencies**

Append to the end of `requirements.txt`:

```
# ingest (Dialygo Phase 1: institutional DICOM handover -> de-identified frame store)
pydicom>=3.0              # 3.x save_as(enforce_file_format=...) API is required by the fixture
pylibjpeg                 # plugin host: supplies the JPEG-Lossless / JPEG-LS decoders cath-lab cine needs
pylibjpeg-libjpeg         # JPEG Baseline/Extended/Lossless (1.2.840.10008.1.2.4.50/51/57/70)
pylibjpeg-rle             # RLE Lossless (1.2.840.10008.1.2.5)
openpyxl                  # vendor handovers ship the patient worksheet as .xlsx
```

- [ ] **Step 2: Install them**

Run: `python -m pip install "pydicom>=3.0" pylibjpeg pylibjpeg-libjpeg pylibjpeg-rle openpyxl`
Expected: `Successfully installed pydicom-3.x ... openpyxl-3.x`
Verify: `python -c "import pydicom; print(pydicom.__version__)"` prints a version `>= 3.0`.

- [ ] **Step 3: Ignore ingest working state and every re-identification vector**

Append to the end of `.gitignore`:

```
# Dialygo ingest: scan/de-id working state, plus anything that could re-identify a patient
.ingest/
*.crosswalk.csv
*.salt
salt.bin
```

`.ingest/` is the default output directory for the scan manifests (Task 4). The crosswalk and salt
patterns are listed now, before the code that produces them exists, so there is no window in which
a pseudonym-to-MRN mapping can be committed by accident.

- [ ] **Step 4: Create the package skeletons**

`src/ingest/__init__.py`:

```python
"""Dialygo ingest: institutional fistulography handover -> de-identified, patient-grouped frames.

Phase 1 (scan) is read-only inventory; later phases de-identify and extract frames. Every module
here is gated by src.ingest.clearance.require_clearance: nothing touches real patient data until
the institutional agreement (B5) and the IP/engagement agreement (B9) have both executed.

Modules import torch-free, cv2-free and pydicom-free at module level (repo convention); heavy
imports live inside the functions that need them.
"""
```

`tests/fixtures/__init__.py`:

```python
"""Shared synthetic test data. No real patient data may live here (Dialygo B5)."""
```

- [ ] **Step 5: Write the failing test**

Create `tests/test_ingest_fixture.py`:

```python
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
```

- [ ] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/test_ingest_fixture.py -q`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'tests.fixtures.synthetic_dicom'` (`1 error`).

- [ ] **Step 7: Implement the fixture**

Create `tests/fixtures/synthetic_dicom.py`:

```python
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
                    manufacturer="Siemens", study_date="20240517"):
    """Build an in-memory multi-frame XA dataset with PHI + clinical tags + 12-bit pixels.

    UIDs default to freshly generated unique values; pass study_uid/series_uid/sop_uid to force a
    grouping (e.g. two instances that must land in the same series).
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
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ingest_fixture.py -q`
Expected: PASS (10 passed)

- [ ] **Step 9: Confirm nothing else regressed**

Run: `python -m pytest tests/ -q`
Expected: PASS (384 passed) — the 374 pre-existing tests plus the 10 new ones.

- [ ] **Step 10: Commit**

```bash
git add requirements.txt .gitignore src/ingest/__init__.py \
        tests/fixtures/__init__.py tests/fixtures/synthetic_dicom.py \
        tests/test_ingest_fixture.py
git commit -m "feat(ingest): synthetic XA DICOM fixture + package skeleton (B5 test data)"
```

---

### Task 2: `clearance.py` — the B5/B9 legal gate, enforced in code

**Dialygo B5** says no real patient data may be processed until the institutional agreement
executes; **Dialygo B9** says the IP/engagement agreement must execute before real-data
development. Both are currently written down in a requirements document. A requirements document
does not stop `python -m src.ingest.scan --src /Volumes/CATHLAB_HANDOVER`. Nothing about that
command is a mistake a careful person is immune to — the drive is plugged in, the path
autocompletes, the module runs, and by the time anyone notices, unlicensed patient data has been
read, hashed and written into a manifest on a laptop. The prose control fails silently at exactly
the moment it matters.

So the gate becomes an object every ingest entry point must call before it touches a path.
`mode="synthetic"` (the default everywhere) always passes and never reads the marker, so the whole
test suite runs unimpeded. `mode="real"` reads `configs/ingest_clearance.yaml` and refuses unless
**both** flags are a genuine YAML boolean `true`.

The parsing is fail-safe in exactly the shape of `src/serve/registry.py:69-76`: PyYAML's SafeLoader
resolves only the unquoted tokens `true/false/yes/no/on/off` to real Python booleans, so a quoted
`"true"`, the number `1`, a typo like `ture`, an explicit `null`, a list, or a mapping all load as
something that is *not* the `True` singleton. Only `value is True` counts. Everything else — a
malformed value, a missing key, a missing file, unreadable YAML, a file whose top level is not a
mapping, an unrecognized `mode` string — degrades to refusal. There is no input that produces a
confident-looking success by accident; the only way to open the gate is to type the word `true`
into a file that a human had to decide to edit.

The refusal message names both B5 and B9, prints the observed value of each flag, and prints the
absolute marker path, because the operator hitting this needs to know *which* agreement is missing
and *which* file the process actually read (a stale relative path is the obvious way to be misled).

**Files:**
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/clearance.py`
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/configs/ingest_clearance.yaml`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_clearance.py`

**Interfaces:**
- Consumes: nothing from Task 1 (the gate is data-free by design).
- Produces:
  - `src.ingest.clearance.ClearanceError` (subclass of `RuntimeError`)
  - `src.ingest.clearance.read_clearance(path) -> dict`
  - `src.ingest.clearance.is_cleared(c) -> bool`
  - `src.ingest.clearance.require_clearance(mode, clearance_path="configs/ingest_clearance.yaml") -> None`
  - `src.ingest.clearance.DATA_FLAG = "data_agreement_executed"`, `IP_FLAG = "ip_agreement_executed"`, `DEFAULT_CLEARANCE_PATH = "configs/ingest_clearance.yaml"`
  - `src.ingest.clearance.main() -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_clearance.py`:

```python
"""The B5/B9 gate. Only a genuine YAML `true` on BOTH flags opens real-data ingest."""
from pathlib import Path

import pytest

from src.ingest.clearance import (
    DATA_FLAG,
    DEFAULT_CLEARANCE_PATH,
    IP_FLAG,
    ClearanceError,
    is_cleared,
    main,
    read_clearance,
    require_clearance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

BOTH_TRUE = f"{DATA_FLAG}: true\n{IP_FLAG}: true\n"
BOTH_FALSE = f"{DATA_FLAG}: false\n{IP_FLAG}: false\n"

# Present-but-malformed values. None of these is the Python singleton True, so none may open
# the gate -- same failure mode registry.floor_ok guards against.
MALFORMED = [
    f'{DATA_FLAG}: "true"\n{IP_FLAG}: "true"\n',        # quoted -> str, not bool
    f"{DATA_FLAG}: 1\n{IP_FLAG}: 1\n",                  # number (truthy under bool())
    f"{DATA_FLAG}: ture\n{IP_FLAG}: ture\n",            # typo -> str
    f"{DATA_FLAG}: null\n{IP_FLAG}: null\n",            # explicit null
    f"{DATA_FLAG}: [true]\n{IP_FLAG}: [true]\n",        # list
    f"{DATA_FLAG}: {{}}\n{IP_FLAG}: {{}}\n",            # mapping
    "signed: true\n",                                   # right intent, wrong key names
]


def _marker(tmp_path, text):
    p = tmp_path / "ingest_clearance.yaml"
    p.write_text(text)
    return str(p)


def test_synthetic_mode_passes_even_with_missing_marker(tmp_path):
    require_clearance("synthetic", str(tmp_path / "does_not_exist.yaml"))


def test_synthetic_mode_passes_with_unexecuted_marker(tmp_path):
    require_clearance("synthetic", _marker(tmp_path, BOTH_FALSE))


def test_real_mode_passes_only_when_both_flags_true(tmp_path):
    require_clearance("real", _marker(tmp_path, BOTH_TRUE))


def test_real_mode_refuses_when_marker_file_missing(tmp_path):
    with pytest.raises(ClearanceError):
        require_clearance("real", str(tmp_path / "nope.yaml"))


def test_real_mode_refuses_when_only_data_agreement_true(tmp_path):
    p = _marker(tmp_path, f"{DATA_FLAG}: true\n{IP_FLAG}: false\n")
    with pytest.raises(ClearanceError):
        require_clearance("real", p)


def test_real_mode_refuses_when_only_ip_agreement_true(tmp_path):
    p = _marker(tmp_path, f"{DATA_FLAG}: false\n{IP_FLAG}: true\n")
    with pytest.raises(ClearanceError):
        require_clearance("real", p)


def test_real_mode_refuses_when_both_false(tmp_path):
    with pytest.raises(ClearanceError):
        require_clearance("real", _marker(tmp_path, BOTH_FALSE))


@pytest.mark.parametrize("text", MALFORMED)
def test_malformed_flag_fails_safe(tmp_path, text):
    """A present-but-malformed value must refuse, never open the gate, never raise something
    other than ClearanceError."""
    p = _marker(tmp_path, text)
    assert is_cleared(read_clearance(p)) is False
    with pytest.raises(ClearanceError):
        require_clearance("real", p)


def test_error_message_names_b5_b9_flags_and_marker_path(tmp_path):
    p = _marker(tmp_path, f'{DATA_FLAG}: true\n{IP_FLAG}: "true"\n')
    with pytest.raises(ClearanceError) as ei:
        require_clearance("real", p)
    msg = str(ei.value)
    assert "B5" in msg and "B9" in msg
    assert DATA_FLAG in msg and IP_FLAG in msg
    assert "True" in msg and "'true'" in msg          # both observed values are printed
    assert str(Path(p).resolve()) in msg              # the path actually read


@pytest.mark.parametrize("mode", ["REAL", "Real", "prod"])
def test_unknown_mode_refuses(tmp_path, mode):
    """Anything that is not exactly 'synthetic' or 'real' is a typo -> refuse."""
    with pytest.raises(ClearanceError):
        require_clearance(mode, _marker(tmp_path, BOTH_TRUE))


def test_read_clearance_missing_returns_empty_dict(tmp_path):
    assert read_clearance(str(tmp_path / "absent.yaml")) == {}


def test_read_clearance_corrupt_yaml_returns_empty_dict(tmp_path):
    assert read_clearance(_marker(tmp_path, "a: [1, 2\n  b: }{\n")) == {}


def test_read_clearance_non_mapping_returns_empty_dict(tmp_path):
    assert read_clearance(_marker(tmp_path, "- true\n- true\n")) == {}


def test_is_cleared_rejects_non_dict():
    for junk in (None, True, "true", 1, [DATA_FLAG, IP_FLAG]):
        assert is_cleared(junk) is False


def test_shipped_config_is_unexecuted():
    """The committed marker must never ship executed. Flipping it is a legal act."""
    p = REPO_ROOT / DEFAULT_CLEARANCE_PATH
    assert p.exists(), f"{DEFAULT_CLEARANCE_PATH} must be committed"
    c = read_clearance(str(p))
    assert c.get(DATA_FLAG) is False
    assert c.get(IP_FLAG) is False
    assert is_cleared(c) is False


def test_main_returns_nonzero_when_not_cleared(tmp_path, monkeypatch, capsys):
    p = _marker(tmp_path, BOTH_FALSE)
    monkeypatch.setattr("sys.argv", ["clearance", "--mode", "real", "--clearance", p])
    assert main() == 1
    out = capsys.readouterr().out
    assert "B5" in out and "B9" in out


def test_main_returns_zero_for_synthetic_mode(tmp_path, monkeypatch, capsys):
    p = _marker(tmp_path, BOTH_FALSE)
    monkeypatch.setattr("sys.argv", ["clearance", "--mode", "synthetic", "--clearance", p])
    assert main() == 0
    assert "permitted" in capsys.readouterr().out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ingest_clearance.py -q`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'src.ingest.clearance'` (`1 error`).

- [ ] **Step 3: Create the clearance marker config**

Create `configs/ingest_clearance.yaml`:

```yaml
# Dialygo B5 / B9 LEGAL GATE -- read by src.ingest.clearance.require_clearance("real", ...).
#
# Flipping either flag to true is a LEGAL ACT, not a config tweak. It asserts that a signed,
# executed agreement exists on file. Do not flip it to "unblock a run", to test something, or
# because the drive is already plugged in. If you are not the person who can point at the executed
# document, you are not the person who edits this file.
#
#   data_agreement_executed  (B5) -- institutional data-sharing / DPA covering these studies
#   ip_agreement_executed    (B9) -- IP / engagement agreement covering this development work
#
# BOTH must be an unquoted YAML boolean `true` before any real patient data may be read. Anything
# else -- "true", 1, yes-but-quoted, a typo, null, a list -- is treated as NOT executed and the
# ingest refuses. Refusing is the safe outcome; there is no override flag by design.
data_agreement_executed: false
ip_agreement_executed: false
```

- [ ] **Step 4: Implement the gate**

Create `src/ingest/clearance.py`:

```python
"""Dialygo B5/B9 clearance gate: refuse to process real patient data until both agreements execute.

B5 (institutional data agreement) and B9 (IP/engagement agreement) are legal preconditions, and a
prose reminder in a requirements doc does not stop `--src /Volumes/CATHLAB_HANDOVER` from being
typed. This module turns both into a precondition every ingest entry point calls before it opens a
path.

Fail-safe parsing, identical in spirit to src/serve/registry.py's floor_ok: PyYAML's SafeLoader
only resolves the unquoted tokens true/false/yes/no/on/off to real booleans, so a quoted "true",
the number 1, a typo, null, a list or a mapping all load as something that is NOT the `True`
singleton. Only `value is True` opens the gate. A missing file, unreadable YAML, a non-mapping top
level, a missing key, a malformed value, or an unrecognized mode string all degrade to refusal --
never to a confident-looking success.

CLI:  python -m src.ingest.clearance --mode real
"""
import os

import yaml

DATA_FLAG = "data_agreement_executed"     # B5: institutional data-sharing agreement
IP_FLAG = "ip_agreement_executed"         # B9: IP / engagement agreement
DEFAULT_CLEARANCE_PATH = "configs/ingest_clearance.yaml"
VALID_MODES = ("synthetic", "real")


class ClearanceError(RuntimeError):
    """Raised when real-data processing is attempted without executed B5 + B9 agreements."""


def read_clearance(path):
    """Load the clearance marker as a dict. Any failure yields {} (which is never cleared).

    Returns {} for: a missing/unreadable file, invalid YAML, or a top level that is not a mapping.
    Never raises -- the refusal is expressed by require_clearance, not by a parse error.
    """
    try:
        with open(path) as f:
            c = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return c if isinstance(c, dict) else {}


def is_cleared(c):
    """True only if BOTH flags are the genuine YAML boolean `true`. Everything else is False."""
    if not isinstance(c, dict):
        return False
    return c.get(DATA_FLAG) is True and c.get(IP_FLAG) is True


def require_clearance(mode, clearance_path=DEFAULT_CLEARANCE_PATH):
    """Gate an ingest run. Returns None if permitted; raises ClearanceError otherwise.

    mode="synthetic" -> always permitted, marker never read (the whole test suite runs here).
    mode="real"      -> permitted only when read_clearance(clearance_path) is_cleared.
    anything else    -> refused, because an unrecognized mode is a typo, not a permission.
    """
    if mode == "synthetic":
        return None
    if mode not in VALID_MODES:
        raise ClearanceError(
            f"Dialygo B5/B9 REFUSAL: unrecognized ingest mode {mode!r}; expected one of "
            f"{list(VALID_MODES)}. Refusing rather than guessing which one you meant.")

    resolved = os.path.abspath(clearance_path)
    c = read_clearance(resolved)
    if is_cleared(c):
        return None
    raise ClearanceError(
        "Dialygo B5/B9 REFUSAL: real patient data may not be processed until BOTH agreements "
        "have executed.\n"
        f"  B5 institutional data agreement -> {DATA_FLAG} = {c.get(DATA_FLAG)!r}\n"
        f"  B9 IP / engagement agreement    -> {IP_FLAG} = {c.get(IP_FLAG)!r}\n"
        f"  clearance marker read from      -> {resolved}\n"
        "Only an unquoted YAML `true` counts; a quoted \"true\", a 1, a typo, null or a list all "
        "read as NOT executed. Until legal sign-off lands, run with mode='synthetic' against the "
        "synthetic fixture (tests/fixtures/synthetic_dicom.py).")


def main():
    """CLI: report the clearance state and exit non-zero if the requested mode is refused."""
    import argparse

    ap = argparse.ArgumentParser(description="Check the Dialygo B5/B9 ingest clearance gate.")
    ap.add_argument("--mode", default="real", choices=list(VALID_MODES))
    ap.add_argument("--clearance", default=DEFAULT_CLEARANCE_PATH)
    a = ap.parse_args()

    resolved = os.path.abspath(a.clearance)
    c = read_clearance(resolved)
    print(f"clearance marker: {resolved}")
    print(f"  {DATA_FLAG} = {c.get(DATA_FLAG)!r}   (B5)")
    print(f"  {IP_FLAG} = {c.get(IP_FLAG)!r}   (B9)")
    try:
        require_clearance(a.mode, resolved)
    except ClearanceError as e:
        print(str(e))
        return 1
    print(f"OK: mode={a.mode!r} is permitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ingest_clearance.py -q`
Expected: PASS (25 passed)

- [ ] **Step 6: Verify the CLI refuses by hand**

Run: `python -m src.ingest.clearance --mode real; echo "exit=$?"`
Expected: prints both flags as `False`, prints the B5/B9 refusal block naming
`configs/ingest_clearance.yaml`, then `exit=1`.

- [ ] **Step 7: Commit**

```bash
git add src/ingest/clearance.py configs/ingest_clearance.yaml tests/test_ingest_clearance.py
git commit -m "feat(ingest): enforce B5/B9 clearance gate in code, fail-safe to refusal"
```

---

### Task 3: `manifest.py` — content hashing, atomic JSONL, resume state, provenance

Two design decisions in this module are worth stating plainly, because both look like corner-cutting
until you have watched them matter.

**`head_key` hashes size + the first 64 KB, not the whole file.** A hospital handover is not a clean
export; it is a drive with the same series copied into `BACKUP/`, `to_send/`, `Copy of STUDY_A/`
and a vendor CD image, several times over. Duplicate detection is the single highest-value thing
Phase 1 does, and it has to run over the whole drive before anything else can be planned. Full
SHA-256 of a terabyte at ~200 MB/s over USB is roughly ninety minutes of reading to answer a
question that the first 64 KB answers: a DICOM's first 64 KB is its file-meta group plus the whole
identifying header (UIDs, patient, study, series, acquisition parameters), so two files that agree
on total size *and* on that header are the same instance for inventory purposes. The tradeoff is
explicit and tested: two same-size files with identical first 64 KB and different tails collide.
That is acceptable for grouping candidates; `sha256_file` exists for the moment a candidate group
must be proven byte-identical before anything is deleted or merged.

**State writes are atomic; JSONL appends are not.** These are different failure modes. `files.jsonl`
is append-only and line-oriented, so a drive that disconnects mid-write leaves a torn final line,
and `read_jsonl` simply drops it — one file gets re-scanned, nothing is lost. The checkpoint is the
opposite: it is a single JSON object that is rewritten in full after every directory. A truncated
rewrite that still parses (or that parses as an older, shorter `done_dirs`) is worse than having no
checkpoint at all, because the resume run reads it, believes directories were completed that were
never scanned, and skips them — producing a manifest that looks complete and quietly omits whole
subtrees. So the state goes through `write_json_atomic`: write to a temp file in the same
directory, `fsync`, then `os.replace`, which is atomic on POSIX. The reader either sees the whole
old state or the whole new one, never a blend.

`provenance()` stamps the short git commit so a manifest can be tied back to the exact code that
produced it. Outside a checkout (a copied output directory, a machine without git), that resolves
to `"<unknown>"` rather than blowing up — provenance is metadata, not a precondition.

**Files:**
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/manifest.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_manifest.py`

**Interfaces:**
- Consumes: nothing (pure stdlib; no dependency on Task 1 or 2).
- Produces:
  - `src.ingest.manifest.SCHEMA_VERSION = 1`
  - `src.ingest.manifest.sha256_file(path, chunk=1 << 20) -> str`
  - `src.ingest.manifest.head_key(path, n=65536) -> str` (`"<size>:<sha256 of first n bytes>"`)
  - `src.ingest.manifest.append_jsonl(path, row) -> None`
  - `src.ingest.manifest.read_jsonl(path) -> list[dict]`
  - `src.ingest.manifest.write_json_atomic(path, obj) -> None`
  - `src.ingest.manifest.load_state(path) -> dict`
  - `src.ingest.manifest.save_state(path, state) -> None`
  - `src.ingest.manifest.provenance(tool, **extra) -> dict` with keys `tool, schema_version, git_commit, utc, python` plus `extra`
  - `src.ingest.manifest.main() -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_manifest.py`:

```python
"""Hashing, append-only JSONL, atomic state, provenance -- the ingest bookkeeping primitives."""
import hashlib
import json
import os
import re

from src.ingest import manifest


def test_schema_version_is_one():
    assert manifest.SCHEMA_VERSION == 1


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    data = os.urandom(4096)
    p.write_bytes(data)
    assert manifest.sha256_file(str(p)) == hashlib.sha256(data).hexdigest()


def test_sha256_file_chunk_size_does_not_change_digest(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"x" * 100_000)
    assert manifest.sha256_file(str(p), chunk=7) == manifest.sha256_file(str(p))


def test_head_key_format(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    key = manifest.head_key(str(p))
    assert re.fullmatch(r"3:[0-9a-f]{64}", key)
    assert key.split(":")[1] == hashlib.sha256(b"abc").hexdigest()


def test_head_key_reads_only_first_n_bytes(tmp_path):
    """DELIBERATE collision: same size + same first 64K -> same key, tails ignored.

    This is the whole point (vendor CDs duplicate series constantly and full-hashing a terabyte to
    find them is wasteful). sha256_file is the tie-breaker when a group must be PROVEN identical.
    """
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    head = b"H" * 65536
    a.write_bytes(head + b"tail-one")
    b.write_bytes(head + b"tail-two")
    assert manifest.head_key(str(a)) == manifest.head_key(str(b))
    assert manifest.sha256_file(str(a)) != manifest.sha256_file(str(b))


def test_head_key_differs_on_size(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"H" * 65536 + b"xx")
    b.write_bytes(b"H" * 65536 + b"xxx")
    assert manifest.head_key(str(a)) != manifest.head_key(str(b))


def test_head_key_short_file_hashes_whole_file(tmp_path):
    p = tmp_path / "s.bin"
    p.write_bytes(b"tiny")
    assert manifest.head_key(str(p), n=65536) == f"4:{hashlib.sha256(b'tiny').hexdigest()}"


def test_append_jsonl_creates_parent_dirs_and_appends(tmp_path):
    p = tmp_path / "deep" / "nested" / "files.jsonl"
    manifest.append_jsonl(str(p), {"path": "/a", "kind": "dicom"})
    manifest.append_jsonl(str(p), {"path": "/b", "kind": "video"})
    rows = manifest.read_jsonl(str(p))
    assert [r["path"] for r in rows] == ["/a", "/b"]
    assert rows[1]["kind"] == "video"


def test_read_jsonl_missing_returns_empty(tmp_path):
    assert manifest.read_jsonl(str(tmp_path / "nothing.jsonl")) == []


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"a": 2}\n')
    assert manifest.read_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_skips_torn_tail_line(tmp_path):
    """A drive yanked mid-append leaves a half-written line. Drop it, keep the rest."""
    p = tmp_path / "f.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "pa')
    assert manifest.read_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_write_json_atomic_leaves_no_temp_files(tmp_path):
    p = tmp_path / "out" / "state.json"
    manifest.write_json_atomic(str(p), {"done_dirs": ["/a"]})
    assert os.listdir(tmp_path / "out") == ["state.json"]
    assert json.loads(p.read_text()) == {"done_dirs": ["/a"]}


def test_write_json_atomic_overwrites(tmp_path):
    p = tmp_path / "state.json"
    manifest.write_json_atomic(str(p), {"n": 1})
    manifest.write_json_atomic(str(p), {"n": 2})
    assert json.loads(p.read_text()) == {"n": 2}
    assert os.listdir(tmp_path) == ["state.json"]


def test_save_load_state_roundtrip(tmp_path):
    p = str(tmp_path / "scan_state.json")
    state = {"schema_version": 1, "site": "site_a", "done_dirs": ["/x", "/y"]}
    manifest.save_state(p, state)
    assert manifest.load_state(p) == state


def test_load_state_missing_returns_empty(tmp_path):
    assert manifest.load_state(str(tmp_path / "absent.json")) == {}


def test_load_state_corrupt_returns_empty(tmp_path):
    p = tmp_path / "scan_state.json"
    p.write_text('{"done_dirs": ["/x", ')
    assert manifest.load_state(str(p)) == {}


def test_load_state_non_mapping_returns_empty(tmp_path):
    p = tmp_path / "scan_state.json"
    p.write_text("[1, 2, 3]")
    assert manifest.load_state(str(p)) == {}


def test_provenance_has_required_keys():
    p = manifest.provenance("ingest.scan")
    assert p["tool"] == "ingest.scan"
    assert p["schema_version"] == manifest.SCHEMA_VERSION
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", p["utc"])
    assert p["python"].startswith("3.")
    assert isinstance(p["git_commit"], str) and p["git_commit"]


def test_provenance_extra_kwargs_included():
    p = manifest.provenance("ingest.scan", site="site_a", roots=["/Volumes/X"])
    assert p["site"] == "site_a"
    assert p["roots"] == ["/Volumes/X"]


def test_provenance_git_commit_is_short_hash_or_unknown():
    assert re.fullmatch(r"[0-9a-f]{7,40}|<unknown>", manifest.provenance("t")["git_commit"])


def test_provenance_git_commit_falls_back_when_git_missing(monkeypatch):
    """Outside a checkout (or with no git binary) provenance still returns -- it is metadata."""
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(manifest.subprocess, "run", boom)
    assert manifest.provenance("ingest.scan")["git_commit"] == "<unknown>"


def test_main_prints_row_and_state_counts(tmp_path, monkeypatch, capsys):
    jl = tmp_path / "files.jsonl"
    manifest.append_jsonl(str(jl), {"path": "/a", "kind": "dicom"})
    manifest.append_jsonl(str(jl), {"path": "/b", "kind": "image"})
    st = tmp_path / "scan_state.json"
    manifest.save_state(str(st), {"done_dirs": ["/a", "/b", "/c"]})
    monkeypatch.setattr("sys.argv", ["manifest", "--jsonl", str(jl), "--state", str(st)])
    assert manifest.main() == 0
    out = capsys.readouterr().out
    assert "2 rows" in out
    assert "done_dirs=3" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ingest_manifest.py -q`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'src.ingest.manifest'` (`1 error`).

- [ ] **Step 3: Implement**

Create `src/ingest/manifest.py`:

```python
"""Ingest bookkeeping: content hashing, append-only JSONL, atomic resume state, provenance.

head_key hashes SIZE + the first 64 KB rather than the whole file. Vendor handovers duplicate the
same series across BACKUP/, to_send/ and CD images constantly, and duplicate detection has to run
over the entire drive before anything else can be planned -- full-SHA-256-ing a terabyte to answer
that is ~90 minutes of USB reads for a question the header answers. A DICOM's first 64 KB contains
the file-meta group and the whole identifying header (UIDs, patient, study, series, acquisition),
so equal size + equal head hash means "same instance" for inventory purposes. The tradeoff is
deliberate and tested: same-size files with equal first 64 KB collide. Use sha256_file to PROVE a
candidate group is byte-identical before anything is merged or deleted.

Write durability is split on purpose:
  * files.jsonl is append-only and line-oriented. A drive disconnecting mid-append leaves a torn
    final line; read_jsonl drops it and that one file is re-scanned. Cheap, self-healing.
  * the state checkpoint is a single object rewritten in full after every directory. A half-written
    checkpoint that still parses is WORSE than no checkpoint: resume trusts it and silently skips
    directories that were never scanned, producing a manifest that looks complete. So state goes
    through write_json_atomic (temp file in the same dir -> fsync -> os.replace), and a reader sees
    either the whole old state or the whole new one.

Module imports are stdlib-only: no torch, no cv2, no pydicom.

CLI:  python -m src.ingest.manifest --jsonl .ingest/files.jsonl --state .ingest/scan_state.json
"""
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone

SCHEMA_VERSION = 1
UNKNOWN_COMMIT = "<unknown>"


def sha256_file(path, chunk=1 << 20):
    """Full SHA-256 of a file, streamed in `chunk`-byte blocks. The expensive, exact answer."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def head_key(path, n=65536):
    """Cheap duplicate key: '<size>:<sha256 of the first n bytes>'.

    Reads at most n bytes. Files shorter than n are hashed whole. Deliberately collides for
    same-size files whose first n bytes match -- see the module docstring.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(n)
    return f"{size}:{hashlib.sha256(head).hexdigest()}"


def append_jsonl(path, row):
    """Append one JSON object as a line, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    """Read a JSONL file into a list of dicts. Missing file -> [].

    Blank lines are skipped. Lines that do not parse (a torn tail from an interrupted append) or
    that parse to something other than an object are skipped rather than raising: one dropped row
    means one re-scanned file, whereas raising loses the entire manifest.
    """
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_json_atomic(path, obj):
    """Write JSON so a reader never observes a partial file: temp -> fsync -> os.replace."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_state(path):
    """Load a resume checkpoint. Missing, unreadable, corrupt, or non-object JSON -> {}.

    {} means "nothing is known to be done", i.e. re-scan everything. Re-scanning is idempotent and
    cheap; trusting a damaged checkpoint silently skips unscanned directories.
    """
    try:
        with open(path) as f:
            st = json.load(f)
    except (OSError, ValueError):
        return {}
    return st if isinstance(st, dict) else {}


def save_state(path, state):
    """Atomically persist a resume checkpoint. Stores `state` verbatim (load_state round-trips)."""
    write_json_atomic(path, state)


def _git_commit():
    """Short HEAD hash of the checkout this module lives in, or '<unknown>' outside one."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_COMMIT
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else UNKNOWN_COMMIT


def provenance(tool, **extra):
    """Stamp an artifact with what produced it. `extra` keys are merged in (and may override)."""
    p = {
        "tool": tool,
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
    }
    p.update(extra)
    return p


def main():
    """CLI: print provenance and summarize an existing JSONL manifest and/or state checkpoint."""
    import argparse

    ap = argparse.ArgumentParser(description="Inspect ingest manifest / state artifacts.")
    ap.add_argument("--jsonl", default=None, help="path to a files.jsonl to count")
    ap.add_argument("--state", default=None, help="path to a scan_state.json to summarize")
    a = ap.parse_args()

    print(json.dumps(provenance("ingest.manifest"), sort_keys=True))
    if a.jsonl:
        print(f"{a.jsonl}: {len(read_jsonl(a.jsonl))} rows")
    if a.state:
        st = load_state(a.state)
        print(f"{a.state}: {len(st)} keys, done_dirs={len(st.get('done_dirs', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ingest_manifest.py -q`
Expected: PASS (22 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ingest/manifest.py tests/test_ingest_manifest.py
git commit -m "feat(ingest): head-key hashing, append-only JSONL, atomic resume state, provenance"
```

---

### Task 4: `scan.py` — Phase 1 read-only walk with magic-byte typing

Phase 1 answers one question: *what is actually on this drive?* It must answer it without
modifying a single byte of the original, without needing the whole drive in one sitting, and
without being killed by whatever the vendor put in the corner of the filesystem.

**Typing is by magic bytes, not extension.** Vendor CD dumps and PACS exports routinely store DICOM
with no extension at all (`IM_0001`, `CD_IMG_0001`, `I0000001`) or with vendor extensions
(`.ima`, `.dcm30`, none). Matching `*.dcm` against a real handover misses most of it, and a scan
that reports "no DICOM found" on a drive full of DICOM is the worst possible outcome — it looks
like a clean, decisive answer. So `is_dicom` reads the first 140 bytes and checks `b"DICM"` at
offset 128 (Part-10), with a fallback for the headerless CD variant: interpret the first four bytes
as a little-endian tag and accept group `0x0002` (file-meta group kept, preamble dropped) or group
`0x0008` (identification module) with a small element number. Task 1's `write_headerless` produces
exactly this shape, with `(0008,0005) SpecificCharacterSet` first, so the detector is tested
against the real byte layout rather than a mock.

Note what this module does *not* do: it never imports pydicom and never parses a dataset. A Phase 1
inventory that calls `dcmread` on 200,000 files inherits every decoder failure, every unsupported
transfer syntax and every malformed private tag on the drive — and each one is a chance to abort the
walk. Reading 140 bytes cannot fail that way.

**Failure degrades to a recorded row, never to a crash.** An unreadable file (permissions, a dangling
symlink, a bad sector) is written as `kind="unreadable"` with the exception text attached. The scan
keeps going and the operator gets a list of what could not be read. A traceback at file 180,000
would leave an inventory that is silently 10% short.

**Resume checkpoints per completed directory.** After every directory the set of completed
directory paths is written atomically (Task 3). A resumed run skips those directories entirely, so
re-running after a disconnect costs nothing and duplicates nothing. `.DS_Store`, `Thumbs.db` and
Spotlight metadata are skipped by name — macOS writes `.DS_Store` into every folder it displays, and
these are noise that inflates the file counts the plan is built from.

**`require_clearance(mode, clearance_path)` is the first statement in `scan_tree`**, before any
`os.walk`, any `os.makedirs`, any stat. The default `mode` on both the function and the CLI is
`"synthetic"`, so the only way to point this at a real drive is to type `--mode real` *and* have a
countersigned marker on disk.

**Files:**
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/scan.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_scan.py`

**Interfaces:**
- Consumes:
  - `src.ingest.clearance.require_clearance(mode, clearance_path)`, `src.ingest.clearance.ClearanceError`, `src.ingest.clearance.DEFAULT_CLEARANCE_PATH` (Task 2)
  - `src.ingest.manifest.head_key`, `append_jsonl`, `read_jsonl`, `load_state`, `save_state`, `write_json_atomic`, `provenance`, `SCHEMA_VERSION` (Task 3)
  - `tests.fixtures.synthetic_dicom.make_xa_dataset`, `write_dataset`, `write_headerless` (Task 1, tests only)
- Produces:
  - `src.ingest.scan.is_dicom(path) -> bool`
  - `src.ingest.scan.classify(path) -> "dicom"|"video"|"image"|"label"|"other"`
  - `src.ingest.scan.scan_tree(roots, out_dir, *, resume=True, mode="synthetic", clearance_path="configs/ingest_clearance.yaml", site="unknown") -> dict`
  - `src.ingest.scan.summarize(rows) -> {"counts": {kind: n}, "bytes": {kind: n}, "n_files": n}`
  - `src.ingest.scan.main() -> int`
  - on-disk: `<out_dir>/files.jsonl` rows `{"path","kind","size","head_key"}` (plus `"error"` when `kind == "unreadable"`), `<out_dir>/scan_state.json`, `<out_dir>/scan_provenance.json`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_scan.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ingest_scan.py -q`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'src.ingest.scan'` (`1 error`).

- [ ] **Step 3: Implement**

Create `src/ingest/scan.py`:

```python
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

    os.makedirs(out_dir, exist_ok=True)
    files_path = os.path.join(out_dir, FILES_JSONL)
    state_path = os.path.join(out_dir, STATE_JSON)
    prov_path = os.path.join(out_dir, PROVENANCE_JSON)

    if resume:
        done = set(manifest.load_state(state_path).get("done_dirs", []))
    else:
        done = set()
        if os.path.exists(files_path):
            os.remove(files_path)

    def _checkpoint():
        manifest.save_state(state_path, {"schema_version": manifest.SCHEMA_VERSION,
                                         "site": site, "roots": roots,
                                         "done_dirs": sorted(done)})

    n_new = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            key = os.path.abspath(dirpath)
            if key in done:
                continue                              # already inventoried on a previous run
            for name in sorted(filenames):
                if name in SKIP_NAMES:
                    continue
                manifest.append_jsonl(files_path, _row(os.path.join(dirpath, name)))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ingest_scan.py -q`
Expected: PASS (24 passed)

- [ ] **Step 5: Prove the gate on the CLI**

Run:
```bash
python -m src.ingest.scan --src tests/fixtures --out /tmp/ingest_smoke --mode real; echo "exit=$?"
```
Expected: no output directory is created; the B5/B9 refusal traceback ends with
`src.ingest.clearance.ClearanceError: Dialygo B5/B9 REFUSAL: real patient data may not be processed until BOTH agreements have executed.`
and `exit=1`.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (455 passed) — the 374 pre-existing tests plus 10 (Task 1) + 25 (Task 2) + 22
(Task 3) + 24 (Task 4).

- [ ] **Step 7: Confirm no scan output was committed by accident**

Run: `git status --porcelain`
Expected: only `src/ingest/scan.py` and `tests/test_ingest_scan.py` appear as untracked; no
`.ingest/` entry (it is ignored by the Task 1 `.gitignore` change).

- [ ] **Step 8: Commit**

```bash
git add src/ingest/scan.py tests/test_ingest_scan.py
git commit -m "feat(ingest): read-only drive scan with magic-byte typing and resumable checkpoints"
```
### Task 5: `index_dicom.py` — Phase 2 header index, hierarchy, SOP dedupe

Phase 1 (`scan.py`) told us *which* files exist. Phase 2 tells us *what they are*. Every file that
Phase 1 tagged `kind="dicom"` gets its header parsed with `stop_before_pixels=True`, so indexing a
terabyte-scale drive costs one header parse per file rather than one pixel decode per file — an XA
cine with 60 frames of 1024x1024 12-bit pixels is ~120 MB of pixel data behind a ~4 KB header, and
we never touch the pixels in this phase. The whole drive can be indexed in minutes instead of hours,
and the index can be re-run cheaply whenever we widen `INDEX_TAGS`.

SOP-level dedupe is not an optimisation, it is a correctness requirement. Institutional handovers
arrive as vendor CD burns, re-burns after a failed copy, and a separate PACS pull of the same study,
all dropped into one folder tree. The same `SOPInstanceUID` therefore routinely appears in two or
three places under different filenames. If we count those duplicates we inflate the apparent patient
count (the cohort looks larger than it is, and every per-patient statistic is wrong), and worse,
Phase 3 would extract the same patient's frames twice. Once de-identification is applied per-file,
two copies of one patient could land under two different pseudonyms — the same person on both sides
of a train/test split, which silently invents generalisation performance that does not exist.
Dedupe by `SOPInstanceUID`, first path wins, sorted by path so the winner is deterministic across
runs and machines.

`read_header` is fail-safe by construction: `force=True` so a vendor CD file missing the 128-byte
preamble and `DICM` magic still parses (these are common in older CD burns and Phase 1 already
flagged them as candidate DICOM), and a `None` return whenever `SOPInstanceUID` is absent or parsing
raises at all. A file that cannot be identified is dropped from the index rather than entered with
guessed identity. All values are coerced to JSON-serialisable scalars, because pydicom hands back
`PersonName`, `DSfloat`, `IS` and `MultiValue` objects that `json.dumps` cannot write.

The index is written next to the source data on the cleared drive and is never committed: it
deliberately carries the identifying tags (`PatientName`, `AccessionNumber`, …) because the PHI audit
in Task 6 reads them to tell a human reviewer what is actually present before anything is scrubbed.

**Files:**
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/index_dicom.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_index.py`
- Modify: none

**Interfaces:**
- Consumes: `src.ingest.clearance.require_clearance(mode, clearance_path)`, `src.ingest.clearance.ClearanceError`; `src.ingest.manifest.append_jsonl(path, row)`, `read_jsonl(path)`, `write_json_atomic(path, obj)`, `provenance(tool, **extra)`; `scan.py` `files.jsonl` rows `{"path","kind","size","head_key"}`; `tests/fixtures/synthetic_dicom.make_xa_dataset`, `write_dataset`, `write_headerless`
- Produces:
  - `INDEX_TAGS: tuple[str, ...]`
  - `read_header(path) -> dict | None`
  - `dedupe_by_sop(records: list[dict]) -> list[dict]`
  - `build_hierarchy(records: list[dict]) -> dict[str, dict[str, dict[str, list[dict]]]]`
  - `build_index(files_rows, out_dir, *, mode="synthetic", clearance_path="configs/ingest_clearance.yaml", site="unknown") -> dict`
  - `main() -> int`

---

- [ ] **Step 1: `read_header` on a well-formed Part-10 file returns a JSON-serialisable record**

Write the failing test at `tests/test_ingest_index.py`:

```python
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
    return {"path": str(path), "kind": kind, "size": path.stat().st_size, "head_key": "hk"}


def test_read_header_parses_part10_and_is_json_serialisable(tmp_path):
    ds = make_xa_dataset("INU-00417", n_frames=8, sop_uid="1.2.826.0.1.3680043.8.498.3001")
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — ModuleNotFoundError: No module named 'src.ingest.index_dicom'`

Implement `src/ingest/index_dicom.py`:

```python
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
    try:
        return "\\".join(str(v) for v in value)      # MultiValue / sequence of scalars
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (1 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): header-only DICOM record reader with JSON-safe coercion (Task 5)"
```

---

- [ ] **Step 2: `read_header` fail-safe paths — headerless parses, non-DICOM returns None**

Append to `tests/test_ingest_index.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (5 passed)` — `read_header` already handles these; this step pins the fail-safe
contract so a later refactor cannot quietly start admitting unidentifiable files. If
`test_read_header_returns_none_for_missing_sop_uid` errors with
`TypeError: save_as() got an unexpected keyword argument 'enforce_file_format'` you are on
pydicom < 3; drop that keyword and re-run.

Commit.

```bash
git add tests/test_ingest_index.py
git commit -m "test(ingest): pin read_header fail-safe behaviour on headerless and non-DICOM input (Task 5)"
```

---

- [ ] **Step 3: `dedupe_by_sop` — first path wins, deterministically**

Append to `tests/test_ingest_index.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — ImportError: cannot import name 'dedupe_by_sop' from 'src.ingest.index_dicom'`

Implement — append to `src/ingest/index_dicom.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (7 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): deterministic SOP-level dedupe for duplicated vendor handover copies (Task 5)"
```

---

- [ ] **Step 4: `build_hierarchy` — patient -> study -> series -> instances**

Append to `tests/test_ingest_index.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — ImportError: cannot import name 'build_hierarchy' from 'src.ingest.index_dicom'`

Implement — append to `src/ingest/index_dicom.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (9 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): patient/study/series hierarchy with sentinel buckets for orphans (Task 5)"
```

---

- [ ] **Step 5: `build_index` end-to-end — exact counts, artifacts, non-DICOM rows ignored**

Append to `tests/test_ingest_index.py`:

```python
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


def test_build_index_counts_and_artifacts(tmp_path, handover):
    rows, dup = handover
    out = tmp_path / "index_out"

    counts = build_index(rows, out, mode="synthetic", site="inu")

    assert counts == {
        "n_files_seen": 7,       # every row handed to us, including the pdf
        "n_dicom": 5,            # headers that actually parsed (the .TXT did not)
        "n_unique_sop": 4,       # the re-burn collapsed away
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


def test_build_index_is_idempotent(tmp_path, handover):
    rows, _ = handover
    out = tmp_path / "index_out"

    first = build_index(rows, out, mode="synthetic", site="inu")
    text_first = (out / "dicom_index.jsonl").read_text()
    second = build_index(rows, out, mode="synthetic", site="inu")

    assert first == second
    assert (out / "dicom_index.jsonl").read_text() == text_first    # no append-on-rerun


def test_build_index_refuses_real_mode_before_the_agreement(tmp_path, handover):
    """Dialygo B5: the clearance gate is checked before a single header is opened."""
    rows, _ = handover
    out = tmp_path / "index_out"

    with pytest.raises(ClearanceError):
        build_index(rows, out, mode="real", site="inu")

    assert not out.exists()
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — ImportError: cannot import name 'build_index' from 'src.ingest.index_dicom'`

Implement — append to `src/ingest/index_dicom.py`:

```python
def build_index(files_rows, out_dir, *, mode="synthetic",
                clearance_path="configs/ingest_clearance.yaml", site="unknown"):
    """Index every kind="dicom" row, dedupe by SOP, and write the Phase 2 artifacts.

    Writes <out_dir>/dicom_index.jsonl (one JSON object per unique SOP instance) and
    <out_dir>/index_summary.json (counts + provenance). Returns the counts dict with keys
    n_files_seen, n_dicom, n_unique_sop, n_patients, n_studies, n_series.

    The clearance gate runs first, before any file is opened, so `mode="real"` cannot read a
    single byte of patient data until the institutional agreement is executed (Dialygo B5).

    Re-running overwrites rather than appends, so the index is idempotent: two runs of the same
    drive produce byte-identical output.
    """
    require_clearance(mode, clearance_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "dicom_index.jsonl"
    if index_path.exists():
        index_path.unlink()

    records = []
    for row in files_rows:
        row = row or {}
        if row.get("kind") != "dicom":
            continue
        rec = read_header(row.get("path"))
        if rec is None:                      # Phase 1 guessed wrong; not a DICOM instance
            continue
        rec["head_key"] = row.get("head_key")
        rec["size"] = row.get("size")
        records.append(rec)

    unique = dedupe_by_sop(records)
    for rec in unique:
        append_jsonl(str(index_path), rec)

    hier = build_hierarchy(unique)
    counts = {
        "n_files_seen": len(files_rows),
        "n_dicom": len(records),
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (12 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): build_index writes deduped dicom_index.jsonl and index_summary.json (Task 5)"
```

---

- [ ] **Step 6: `main()` — run Phase 2 as `python -m src.ingest.index_dicom`**

Append to `tests/test_ingest_index.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — AttributeError: module 'src.ingest.index_dicom' has no attribute 'main'`

Implement — append to `src/ingest/index_dicom.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (14 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): index_dicom CLI entrypoint with non-zero exit on clearance refusal (Task 5)"
```

---

### Task 6: PHI audit report — the human review surface read *before* de-identification

De-identification is irreversible on the output side, so it must never be the first time a human
finds out what is in the data. `phi_audit.md` is generated from the Phase 2 index and is read by a
person before Task 7/8 ever run: it lists which identifying tags are actually populated and in how
many instances, how many instances declare burned-in annotation, and which scanners produced the
cohort. If it shows a tag we did not plan for, the scrub list gets extended before anything is
written, not after.

Two things the report must say in its own text, because the person reading it may not be the person
who wrote the code:

**`BurnedInAnnotation = NO` is the scanner's claim, not proof.** The tag records what the
acquisition software believed, and in interventional angiography it is routinely wrong: patient
name, MRN and date overlays get rendered into the pixel buffer by the C-arm's display pipeline or by
a downstream review workstation, while the tag is left unset or defaulted to `NO`. A cohort that
reports zero burned-in annotation still needs every extracted frame screened for pixel-domain text.
The report states this so nobody reads a clean count as clearance to skip screening.

**The vendor table is retained on purpose (Dialygo B6).** `Manufacturer` and `ManufacturerModelName`
survive de-identification. Leave-one-site-out external validation is the honest test of whether the
model learned fistula pathology or learned one C-arm's post-processing, and that experiment is only
possible if vendor identity is still attached to each study. A device model is not a patient
identity, and the audit labels the table as retained so a reviewer does not read it as an oversight.

**Files:**
- Modify: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/index_dicom.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_index.py` (append)
- Create: none

**Interfaces:**
- Consumes: `INDEX_TAGS`, records emitted by `read_header` / rows of `dicom_index.jsonl`
- Produces:
  - `PHI_TAGS: tuple[str, ...]` (a subset of `INDEX_TAGS`)
  - `phi_report(records) -> {"populated": {tag: int}, "burned_in_yes": int, "n_records": int, "vendors": {str: int}}`
  - `render_phi_markdown(report) -> str`
  - `write_phi_audit(records, out_dir) -> pathlib.Path` (writes `<out_dir>/phi_audit.md`)
  - `build_index(...)` additionally writes `<out_dir>/phi_audit.md`

---

- [ ] **Step 1: `PHI_TAGS` + `phi_report` counts populated tags, burned-in, vendors**

Append to `tests/test_ingest_index.py`:

```python
from src.ingest.index_dicom import PHI_TAGS, phi_report, render_phi_markdown, write_phi_audit


def test_phi_tags_are_a_subset_of_index_tags():
    """The audit can only report what the index captured."""
    assert set(PHI_TAGS).issubset(set(INDEX_TAGS))
    assert "PatientName" in PHI_TAGS and "AccessionNumber" in PHI_TAGS
    # B6: vendor identity is not PHI and must never be counted as such.
    assert "Manufacturer" not in PHI_TAGS
    assert "ManufacturerModelName" not in PHI_TAGS


def test_phi_report_counts_populated_burned_in_and_vendors():
    records = [
        {"PatientName": "REDDY^SURESH^^Mr", "AccessionNumber": "ACC-1",
         "InstitutionName": "Institute of Nephro-Urology", "StudyDescription": "",
         "BurnedInAnnotation": "YES", "Manufacturer": "Siemens",
         "ManufacturerModelName": "Artis Zee"},
        {"PatientName": "RAO^LATHA", "AccessionNumber": None,
         "InstitutionName": "   ", "StudyDescription": "REDDY fistulogram",
         "BurnedInAnnotation": "NO", "Manufacturer": "Siemens",
         "ManufacturerModelName": "Artis Zee"},
        {"PatientName": "", "AccessionNumber": "ACC-3",
         "InstitutionName": "Institute of Nephro-Urology", "StudyDescription": None,
         "BurnedInAnnotation": None, "Manufacturer": "Philips",
         "ManufacturerModelName": "Allura Xper"},
    ]

    rep = phi_report(records)

    assert rep["n_records"] == 3
    assert rep["populated"]["PatientName"] == 2          # "" does not count
    assert rep["populated"]["AccessionNumber"] == 2      # None does not count
    assert rep["populated"]["InstitutionName"] == 2      # whitespace-only does not count
    assert rep["populated"]["StudyDescription"] == 1
    assert rep["populated"]["PatientBirthDate"] == 0     # absent key -> reported as zero
    assert rep["burned_in_yes"] == 1
    assert rep["vendors"] == {"Siemens / Artis Zee": 2, "Philips / Allura Xper": 1}


def test_phi_report_handles_an_empty_cohort():
    """Zero records must not divide by zero -- the audit still renders."""
    rep = phi_report([])

    assert rep["n_records"] == 0
    assert rep["burned_in_yes"] == 0
    assert rep["vendors"] == {}
    assert set(rep["populated"]) == set(PHI_TAGS)
    assert all(v == 0 for v in rep["populated"].values())
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — ImportError: cannot import name 'PHI_TAGS' from 'src.ingest.index_dicom'`

Implement — append to `src/ingest/index_dicom.py`:

```python
#: Identifying keywords the audit reports on. Deliberately excludes Manufacturer and
#: ManufacturerModelName: those are device identity, not patient identity, and Dialygo B6 retains
#: them through de-identification so leave-one-site-out external validation stays possible.
PHI_TAGS = (
    "PatientID", "PatientName", "PatientBirthDate", "PatientSex", "OtherPatientIDs",
    "AccessionNumber",
    "StudyDate", "StudyTime", "SeriesDate", "AcquisitionDate",
    "StudyDescription",
    "InstitutionName", "InstitutionAddress",
    "ReferringPhysicianName", "PerformingPhysicianName",
)


def _populated(value):
    """True when an element carries something a human could read as an identifier."""
    return value is not None and str(value).strip() != ""


def phi_report(records):
    """Summarise which identifying tags are populated across `records`.

    Returns {"populated": {tag: n}, "burned_in_yes": n, "n_records": n, "vendors": {name: n}}.
    Every PHI_TAGS key is present in "populated" even when the count is zero, so the audit table
    is the same shape for every cohort and a missing row can never be mistaken for a clean one.

    "vendors" is a Manufacturer / ManufacturerModelName tally, reported because it is *kept*
    (Dialygo B6), not because it needs removing.
    """
    populated = {kw: 0 for kw in PHI_TAGS}
    vendors = {}
    burned = 0
    n = 0
    for rec in records:
        n += 1
        for kw in PHI_TAGS:
            if _populated(rec.get(kw)):
                populated[kw] += 1
        if str(rec.get("BurnedInAnnotation") or "").strip().upper() == "YES":
            burned += 1
        maker = str(rec.get("Manufacturer") or "unknown").strip() or "unknown"
        model = str(rec.get("ManufacturerModelName") or "unknown").strip() or "unknown"
        key = f"{maker} / {model}"
        vendors[key] = vendors.get(key, 0) + 1
    return {"populated": populated, "burned_in_yes": burned, "n_records": n, "vendors": vendors}
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (17 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): phi_report tallies identifying tags, burned-in flags and vendors (Task 6)"
```

---

- [ ] **Step 2: `render_phi_markdown` — the document a human actually reads**

Append to `tests/test_ingest_index.py`:

```python
def test_render_phi_markdown_contains_the_required_sections():
    rep = phi_report([
        {"PatientName": "REDDY^SURESH^^Mr", "AccessionNumber": "ACC-1",
         "BurnedInAnnotation": "NO", "Manufacturer": "Siemens",
         "ManufacturerModelName": "Artis Zee"},
        {"PatientName": "RAO^LATHA", "AccessionNumber": "ACC-2",
         "BurnedInAnnotation": "YES", "Manufacturer": "Philips",
         "ManufacturerModelName": "Allura Xper"},
    ])

    md = render_phi_markdown(rep)

    assert md.startswith("# PHI audit")
    assert "Instances audited: 2" in md
    # identifying-tag table
    assert "| Tag | Instances populated | % |" in md
    assert "| PatientName | 2 | 100.0% |" in md
    assert "| PatientBirthDate | 0 | 0.0% |" in md
    # burned-in count + the warning that a NO is only a claim
    assert "Burned-in annotation declared (BurnedInAnnotation=YES): 1 / 2" in md
    assert "is the scanner's claim, not proof" in md
    assert "pixel screening still runs on every frame" in md
    # vendor table, explicitly labelled as retained
    assert "## Vendors (RETAINED through de-identification -- Dialygo B6)" in md
    assert "| Vendor | Instances |" in md
    assert "| Siemens / Artis Zee | 1 |" in md
    assert "| Philips / Allura Xper | 1 |" in md


def test_render_phi_markdown_survives_an_empty_cohort():
    md = render_phi_markdown(phi_report([]))

    assert md.startswith("# PHI audit")
    assert "Instances audited: 0" in md
    assert "| PatientName | 0 | 0.0% |" in md
    assert "_No instances indexed._" in md
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — ImportError: cannot import name 'render_phi_markdown' from 'src.ingest.index_dicom'`

Implement — append to `src/ingest/index_dicom.py`:

```python
#: Stated in every audit. BurnedInAnnotation is populated by the acquisition software and in
#: interventional angiography it is frequently wrong -- name/MRN/date overlays get rendered into
#: the pixel buffer by the C-arm display pipeline or a review workstation while the tag stays NO.
BURNED_IN_CAVEAT = (
    "`BurnedInAnnotation = NO` is the scanner's claim, not proof. Angiography overlays "
    "(patient name, MRN, date, annotations) are frequently burned into the pixel data with this "
    "tag left unset or defaulted to NO. A zero count here is not clearance to skip screening: "
    "pixel screening still runs on every frame extracted in Phase 3."
)


def render_phi_markdown(report):
    """Render a phi_report() dict as the reviewer-facing phi_audit.md text.

    The document is read by a human *before* de-identification runs, so it is written to be read
    standalone: what is populated, how often, what the burned-in tag does and does not tell you,
    and which scanners are in the cohort.
    """
    n = report.get("n_records", 0)
    pct = (lambda c: f"{(100.0 * c / n):.1f}%" if n else "0.0%")
    lines = [
        "# PHI audit",
        "",
        "Generated from the Phase 2 DICOM index, **before** de-identification. Review this "
        "document and extend the scrub list if any tag below is unexpected.",
        "",
        f"Instances audited: {n}",
        "",
        "## Identifying tags populated",
        "",
        "| Tag | Instances populated | % |",
        "| --- | --- | --- |",
    ]
    for kw in PHI_TAGS:
        count = report.get("populated", {}).get(kw, 0)
        lines.append(f"| {kw} | {count} | {pct(count)} |")

    lines += [
        "",
        "## Burned-in annotation",
        "",
        f"Burned-in annotation declared (BurnedInAnnotation=YES): {report.get('burned_in_yes', 0)}"
        f" / {n}",
        "",
        BURNED_IN_CAVEAT,
        "",
        "## Vendors (RETAINED through de-identification -- Dialygo B6)",
        "",
        "Manufacturer and ManufacturerModelName are kept on purpose. Device identity is not "
        "patient identity, and leave-one-site-out external validation needs to know which C-arm "
        "produced each study.",
        "",
        "| Vendor | Instances |",
        "| --- | --- |",
    ]
    vendors = report.get("vendors", {})
    if not vendors:
        lines.append("| _none_ | 0 |")
    for name, count in sorted(vendors.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {name} | {count} |")

    if n == 0:
        lines += ["", "_No instances indexed._"]
    return "\n".join(lines) + "\n"
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (19 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): render_phi_markdown with burned-in caveat and B6 vendor-retention table (Task 6)"
```

---

- [ ] **Step 3: `write_phi_audit` and wire it into `build_index`**

Append to `tests/test_ingest_index.py`:

```python
def test_write_phi_audit_writes_the_markdown(tmp_path):
    out = tmp_path / "audit_out"
    records = [{"PatientName": "REDDY^SURESH^^Mr", "BurnedInAnnotation": "NO",
                "Manufacturer": "Siemens", "ManufacturerModelName": "Artis Zee"}]

    path = write_phi_audit(records, out)

    assert path == out / "phi_audit.md"
    text = path.read_text()
    assert text.startswith("# PHI audit")
    assert "| PatientName | 1 | 100.0% |" in text


def test_build_index_emits_the_phi_audit(tmp_path, handover):
    """Phase 2 always leaves a review surface behind -- the audit is not opt-in."""
    rows, _ = handover
    out = tmp_path / "index_out"

    build_index(rows, out, mode="synthetic", site="inu")

    text = (out / "phi_audit.md").read_text()
    assert "Instances audited: 4" in text
    assert "| PatientName | 4 | 100.0% |" in text
    assert "| InstitutionName | 4 | 100.0% |" in text
    assert "Burned-in annotation declared (BurnedInAnnotation=YES): 0 / 4" in text
    assert "is the scanner's claim, not proof" in text
    assert "| Siemens / Artis Zee | 4 |" in text
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: FAIL — ImportError: cannot import name 'write_phi_audit' from 'src.ingest.index_dicom'`

Implement — append `write_phi_audit` to `src/ingest/index_dicom.py`:

```python
def write_phi_audit(records, out_dir):
    """Write <out_dir>/phi_audit.md from `records` and return its Path.

    Called unconditionally by build_index: Phase 2 must never finish without leaving a review
    surface behind, because de-identification is irreversible on the output side and this file
    is the last chance to notice an unexpected identifying tag.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phi_audit.md"
    path.write_text(render_phi_markdown(phi_report(records)), encoding="utf-8")
    return path
```

and wire it into `build_index` by inserting the call immediately before the summary write:

```python
    write_phi_audit(unique, out)

    write_json_atomic(str(out / "index_summary.json"), {
        "counts": counts,
        "site": site,
        "mode": mode,
        "provenance": provenance("src.ingest.index_dicom", site=site, mode=mode),
    })
    return counts
```

Run it.

```bash
python -m pytest tests/test_ingest_index.py -q
```

`Expected: PASS (21 passed)`

Commit.

```bash
git add src/ingest/index_dicom.py tests/test_ingest_index.py
git commit -m "feat(ingest): emit phi_audit.md from build_index as the pre-deid review surface (Task 6)"
```

---

### Task 7: `deid.py` — pseudonymization key material

Before any tag is touched, we need the machinery that turns a real identifier into a stable
pseudonym. Three design decisions carry the whole de-identification design, and each of them is a
choice against an obvious simpler alternative.

**Pseudonyms are `HMAC-SHA256(salt, real_id)`, not a counter.** A counter (`inu_0001`, `inu_0002`, …)
is simpler and it is wrong here. Handovers arrive in batches: the institution hands over a drive,
then three months later hands over another one that partly overlaps. With a counter, the second run
renumbers everything — the same patient becomes `inu_0001` in batch one and `inu_0347` in batch two,
the two batches cannot be merged, and the cohort silently gains a duplicate patient who appears on
both sides of a train/test split. With a keyed hash the mapping is a pure function of
`(salt, real_id)`: re-running the pipeline is a no-op, and a second batch maps the same patient to
the same pseudonym automatically, so batches merge cleanly. HMAC rather than a plain hash because a
bare `sha256(patient_id)` is trivially reversible by dictionary attack — hospital MRNs come from a
small, guessable space — and the salt is what makes that attack impossible without the key.

**Dates are shifted per patient, not deleted.** Deleting dates destroys the intervals, and intervals
are the clinical signal: whether a stenosis recurred at three months or three years is the difference
between a failed intervention and a normal surveillance interval, and any longitudinal AVF question
needs it. So each patient gets a stable offset in `[-364, 0]` days derived from the same keyed hash,
applied to every date in every study for that patient. Intervals within a patient survive exactly;
absolute dates do not survive at all; and because the offset is per patient, you cannot align two
patients' timelines to recover a real calendar date.

**Domain separation.** The same HMAC key is used for patient pseudonyms, day offsets and UID
remapping, so each call prefixes a domain byte (`"P"`, `"D"`, `"U"`) to the message. Without it, a
patient ID and a UID that happen to be the same text would derive the same value, leaking the fact
that they are equal and creating cross-namespace collisions.

**The salt and the crosswalk are the re-identification key.** Either alone is close to useless;
together they are exactly what maps `inu_3f9c21b04e` back to a named human being. Both are written
`0600` (owner read/write only) on the encrypted drive, alongside the source data and inside the
scope of the institutional agreement. Neither is ever written into the repository, and both must be
covered by `.gitignore` — the crosswalk is the single artifact whose leak would un-anonymise the
entire cohort at once.

**Files:**
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/deid.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_deid.py`
- Modify: none

**Interfaces:**
- Consumes: nothing from earlier ingest modules (this task is pure key material; `deid_dataset` in Task 8 is what touches pydicom)
- Produces:
  - `load_or_create_salt(path, n_bytes=32) -> bytes`
  - `pseudo_id(salt, real_id, *, site, kind="P") -> str`
  - `day_offset(salt, real_patient_id) -> int`
  - `shift_date(yyyymmdd, days) -> str`
  - `remap_uid(salt, uid) -> str`
  - `write_crosswalk(path, mapping) -> pathlib.Path`
  - `main() -> int`

---

- [ ] **Step 1: `load_or_create_salt` — created once, `0600`, reused thereafter**

Write the failing test at `tests/test_ingest_deid.py`:

```python
"""De-identification key material: salt, pseudonyms, date shifts, UID remapping, crosswalk.

Dialygo B5: this suite never reads real patient data. Identifiers here are invented strings
and every DICOM comes from tests/fixtures/synthetic_dicom.py.
"""
import stat

import pytest

from src.ingest.deid import load_or_create_salt


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ModuleNotFoundError: No module named 'src.ingest.deid'`

Implement `src/ingest/deid.py`:

```python
"""De-identification: pseudonymization key material and the DICOM tag scrub.

Pseudonyms are HMAC-SHA256(salt, real_id) rather than a counter. A counter renumbers the cohort
every run, so a second handover batch would map the same patient to a new pseudonym and the two
batches could not be merged -- the same person would appear twice, potentially on both sides of a
train/test split. A keyed hash makes the mapping a pure function of (salt, real_id): re-runs are
no-ops and overlapping batches merge cleanly. HMAC rather than a bare hash because hospital MRNs
come from a small, guessable space and an unkeyed digest of one is reversible by dictionary attack.

Dates are shifted per patient, not deleted, because intervals are clinical signal: whether a
stenosis recurred at three months or three years changes the interpretation entirely. Each patient
gets a stable offset in [-364, 0] days applied to every date in every one of their studies, so
within-patient intervals survive exactly while absolute dates do not, and per-patient offsets stop
anyone aligning two patients to recover a calendar date.

Every derivation is domain-separated ("P" patient, "D" date offset, "U" UID) so a patient ID and a
UID with the same text cannot derive the same value.

The salt and the crosswalk together are exactly what re-identifies the cohort. Both are written
0600 on the encrypted drive, inside the scope of the institutional agreement, and neither is ever
written into the repository. `pydicom` is imported inside functions so this module stays importable
without the imaging stack.
"""
import argparse
import csv
import hmac
import os
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path


def load_or_create_salt(path, n_bytes=32):
    """Return the HMAC salt at `path`, creating it 0600 on first use.

    Created once and reused forever: regenerating the salt would remap the whole cohort to a new
    pseudonym namespace, splitting every already-processed patient in two. A salt file that exists
    but is shorter than `n_bytes` is treated as corruption and raises rather than being silently
    replaced, for the same reason.
    """
    path = Path(path)
    if path.exists():
        salt = path.read_bytes()
        if len(salt) < n_bytes:
            raise ValueError(
                f"salt at {path} is {len(salt)} bytes, expected >= {n_bytes}; refusing to "
                "regenerate -- a new salt would remap every already-processed patient"
            )
        return salt
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = os.urandom(n_bytes)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, salt)
    finally:
        os.close(fd)
    os.chmod(str(path), 0o600)
    return salt
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (4 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): 0600 salt file created once and reused for stable pseudonyms (Task 7)"
```

---

- [ ] **Step 2: `pseudo_id` — deterministic, site-scoped, salt-sensitive**

Append to `tests/test_ingest_deid.py`:

```python
import re

from src.ingest.deid import pseudo_id

SALT_A = b"A" * 32
SALT_B = b"B" * 32


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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ImportError: cannot import name 'pseudo_id' from 'src.ingest.deid'`

Implement — append to `src/ingest/deid.py`:

```python
def _digest(salt, kind, value):
    """Domain-separated HMAC-SHA256 over (kind, value).

    The domain prefix keeps namespaces apart: a PatientID and a UID with identical text derive
    different digests, so nothing leaks the fact that they are equal and nothing collides across
    namespaces. The separator is a NUL byte, which cannot occur in a DICOM identifier.
    """
    msg = kind.encode("ascii") + b"\x00" + str(value).encode("utf-8")
    return hmac.new(salt, msg, sha256).digest()


def pseudo_id(salt, real_id, *, site, kind="P"):
    """Return a stable site-scoped pseudonym for `real_id`, e.g. "inu_3f9c21b04e".

    Pure function of (salt, kind, real_id) with the site as a readable prefix, so a re-run or a
    later handover batch reproduces the same pseudonym and multi-site cohorts stay visually
    separable in filenames and logs.

    Raises ValueError on an empty identifier: an absent MRN must not collapse several distinct
    patients into one shared pseudonym bucket.
    """
    if real_id is None or str(real_id).strip() == "":
        raise ValueError("real_id is empty; refusing to mint a shared pseudonym for it")
    return f"{site}_{_digest(salt, kind, str(real_id).strip()).hex()[:10]}"
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (8 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): HMAC-based site-scoped pseudo_id with domain separation (Task 7)"
```

---

- [ ] **Step 3: `day_offset` — stable per patient, inside `[-364, 0]`**

Append to `tests/test_ingest_deid.py`:

```python
from src.ingest.deid import day_offset


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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ImportError: cannot import name 'day_offset' from 'src.ingest.deid'`

Implement — append to `src/ingest/deid.py`:

```python
def day_offset(salt, real_patient_id):
    """Return this patient's date shift in days, an int in [-364, 0].

    Derived from the patient identifier so it is identical for every study, series and instance of
    that patient across every run and every batch -- which is what makes within-patient intervals
    survive the shift exactly. The offset is always backwards and under a year so shifted dates
    stay plausible and never land in the future.
    """
    if real_patient_id is None or str(real_patient_id).strip() == "":
        raise ValueError("real_patient_id is empty; cannot derive a stable date offset")
    raw = int.from_bytes(_digest(salt, "D", str(real_patient_id).strip())[:8], "big")
    return -(raw % 365)
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (11 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): per-patient day_offset so intervals survive date shifting (Task 7)"
```

---

- [ ] **Step 4: `shift_date` — intervals preserved, malformed input degrades to `""`**

Append to `tests/test_ingest_deid.py`:

```python
from src.ingest.deid import shift_date


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
```

Add the import at the top of the test file:

```python
from datetime import datetime
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ImportError: cannot import name 'shift_date' from 'src.ingest.deid'`

Implement — append to `src/ingest/deid.py`:

```python
def shift_date(yyyymmdd, days):
    """Shift a DICOM DA value by `days`, returning YYYYMMDD, or "" when it cannot be parsed.

    Fail-safe by design: a malformed or absent date becomes an empty element rather than a
    plausible-looking wrong one. An empty DA is honest ("we do not know"); a silently corrected
    date would be read downstream as fact.
    """
    if yyyymmdd is None:
        return ""
    text = str(yyyymmdd).strip()
    if len(text) != 8 or not text.isdigit():
        return ""
    try:
        shifted = datetime.strptime(text, "%Y%m%d") + timedelta(days=int(days))
    except (ValueError, OverflowError, TypeError):
        return ""
    return shifted.strftime("%Y%m%d")
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (23 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): shift_date preserves intervals and degrades to empty on malformed dates (Task 7)"
```

---

- [ ] **Step 5: `remap_uid` — deterministic, collision-free, DICOM-legal**

Append to `tests/test_ingest_deid.py`:

```python
from src.ingest.deid import remap_uid

REAL_UID = "1.2.826.0.1.3680043.8.498.1001"


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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ImportError: cannot import name 'remap_uid' from 'src.ingest.deid'`

Implement — append to `src/ingest/deid.py`:

```python
#: DICOM's registered root for locally generated UIDs derived from a value (PS3.5 B.2).
DERIVED_UID_ROOT = "2.25."


def remap_uid(salt, uid):
    """Map a DICOM UID to a stable pseudonymous UID under the 2.25. root.

    Deterministic, so the study/series/instance relationships in the original data are reproduced
    exactly under the new identifiers: two instances of one series map to one remapped series UID
    without any bookkeeping. 96 bits of HMAC output keeps the result under the 64-byte UI limit
    (2.25. + at most 29 digits = 34 characters) while making collisions unreachable at cohort
    scale.
    """
    if uid is None or str(uid).strip() == "":
        raise ValueError("uid is empty; refusing to mint a shared remapped uid for it")
    value = int.from_bytes(_digest(salt, "U", str(uid).strip())[:12], "big")
    return f"{DERIVED_UID_ROOT}{value}"
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (30 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): deterministic 2.25 UID remapping preserving study/series structure (Task 7)"
```

---

- [ ] **Step 6: `write_crosswalk` — `0600`, round-trips, never in the repo**

Append to `tests/test_ingest_deid.py`:

```python
import csv

from src.ingest.deid import write_crosswalk


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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ImportError: cannot import name 'write_crosswalk' from 'src.ingest.deid'`

Implement — append to `src/ingest/deid.py`:

```python
def write_crosswalk(path, mapping):
    """Write the real_id -> pseudo_id CSV 0600 and return its Path.

    This file plus the salt is precisely what re-identifies the cohort, so it lives on the
    encrypted drive inside the scope of the institutional agreement and is never committed. Rows
    are sorted by real_id so the file is byte-stable across runs and a diff shows only genuinely
    new patients.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["real_id", "pseudo_id"])
        for real, pseudo in sorted(mapping.items()):
            writer.writerow([real, pseudo])
    os.chmod(str(path), 0o600)
    return path
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (32 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): 0600 sorted real->pseudo crosswalk writer (Task 7)"
```

---

- [ ] **Step 7: `main()` — provision key material as `python -m src.ingest.deid`**

Append to `tests/test_ingest_deid.py`:

```python
def test_main_provisions_the_salt_and_reports_a_fingerprint(tmp_path, monkeypatch, capsys):
    import json

    from src.ingest import deid

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

    from src.ingest import deid

    salt_path = tmp_path / "salt.bin"
    monkeypatch.setattr("sys.argv", ["deid", "--salt", str(salt_path), "--site", "inu"])
    deid.main()
    first = json.loads(capsys.readouterr().out)
    deid.main()
    second = json.loads(capsys.readouterr().out)

    assert first["salt_fingerprint"] == second["salt_fingerprint"]
    assert second["created"] is False
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — AttributeError: module 'src.ingest.deid' has no attribute 'main'`

Implement — append to `src/ingest/deid.py`:

```python
def salt_fingerprint(salt):
    """A short public identifier for a salt: HMAC of a fixed label under the salt itself.

    Lets logs and summaries record *which* key a run used -- so a cohort processed with a
    different salt is detectable -- without ever printing the key.
    """
    return _digest(salt, "F", "dialygo-salt-fingerprint").hex()[:16]


def main():
    """CLI: python -m src.ingest.deid --salt <path> [--site inu]

    Provisions the 0600 salt if it does not exist and prints its fingerprint as JSON. Idempotent:
    running twice reuses the existing salt, because a new salt would remap every already-processed
    patient into a fresh pseudonym namespace.
    """
    import json
    import sys

    ap = argparse.ArgumentParser(description="Provision de-identification key material.")
    ap.add_argument("--salt", required=True, help="path to the 0600 salt file on the secure drive")
    ap.add_argument("--site", default="unknown", help="site prefix used in pseudonyms")
    ap.add_argument("--bytes", type=int, default=32, dest="n_bytes")
    args = ap.parse_args()

    existed = Path(args.salt).exists()
    try:
        salt = load_or_create_salt(args.salt, n_bytes=args.n_bytes)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "salt_path": str(args.salt),
        "salt_fingerprint": salt_fingerprint(salt),
        "site": args.site,
        "created": not existed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (34 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): deid CLI provisions salt and prints a key fingerprint (Task 7)"
```

---

### Task 8: `deid.py` — the tag scrub

This implements the DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile: the
standard's list of elements that must be removed or emptied before a dataset can be called
de-identified, plus the retention decisions Dialygo makes on top of it.

**The order of operations is part of the correctness, not an implementation detail.**

1. **Private tags and 60xx overlay planes are removed first.** Private elements are vendor-defined
   and can hold literally anything — some workstations stash the full patient demographic block, an
   accession number, or an operator's free-text note in an odd group. We cannot enumerate them, so
   we do not try: every odd-group element goes. Overlay groups `6000`–`601E` are worse, because an
   overlay is a *bitmap* — a burned-in name rendered as an overlay plane survives every tag-level
   scrub ever written, and no amount of emptying string elements touches it. Both classes are
   destroyed before anything else runs, so that nothing later in the pipeline can read a value out
   of them or copy one forward.
2. **Identifying elements are emptied** (Annex E's Z action: the element stays, its value goes).
   Emptying rather than deleting keeps the dataset conformant and keeps the absence explicit —
   a downstream reader sees "PatientName is empty" rather than having to infer it from a missing
   tag. `StudyDescription` is emptied along with the obvious ones because in practice it carries
   patient names typed by whoever booked the study.
3. **Dates are shifted** by the patient's stable offset (Task 7), because intervals are signal and
   absolute dates are not. Times are emptied outright — a time of day carries no clinical value
   here and is a useful re-identification handle when combined with a hospital schedule.
4. **UIDs are remapped last**, after the identifying content is already gone. Study, series and SOP
   UIDs each map through `remap_uid`, which is deterministic, so the patient/study/series/instance
   *relationships* survive intact under new identifiers: two instances of one series still share a
   series UID, and that series still belongs to the same study. This is what lets Phase 3 group
   frames by patient and lets the split logic keep a patient whole, without ever consulting a real
   identifier.

`PatientID` is *replaced* with the pseudonym rather than emptied, so the de-identified dataset is
still self-describing. `PatientIdentityRemoved="YES"` and a `DeidentificationMethod` string are set
so any downstream reader — including a viewer someone opens by hand — can see at a glance that this
file has been processed. `file_meta.MediaStorageSOPInstanceUID` is updated to the remapped SOP UID:
the file meta group carries its own copy of the instance UID, and leaving it stale would both leak
the original UID and produce a file whose meta and dataset disagree, which some readers reject.

**Dialygo B6 applies here explicitly:** `Manufacturer` and `ManufacturerModelName` are in
`KEEP_TAGS` and survive the scrub untouched. Vendor identity is not patient identity, and
leave-one-site-out external validation — the only honest test of whether the model learned fistula
pathology rather than one C-arm's post-processing — is impossible without it.

**Files:**
- Modify: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/deid.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_deid.py` (append)
- Create: none

**Interfaces:**
- Consumes: `pseudo_id`, `day_offset`, `shift_date`, `remap_uid` (Task 7); `tests/fixtures/synthetic_dicom.make_xa_dataset`, `write_dataset`
- Produces:
  - `REMOVE_TAGS: tuple[str, ...]`, `KEEP_TAGS: tuple[str, ...]`, `DATE_TAGS: tuple[str, ...]`
  - `DEIDENTIFICATION_METHOD: str`
  - `deid_dataset(ds, salt, *, site) -> (pydicom.Dataset, dict)` with `ids` keys `real_patient`, `pseudo_patient`, `pseudo_study`, `pseudo_series`, `pseudo_sop`
  - `residual_phi(ds) -> list[str]`

---

- [ ] **Step 1: `REMOVE_TAGS` / `KEEP_TAGS` and `residual_phi` flags a dirty dataset**

Append to `tests/test_ingest_deid.py`:

```python
from src.ingest.deid import KEEP_TAGS, REMOVE_TAGS, residual_phi
from tests.fixtures.synthetic_dicom import make_xa_dataset, write_dataset


def test_remove_and_keep_tags_do_not_overlap():
    assert not (set(REMOVE_TAGS) & set(KEEP_TAGS))
    # B6: vendor identity is retained, never scrubbed.
    assert "Manufacturer" in KEEP_TAGS and "Manufacturer" not in REMOVE_TAGS
    assert "ManufacturerModelName" in KEEP_TAGS
    # PatientID is replaced with the pseudonym, not emptied, so it is in neither list.
    assert "PatientID" not in REMOVE_TAGS and "PatientID" not in KEEP_TAGS


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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ImportError: cannot import name 'KEEP_TAGS' from 'src.ingest.deid'`

Implement — append to `src/ingest/deid.py`:

```python
#: PS3.15 Annex E "Z" action: the element stays, its value is emptied. Emptying rather than
#: deleting keeps the dataset conformant and makes the absence explicit to a downstream reader.
#: Times are emptied outright -- a time of day is no clinical use here and is a re-identification
#: handle when combined with a hospital schedule. StudyDescription is emptied because in practice
#: it carries patient names typed in at booking. PatientID is NOT here: it is replaced with the
#: pseudonym so the de-identified file stays self-describing.
REMOVE_TAGS = (
    "PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "PatientBirthName", "PatientMotherBirthName",
    "AccessionNumber", "StudyID",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "NameOfPhysiciansReadingStudy", "PhysiciansOfRecord",
    "RequestingPhysician", "OperatorsName",
    "StudyDescription", "AdmissionID", "PatientComments", "ImageComments",
    "DeviceSerialNumber", "StationName",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
)

#: Clinically or methodologically required -- these must survive the scrub untouched.
#: Manufacturer and ManufacturerModelName are retained deliberately (Dialygo B6): vendor identity
#: is not patient identity, and leave-one-site-out external validation cannot be run without it.
KEEP_TAGS = (
    "Modality", "Manufacturer", "ManufacturerModelName",
    "PatientSex", "SeriesDescription",
    "KVP", "ExposureTime", "DistanceSourceToDetector", "DistanceSourceToPatient",
    "PositionerPrimaryAngle", "ImagerPixelSpacing", "CineRate", "FrameTime",
    "Rows", "Columns", "NumberOfFrames", "BitsAllocated", "BitsStored",
    "PhotometricInterpretation", "WindowCenter", "WindowWidth",
)

#: Shifted by the patient's stable offset, never deleted -- intervals are the clinical signal.
DATE_TAGS = ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate")

DEIDENTIFICATION_METHOD = (
    "Dialygo ingest: PS3.15 Annex E basic profile; private tags and 60xx overlays removed; "
    "identifying elements emptied; dates shifted per-patient; UIDs remapped under 2.25.; "
    "Manufacturer/ManufacturerModelName retained for site-level external validation"
)


def residual_phi(ds):
    """Return the REMOVE_TAGS keywords still carrying a value. An empty list means clean.

    This is the post-condition check for deid_dataset and the gate Phase 3 runs before writing a
    single frame: if it is non-empty, the instance is refused rather than exported.
    """
    from pydicom.datadict import tag_for_keyword

    dirty = []
    for kw in REMOVE_TAGS:
        tag = tag_for_keyword(kw)
        if tag is None or tag not in ds:
            continue
        value = ds[tag].value
        if value is not None and str(value).strip() != "":
            dirty.append(kw)
    return dirty
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (37 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): PS3.15 Annex E remove/keep tag lists and residual_phi check (Task 8)"
```

---

- [ ] **Step 2: `deid_dataset` empties every identifying element and reports clean**

Append to `tests/test_ingest_deid.py`:

```python
from src.ingest.deid import DEIDENTIFICATION_METHOD, deid_dataset


def _read_back(ds, tmp_path, name="case.dcm"):
    """Round-trip through Part-10 so file_meta exists exactly as it would on the drive."""
    import pydicom

    path = write_dataset(ds, tmp_path / name)
    return pydicom.dcmread(str(path))


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
    assert clean.DeidentificationMethod == DEIDENTIFICATION_METHOD
    assert set(ids) == {"real_patient", "pseudo_patient", "pseudo_study",
                        "pseudo_series", "pseudo_sop"}


def test_deid_dataset_refuses_an_instance_with_no_patient_id(tmp_path):
    """Fail-safe: no PatientID means no stable pseudonym and no stable date offset."""
    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)
    ds.PatientID = ""

    with pytest.raises(ValueError, match="PatientID"):
        deid_dataset(ds, SALT_A, site="inu")
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: FAIL — ImportError: cannot import name 'DEIDENTIFICATION_METHOD' from 'src.ingest.deid'`
(after adding the constant in Step 1 this becomes
`ImportError: cannot import name 'deid_dataset' from 'src.ingest.deid'`)

Implement — append to `src/ingest/deid.py`:

```python
def deid_dataset(ds, salt, *, site):
    """De-identify `ds` in place and return (ds, ids).

    Order matters and is deliberate:

    1. private tags and 60xx overlay planes are removed FIRST -- private elements are
       vendor-defined and can hold anything, and an overlay is a bitmap, so a name rendered into
       an overlay plane survives every tag-level scrub. Both go before anything else can read
       from them or copy a value forward.
    2. identifying elements are emptied (Annex E "Z": element stays, value goes).
    3. dates are shifted by this patient's stable offset; times are already emptied in step 2.
    4. UIDs are remapped last, deterministically, so patient/study/series/instance relationships
       survive intact under the new identifiers.

    PatientID is replaced with the pseudonym (not emptied) so the file stays self-describing, and
    file_meta.MediaStorageSOPInstanceUID is updated to match the remapped SOP UID -- leaving it
    stale would leak the original UID and produce a file whose meta and dataset disagree.

    Raises ValueError when the instance has no PatientID: without one there is no stable pseudonym
    and no stable date offset, and guessing either would corrupt the cohort.
    """
    from pydicom.datadict import tag_for_keyword

    real_patient = str(getattr(ds, "PatientID", "") or "").strip()
    if not real_patient:
        raise ValueError(
            "instance has no PatientID; refusing to de-identify an unidentifiable instance"
        )

    # 1. private tags and overlay planes -- unenumerable content, destroyed first
    ds.remove_private_tags()
    for tag in [t for t in list(ds.keys()) if 0x6000 <= t.group <= 0x601F]:
        del ds[tag]

    # 2. empty the identifying elements
    for kw in REMOVE_TAGS:
        tag = tag_for_keyword(kw)
        if tag is not None and tag in ds:
            ds[tag].value = ""

    # 3. shift the dates by this patient's stable offset
    offset = day_offset(salt, real_patient)
    for kw in DATE_TAGS:
        tag = tag_for_keyword(kw)
        if tag is not None and tag in ds:
            ds[tag].value = shift_date(ds[tag].value, offset)

    # 4. remap the UIDs, preserving structure
    def _remap_in_place(keyword):
        tag = tag_for_keyword(keyword)
        if tag is None or tag not in ds:
            return ""
        old = str(ds[tag].value or "").strip()
        if not old:
            return ""
        new = remap_uid(salt, old)
        ds[tag].value = new
        return new

    pseudo_study = _remap_in_place("StudyInstanceUID")
    pseudo_series = _remap_in_place("SeriesInstanceUID")
    pseudo_sop = _remap_in_place("SOPInstanceUID")

    meta = getattr(ds, "file_meta", None)
    if meta is not None and pseudo_sop:
        meta.MediaStorageSOPInstanceUID = pseudo_sop

    pseudo_patient = pseudo_id(salt, real_patient, site=site, kind="P")
    ds.PatientID = pseudo_patient
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = DEIDENTIFICATION_METHOD

    return ds, {
        "real_patient": real_patient,
        "pseudo_patient": pseudo_patient,
        "pseudo_study": pseudo_study,
        "pseudo_series": pseudo_series,
        "pseudo_sop": pseudo_sop,
    }
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (39 passed)`

Commit.

```bash
git add src/ingest/deid.py tests/test_ingest_deid.py
git commit -m "feat(ingest): deid_dataset implements the PS3.15 Annex E scrub in fixed order (Task 8)"
```

---

- [ ] **Step 3: `PatientID` becomes the pseudonym and `ids` matches the key material**

Append to `tests/test_ingest_deid.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (41 passed)` — `deid_dataset` already assigns the pseudonym; this step pins the
cross-batch stability guarantee that the whole HMAC design exists to provide.

Commit.

```bash
git add tests/test_ingest_deid.py
git commit -m "test(ingest): pin pseudonym stability across separate handover batches (Task 8)"
```

---

- [ ] **Step 4: clinically required tags survive — including vendor identity (B6)**

Append to `tests/test_ingest_deid.py`:

```python
def test_clinically_required_tags_survive_the_scrub(tmp_path):
    ds = _read_back(make_xa_dataset("INU-00417"), tmp_path)

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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (43 passed)`

Commit.

```bash
git add tests/test_ingest_deid.py
git commit -m "test(ingest): assert B6 vendor retention and acquisition tag survival through deid (Task 8)"
```

---

- [ ] **Step 5: UID remapping preserves study/series structure**

Append to `tests/test_ingest_deid.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (45 passed)`

Commit.

```bash
git add tests/test_ingest_deid.py
git commit -m "test(ingest): verify UID remapping preserves study/series grouping (Task 8)"
```

---

- [ ] **Step 6: dates shifted, the 31-day interval preserved, the original gone**

Append to `tests/test_ingest_deid.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (48 passed)` — if `test_two_patients_get_different_shifts` fails, the two synthetic
IDs happened to collide on one of 365 offsets; substitute another patient ID rather than weakening
the assertion.

Commit.

```bash
git add tests/test_ingest_deid.py
git commit -m "test(ingest): verify per-patient date shifting preserves intervals only (Task 8)"
```

---

- [ ] **Step 7: private tags and overlay planes are destroyed**

Append to `tests/test_ingest_deid.py`:

```python
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
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (50 passed)`

Commit.

```bash
git add tests/test_ingest_deid.py
git commit -m "test(ingest): verify private tags and 60xx overlay planes are removed first (Task 8)"
```

---

- [ ] **Step 8: `file_meta` SOP UID matches the dataset, and the file survives a round-trip**

Append to `tests/test_ingest_deid.py`:

```python
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

    ds = _read_back(make_xa_dataset("INU-00417", sop_uid="1.2.3.4.5.1"), tmp_path)
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
    # nothing from the original identity remains anywhere in the file bytes
    raw = out.read_bytes()
    assert b"REDDY" not in raw
    assert b"INU-00417" not in raw
    assert b"Institute of Nephro-Urology" not in raw
```

Run it.

```bash
python -m pytest tests/test_ingest_deid.py -q
```

`Expected: PASS (52 passed)`

Then run the whole suite to confirm nothing regressed.

```bash
python -m pytest -q
```

`Expected: PASS (507 passed)` — the 374 pre-existing tests, plus the 81 added across Tasks 1-4,
plus the 52 added across Tasks 5-8. If your total differs, check the per-task counts above before
assuming a regression: the cumulative figure only holds when tasks run in plan order.

Commit.

```bash
git add tests/test_ingest_deid.py
git commit -m "test(ingest): assert file_meta SOP UID sync and PHI-free on-disk round-trip (Task 8)"
```

### Task 9: `pixel_deid.py` — burned-in overlay text detection and masking

Tag-level de-identification (Task 7) strips PHI from the DICOM *header*. It does nothing about PHI burned into the *pixels*. Cath-lab and angio workstations routinely render the patient name, the study date, the accession number and the institution directly into the image raster before export — the text is part of the picture, not part of a tag. Worse, the one tag that is supposed to warn us about this, `BurnedInAnnotation`, is unreliable in practice: on real institutional exports it is frequently absent altogether, or present and set to `NO` while a full patient banner sits across the top of every frame. Trusting that tag is how PHI reaches a training set. So `pixel_deid` screens **every frame regardless of what the tag says**, and the tag is used only as an additional reason to escalate, never as a reason to skip.

Detection is deliberately **OCR-free**. Adding Tesseract or an OCR model would pull a heavy new dependency and a new failure mode into the one module whose job is to be boringly reliable, and we do not need to *read* the text — we only need to know where it is so we can destroy it. The screen is therefore geometric: overlay banners live at the very top or the very bottom of the frame, so only the top and bottom `SCREEN_FRACTION` (0.15) bands are examined; overlay glyphs are rendered at or near full white, so the band is thresholded for near-saturated pixels; individual glyphs are separated by small gaps, so an OpenCV morphological close with a wide, short rectangular kernel joins the letters of a word into one run; and a *text run* is wider than it is tall, so connected components are kept only when `w > h` and the component clears a minimum area. That last rule is what keeps the vessel out: the contrast-filled artery is a thin diagonal, so within a 10-row band it produces a component roughly as wide as it is tall and is rejected, while a burned-in banner produces one component spanning the full frame width.

`cv2` and `numpy` are imported **inside** the functions, per the module-level import rule. `mask_regions` returns a **copy** with the boxes zeroed and must never mutate its input — the caller in Task 10 iterates frames from `ds.pixel_array`, and an in-place write there would silently corrupt the dataset for every later consumer. `needs_review` is the fail-safe escalation: it returns `True` when the tag says `YES` **or** when any box was found, and also when there is no header to consult at all. A false positive costs a human one glance at one frame; a false negative leaks a patient's name into a model's training data and into every artifact derived from it. The asymmetry is not close, so the flag is biased hard toward review.

**Files:**
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/pixel_deid.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_pixel_deid.py`

**Interfaces:**
- Consumes: `tests/fixtures/synthetic_dicom.make_xa_dataset(patient_id=..., *, n_frames, rows, cols, burned_in, ...) -> pydicom.Dataset` (Task 1); `src/ingest/clearance.require_clearance(mode, clearance_path=...) -> None` (CLI only)
- Produces:
  - `SCREEN_FRACTION = 0.15`
  - `detect_text_regions(arr) -> list[tuple[int, int, int, int]]` — `arr` is a 2-D uint8 frame; boxes are `(x, y, w, h)` in **full-frame** coordinates
  - `mask_regions(arr, boxes) -> numpy.ndarray` — copy with every box zeroed; input untouched
  - `needs_review(ds, boxes) -> bool`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the detection tests (they fail — the module does not exist yet).**

```python
# tests/test_ingest_pixel_deid.py
"""Burned-in overlay text: geometric detection, masking, and the fail-safe review flag.

Dialygo B5: every test here runs on the SYNTHETIC DICOM fixture only. No real study is ever
opened by the test suite.
"""
import numpy as np

from src.ingest.pixel_deid import SCREEN_FRACTION, detect_text_regions

from tests.fixtures.synthetic_dicom import make_xa_dataset

ROWS = COLS = 64


def _u8(frame):
    """Linear 12-bit -> 8-bit scale (no VOI LUT).

    The fixture's burned-in band is written at ``np.iinfo(np.uint16).max >> 4`` == 4095, i.e. full
    scale for BitsStored 12, so ``>> 4`` puts it at 255 while leaving the vessel below saturation.
    Deliberately NOT ``to_8bit`` (Task 10) -- this module must be testable on its own.
    """
    return np.clip(np.asarray(frame).astype(np.int32) >> 4, 0, 255).astype(np.uint8)


def _bands(h):
    """(top_end, bottom_start) row indices the module is expected to screen."""
    band = max(1, int(round(h * SCREEN_FRACTION)))
    return band, max(band, h - band)


def test_detects_the_burned_in_band_as_a_wide_text_run():
    ds = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS, burned_in=True)
    boxes = detect_text_regions(_u8(ds.pixel_array[0]))

    assert boxes, "the fixture's burned-in banner must be detected"
    wide = [b for b in boxes if b[2] >= COLS // 2]
    assert wide, f"a full-width banner must yield a wide box, got {boxes}"
    x, y, w, h = wide[0]
    top_end, _ = _bands(ROWS)
    assert y < top_end, f"the banner sits in the TOP screened band, got y={y}"
    assert y + h <= top_end, "a top-band box must not extend past the screened band"


def test_clean_frame_yields_no_wide_run_and_nothing_in_the_interior():
    # The clean fixture holds only the diagonal 'vessel'. A diagonal is as tall as it is wide inside
    # a screening band, so it must never be mistaken for a line of text.
    ds = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS, burned_in=False)
    boxes = detect_text_regions(_u8(ds.pixel_array[0]))

    assert not [b for b in boxes if b[2] >= COLS // 2], f"vessel must not read as text, got {boxes}"
    top_end, bot_start = _bands(ROWS)
    for x, y, w, h in boxes:
        assert y + h <= top_end or y >= bot_start, f"box {(x, y, w, h)} leaked into the interior"


def test_boxes_are_confined_to_the_screen_fraction_bands():
    ds = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS, burned_in=True)
    top_end, bot_start = _bands(ROWS)
    for x, y, w, h in detect_text_regions(_u8(ds.pixel_array[0])):
        assert 0 <= x and x + w <= COLS, "box must stay inside the frame horizontally"
        assert y + h <= top_end or y >= bot_start, "only the top/bottom bands are screened"


def test_malformed_input_degrades_to_no_boxes():
    # Fail-safe: a colour/3-D array or an empty array must not raise -- it returns nothing to mask
    # and needs_review (below) is what keeps such a frame out of the clean store.
    assert detect_text_regions(np.zeros((0, 0), np.uint8)) == []
    assert detect_text_regions(np.zeros((8, 8, 3), np.uint8)) == []
```

- [ ] **Step 2: Run it and confirm the expected failure.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_pixel_deid.py -q
```

Expected: FAIL — `ImportError while importing test module '.../tests/test_ingest_pixel_deid.py' ... ModuleNotFoundError: No module named 'src.ingest.pixel_deid'` (1 error).

- [ ] **Step 3: Implement the screen.**

```python
# src/ingest/pixel_deid.py
"""Screen angiographic frames for BURNED-IN overlay text and mask it out.

Header de-identification (``src.ingest.deid``) cannot touch PHI that the cath-lab workstation
rendered into the pixel raster -- patient name, study date, accession, institution -- and the
``BurnedInAnnotation`` tag that is supposed to warn about it is unreliable on real exports (often
absent, often ``NO`` while a banner is plainly visible). So every frame is screened regardless of
the tag, and the tag only ever ADDS a reason to escalate.

The screen is intentionally OCR-free -- no new heavy dependency, and we never need to READ the
text, only locate it:
  1. look only at the top and bottom ``SCREEN_FRACTION`` bands (overlays live at the frame edges);
  2. threshold for near-saturated pixels (overlay glyphs are rendered at ~full white);
  3. morphological CLOSE with a wide, short kernel to join glyphs into word/line runs;
  4. keep connected components that are WIDER THAN TALL and above a minimum area.

Rule 4 is what rejects the contrast-filled vessel: a thin diagonal inside a ~10-row band produces a
component about as wide as it is tall, while a banner spans the full frame width.

``cv2``/``numpy`` are imported inside functions so this module stays torch-free AND cv2-free at
import time. Runs standalone: ``python -m src.ingest.pixel_deid <dicom> [--clearance PATH]``.
"""
import os

SCREEN_FRACTION = 0.15          # top/bottom fraction of the frame that is screened for text
SATURATION_FRACTION = 0.90      # pixels at >=90% of full 8-bit scale count as "overlay bright"
MIN_BOX_AREA = 12               # px; below this it is speckle, not a glyph run
CLOSE_KERNEL = (9, 3)           # (w, h) rect: joins glyphs horizontally, never vertically


def _bands(h):
    """Return ``(top_end, bottom_start)`` row indices of the two screened bands."""
    band = max(1, int(round(h * SCREEN_FRACTION)))
    return band, max(band, h - band)


def detect_text_regions(arr):
    """Locate burned-in text runs in a 2-D uint8 frame.

    Returns a sorted list of ``(x, y, w, h)`` boxes in FULL-FRAME coordinates (``y`` is already
    offset back out of the band it was found in). Malformed input -> ``[]`` (never raises): the
    caller's ``needs_review`` is what keeps an unscreenable frame out of the clean store.
    """
    import cv2
    import numpy as np

    a = np.asarray(arr)
    if a.ndim != 2 or a.size == 0:
        return []
    a = a.astype(np.uint8, copy=False)
    h, w = a.shape
    top_end, bot_start = _bands(h)
    thresh = int(round(255 * SATURATION_FRACTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)

    boxes = []
    for y0, y1 in ((0, top_end), (bot_start, h)):
        if y1 <= y0:
            continue
        band = a[y0:y1]
        mask = (band >= thresh).astype(np.uint8) * 255
        if not mask.any():
            continue
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for i in range(1, n_labels):                       # label 0 is the background
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if bw <= bh or area < MIN_BOX_AREA:            # not a text run (vessel, speckle)
                continue
            boxes.append((bx, y0 + by, bw, bh))
    return sorted(boxes)
```

- [ ] **Step 4: Run the detection tests.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_pixel_deid.py -q
```

Expected: PASS (4 passed)

- [ ] **Step 5: Add the masking and review-flag tests.** Replace the import line at the top of `tests/test_ingest_pixel_deid.py` with the first block, then append the second block to the end of the file.

```python
from src.ingest.pixel_deid import (SCREEN_FRACTION, detect_text_regions, mask_regions,
                                   needs_review)
```

```python
# --- masking: destroy the pixels, keep the anatomy, never touch the caller's array --------------

def test_mask_zeroes_the_burned_in_band_and_keeps_the_vessel():
    ds = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS, burned_in=True)
    frame = _u8(ds.pixel_array[0])
    boxes = detect_text_regions(frame)

    masked = mask_regions(frame, boxes)

    top_end, bot_start = _bands(ROWS)
    assert masked[0:8, :].max() == 0, "the fixture's 8-row banner must be fully zeroed"
    interior = slice(top_end, bot_start)
    assert np.array_equal(masked[interior], frame[interior]), "the interior must be untouched"
    assert masked[interior].max() > 0, "the diagonal vessel must survive masking"


def test_mask_regions_does_not_mutate_its_input():
    # Task 10 masks frames straight off ds.pixel_array; an in-place write would corrupt the
    # dataset for every later consumer.
    ds = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS, burned_in=True)
    frame = _u8(ds.pixel_array[0])
    before = frame.copy()

    out = mask_regions(frame, detect_text_regions(frame))

    assert np.array_equal(frame, before), "mask_regions must not modify the array it was given"
    assert out is not frame
    assert not np.array_equal(out, frame), "the copy must actually differ (banner zeroed)"


def test_mask_regions_clips_boxes_to_the_frame():
    frame = np.full((ROWS, COLS), 200, np.uint8)
    out = mask_regions(frame, [(-5, -5, 10, 10), (COLS - 2, ROWS - 2, 50, 50), (0, 0, 0, 0)])
    assert out.shape == frame.shape
    assert out[0:5, 0:5].max() == 0 and out[ROWS - 2:, COLS - 2:].max() == 0
    assert out[ROWS // 2, COLS // 2] == 200, "clipping must not blank the whole frame"


# --- needs_review: fail-safe escalation ---------------------------------------------------------

def test_needs_review_true_when_tag_says_yes_even_with_no_boxes():
    ds = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS, burned_in=True)
    assert str(ds.BurnedInAnnotation).upper() == "YES"
    assert needs_review(ds, []) is True


def test_needs_review_true_when_boxes_found_even_when_tag_says_no():
    # The whole point: the tag lies. Pixels win.
    ds = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS, burned_in=False)
    ds.BurnedInAnnotation = "NO"
    assert needs_review(ds, [(0, 0, 64, 8)]) is True


def test_needs_review_false_only_when_tag_is_clean_and_nothing_was_found():
    ds = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS, burned_in=False)
    ds.BurnedInAnnotation = "NO"
    assert needs_review(ds, []) is False


def test_needs_review_true_when_there_is_no_header_to_check():
    assert needs_review(None, []) is True, "no header -> assume unscreened, defer to a human"
```

- [ ] **Step 6: Run and confirm the expected failure.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_pixel_deid.py -q
```

Expected: FAIL — `ImportError while importing test module '.../tests/test_ingest_pixel_deid.py' ... ImportError: cannot import name 'mask_regions' from 'src.ingest.pixel_deid'` (1 error).

- [ ] **Step 7: Implement masking, the review flag, and the CLI.** Append to `src/ingest/pixel_deid.py`.

```python
def mask_regions(arr, boxes):
    """Return a COPY of ``arr`` with every ``(x, y, w, h)`` box set to 0.

    Never mutates ``arr``. Boxes are clipped to the frame, so an over-wide or negative box blanks
    only the overlapping region instead of raising or wrapping around.
    """
    import numpy as np

    out = np.array(arr, copy=True)
    if out.ndim != 2 or out.size == 0:
        return out
    h, w = out.shape
    for box in boxes or ():
        try:
            bx, by, bw, bh = (int(v) for v in box)
        except (TypeError, ValueError):
            continue                                       # malformed box -> skip, never raise
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(w, x0 + max(0, bw)), min(h, y0 + max(0, bh))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
    return out


def needs_review(ds, boxes):
    """True when this series must be looked at by a human before it enters the clean store.

    Fail-safe by construction: ``True`` if ``BurnedInAnnotation`` says YES, OR if the pixel screen
    found anything, OR if there is no header to consult. A false positive costs one human glance;
    a false negative leaks a patient's name into the training set and everything downstream of it.
    """
    if ds is None:
        return True
    tag = str(getattr(ds, "BurnedInAnnotation", "") or "").strip().upper()
    return tag == "YES" or bool(boxes)


def main(argv=None):
    """CLI: screen every frame of one DICOM and print the boxes that would be masked."""
    import argparse
    import json

    import numpy as np
    import pydicom

    from src.ingest.clearance import require_clearance

    ap = argparse.ArgumentParser(description="Screen a DICOM's frames for burned-in overlay text.")
    ap.add_argument("dicom", help="path to a DICOM file")
    ap.add_argument("--clearance", default=None, help="path to the signed clearance record")
    args = ap.parse_args(argv)
    require_clearance("read", **({"clearance_path": args.clearance} if args.clearance else {}))

    ds = pydicom.dcmread(args.dicom)
    arr = np.asarray(ds.pixel_array)
    stack = arr if arr.ndim == 3 else arr[None, ...]
    detections = []
    for i in range(stack.shape[0]):
        f = stack[i].astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        u8 = (np.full(f.shape, 128, np.uint8) if hi <= lo
              else np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8))
        boxes = [list(b) for b in detect_text_regions(u8)]
        detections.append({"frame": i, "boxes": boxes})
    any_boxes = any(d["boxes"] for d in detections)
    print(json.dumps({"source": os.path.basename(args.dicom),
                      "frames": int(stack.shape[0]),
                      "burned_in_tag": str(getattr(ds, "BurnedInAnnotation", "") or ""),
                      "review_required": needs_review(ds, [1] if any_boxes else []),
                      "detections": detections}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run the full module test file.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_pixel_deid.py -q
```

Expected: PASS (11 passed)

- [ ] **Step 9: Smoke-test the CLI entry point and commit.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && \
python -m src.ingest.pixel_deid --help && \
git add src/ingest/pixel_deid.py tests/test_ingest_pixel_deid.py && \
git commit -m "feat(ingest): screen and mask burned-in overlay text (pixel_deid)

BurnedInAnnotation is unreliable on real exports, so every frame is screened
regardless of the tag. OCR-free geometric screen: top/bottom SCREEN_FRACTION
bands -> saturation threshold -> morphological close -> wider-than-tall
connected components. mask_regions returns a copy; needs_review escalates on
tag OR pixels OR a missing header.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: `--help` prints the usage line; commit succeeds.

---

### Task 10: `extract.py` — VOI-LUT windowing and multi-frame → PNG

Cath cine is stored as MONOCHROME2 with 8–12 *significant* bits packed into 16-bit words, and the meaningful diagnostic contrast occupies only a narrow slice of that range. The DICOM header carries the window the acquisition intended (`WindowCenter` / `WindowWidth`, and sometimes an explicit `VOILUTSequence`). **Skipping that VOI LUT is the single most common reason angiography frames come out washed out** — code that just does `arr / arr.max() * 255` maps the whole 12-bit range linearly, dumps almost all of the dynamic range into pixel values nobody uses, and produces flat grey images where the vessel is barely separable from the background. So `to_8bit` applies `pydicom.pixels.apply_voi_lut` (falling back to `pydicom.pixel_data_handlers.util.apply_voi_lut` on pydicom 2.x, where the newer module does not exist), then inverts when `PhotometricInterpretation == "MONOCHROME1"` — in MONOCHROME1 the *minimum* value is white, so an uninverted frame is a photographic negative and every downstream contrast assumption is backwards — and only then min-max scales to 0–255 uint8. When a frame is constant (a dropped frame, a fully blanked frame, an all-black lead-in) the min-max denominator is zero; rather than divide by zero, emit a flat mid-grey array. That is the fail-safe default: an obviously blank frame is honest, a NaN-poisoned or exception-raising frame is not.

`extract_series` writes ordered, zero-padded PNGs `f00000.png, f00001.png, …` into `frames/<stem_prefix>/`. The zero padding and the strict correspondence between array index and filename index are load-bearing: **temporal order is preserved on disk**, so later temporal work (frame voting, motion) can reconstruct the sequence by sorting filenames, while frame-level tasks glob exactly the same tree and treat each PNG as an independent still. One layout serves both; there is no second export path to keep in sync. `extract_series` accepts the optional `mask_boxes` produced by Task 9 and applies them to **every** frame, because an overlay banner is burned into the whole cine, not one frame of it. It returns `{"stem_prefix", "n_frames", "dir", "review_required"}`. `write_sidecar` records the pseudo IDs, vendor, geometry, `FrameTime`/derived fps, a SHA-256 of the source pixel data, and the de-identification method string, so any frame on disk can be traced back to a provenance record without ever touching the original drive again.

One thing extraction deliberately does **not** do: **B3's "cropped to the segment of interest" crop is NOT automated here.** Which part of a fistulography run is *the segment of interest* — juxta-anastomotic, cannulation zone, outflow vein, central vein — is a clinical judgement, and it arrives with the labels in Task 13, not from a heuristic in an ingest script. A guessed crop would silently discard the very stenosis it was meant to centre and there would be no way to notice after the fact. So extraction writes **full frames** and records `crop: {"applied": false, "reason": ...}` in the sidecar, making the deferral explicit and auditable rather than an unstated omission.

Note also what the sidecar does *not* record: the source filename. Filenames on institutional handover drives routinely contain the patient's name (`SMITH_JOHN_20240517.dcm`); recording one would re-introduce, in the clean store, exactly the PHI the tag de-identification just removed. Only the content hash is kept.

**Files:**
- Create: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/extract.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_extract.py`

**Interfaces:**
- Consumes: `src/ingest/manifest.write_json_atomic(path, obj)`, `src/ingest/manifest.provenance(tool, **extra)`; `src/ingest/pixel_deid.mask_regions(arr, boxes)`, `src/ingest/pixel_deid.needs_review(ds, boxes)`, `src/ingest/pixel_deid.detect_text_regions(arr)`; `src/ingest/deid.load_or_create_salt(path)`, `src/ingest/deid.deid_dataset(ds, salt, *, site)`; `src/ingest/clearance.require_clearance(mode, clearance_path=...)`
- Produces:
  - `to_8bit(arr2d, ds) -> numpy.ndarray` (uint8)
  - `stem_prefix(site, pseudo_patient, series_idx) -> str` — `"avf_<site>_<pid_hex10>_s<NN>"`
  - `frame_stem(prefix, frame_idx) -> str` — `"<prefix>_<FFFFF>"`
  - `write_sidecar(out_root, prefix, meta) -> str` — path of `<out_root>/sidecar/<prefix>.json`
  - `extract_series(ds, out_root, *, site, pseudo_patient, series_idx, mask_boxes=None) -> dict` with keys `stem_prefix`, `n_frames`, `dir`, `review_required`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the windowing and naming tests.**

```python
# tests/test_ingest_extract.py
"""DICOM cine -> ordered de-identified PNG frames + sidecar.

Dialygo B5: synthetic fixture only. The VOI LUT tests are the ones that matter -- skipping the
DICOM window is why angio frames come out washed out.
"""
import json
import os
import re

import numpy as np

from src.ingest.extract import frame_stem, stem_prefix, to_8bit, write_sidecar

from tests.fixtures.synthetic_dicom import make_xa_dataset

PID = "inu_3f9c21b04e"          # shape of deid.pseudo_id(...) output
ROWS = COLS = 64


def test_to_8bit_returns_uint8_spanning_the_full_display_range():
    ds = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS)
    out = to_8bit(ds.pixel_array[0], ds)

    assert out.dtype == np.uint8, f"PNG store is 8-bit, got {out.dtype}"
    assert out.shape == (ROWS, COLS)
    assert int(out.min()) == 0 and int(out.max()) == 255, "a non-constant frame must span 0..255"


def test_to_8bit_inverts_monochrome1():
    # In MONOCHROME1 the MINIMUM value is white. Without the invert the frame is a negative and
    # every downstream contrast assumption is backwards.
    ds2 = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS)
    ds2.PhotometricInterpretation = "MONOCHROME2"
    ds1 = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS)
    ds1.PhotometricInterpretation = "MONOCHROME1"
    frame = ds2.pixel_array[0]

    m2 = to_8bit(frame, ds2)
    m1 = to_8bit(frame, ds1)

    hot = np.unravel_index(int(np.argmax(m2)), m2.shape)
    cold = np.unravel_index(int(np.argmin(m2)), m2.shape)
    assert m2[hot] > m2[cold]
    assert m1[hot] < m1[cold], "MONOCHROME1: the brightest MONOCHROME2 pixel must be the darkest"
    assert int(np.abs(m1.astype(int) + m2.astype(int) - 255).max()) <= 1, "invert must be exact"


def test_to_8bit_constant_frame_is_flat_mid_grey_not_a_zero_division():
    ds = make_xa_dataset(n_frames=1, rows=ROWS, cols=COLS)
    flat = np.full((ROWS, COLS), 1000, dtype=np.uint16)

    out = to_8bit(flat, ds)

    assert out.dtype == np.uint8
    assert int(out.min()) == 128 and int(out.max()) == 128, "constant frame -> flat mid-grey"
    assert np.isfinite(out.astype(float)).all(), "never NaN/inf"


def test_stem_prefix_matches_the_locked_grammar():
    assert stem_prefix("inu", PID, 1) == "avf_inu_3f9c21b04e_s01"
    assert stem_prefix("inu", PID, 12) == "avf_inu_3f9c21b04e_s12"
    assert re.match(r"^avf_[a-z0-9]+_[0-9a-f]{10}_s\d{2}$", stem_prefix("inu", PID, 3))


def test_stem_prefix_does_not_double_the_site():
    # deid.pseudo_id already returns "<site>_<hex10>"; prefixing the site again would break the
    # AVF group-key regex in Task 12 and re-open the leakage hole.
    assert stem_prefix("inu", "inu_3f9c21b04e", 1) == "avf_inu_3f9c21b04e_s01"
    assert stem_prefix("inu", "3f9c21b04e", 1) == "avf_inu_3f9c21b04e_s01"


def test_frame_stem_is_zero_padded_to_five_digits():
    assert frame_stem("avf_inu_3f9c21b04e_s01", 12) == "avf_inu_3f9c21b04e_s01_00012"
    assert frame_stem("avf_inu_3f9c21b04e_s01", 0) == "avf_inu_3f9c21b04e_s01_00000"


def test_write_sidecar_lands_in_the_locked_layout(tmp_path):
    p = write_sidecar(str(tmp_path), "avf_inu_3f9c21b04e_s01", {"n_frames": 3, "provenance": "dicom"})

    assert p == os.path.join(str(tmp_path), "sidecar", "avf_inu_3f9c21b04e_s01.json")
    meta = json.load(open(p))
    assert meta["stem_prefix"] == "avf_inu_3f9c21b04e_s01"
    assert meta["n_frames"] == 3
    assert meta["deid_method"], "every sidecar must state how the frames were de-identified"
```

- [ ] **Step 2: Run it and confirm the expected failure.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_extract.py -q
```

Expected: FAIL — `ImportError while importing test module '.../tests/test_ingest_extract.py' ... ModuleNotFoundError: No module named 'src.ingest.extract'` (1 error).

- [ ] **Step 3: Implement windowing, the stem grammar, and the sidecar writer.**

```python
# src/ingest/extract.py
"""DICOM cine -> ordered, de-identified 8-bit PNG frames + a sidecar provenance record.

Canonical layout (locked):
    <clean_root>/<site>/frames/<stem_prefix>/f00000.png
    <clean_root>/<site>/sidecar/<stem_prefix>.json
Stem grammar:
    avf_<site>_<pid_hex10>_s<NN>_<FFFFF>   e.g. avf_inu_3f9c21b04e_s01_00012

Cath cine is MONOCHROME2 with 8-12 SIGNIFICANT bits in 16-bit words, and the diagnostic contrast
sits in a narrow slice of that range described by the header's VOI LUT. Ignoring that LUT is why
angio frames so often come out washed out, so ``to_8bit`` applies it before scaling.

B3 says model input is "cropped to the segment of interest". That crop is NOT automated here --
which part of a fistulography run matters (juxta-anastomotic, cannulation zone, outflow, central)
is a clinical decision that arrives with the labels (Task 13). Guessing would silently discard the
stenosis. Extraction writes FULL frames and records the deferral in the sidecar.

The sidecar never records the source filename: handover filenames routinely contain the patient
name, so only a content hash is kept.

``cv2``/``numpy``/``pydicom`` are imported inside functions. Runs standalone:
``python -m src.ingest.extract <dicom> --out-root <clean_root>/<site>``.
"""
import hashlib
import os

FRAME_PATTERN = "f%05d.png"
SIDECAR_SCHEMA = "dialygo.ingest.sidecar/1"
DEID_METHOD = "tag:src.ingest.deid + pixel-screen:src.ingest.pixel_deid"
CROP_DEFERRAL = ("B3 segment-of-interest crop is a clinical/annotation decision and arrives with "
                 "the labels (Task 13); ingest writes full frames")


def _as_float(v):
    """Best-effort float from a pydicom value (DSfloat, MultiValue, str). None when unusable."""
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _jsonable(v):
    """Coerce pydicom value types (DSfloat/IS/MultiValue/UID/PersonName) to plain JSON types."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _sha256_bytes(blob):
    h = hashlib.sha256()
    h.update(blob)
    return h.hexdigest()


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def to_8bit(arr2d, ds):
    """VOI LUT -> MONOCHROME1 invert -> min-max to 0..255 uint8.

    A constant frame (dropped/blanked/lead-in) returns a flat mid-grey array instead of dividing by
    zero -- an obviously blank frame is honest, a NaN-poisoned one is not.
    """
    import numpy as np

    try:                                                   # pydicom >= 3
        from pydicom.pixels import apply_voi_lut
    except ImportError:                                    # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_voi_lut

    raw = np.asarray(arr2d)
    try:
        a = np.asarray(apply_voi_lut(raw, ds))
    except Exception:                                      # absent or broken LUT -> raw values
        a = raw
    a = a.astype(np.float32, copy=False)

    if str(getattr(ds, "PhotometricInterpretation", "") or "").strip().upper() == "MONOCHROME1":
        a = float(a.max()) - a                             # MONOCHROME1: minimum value is WHITE

    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.full(a.shape, 128, dtype=np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def stem_prefix(site, pseudo_patient, series_idx):
    """``avf_<site>_<pid_hex10>_s<NN>`` -- the per-series half of the locked stem grammar.

    ``deid.pseudo_id`` already returns ``"<site>_<hex10>"``, so the site is prepended only when it
    is not already there. Doubling it would break the AVF group-key regex (Task 12) and silently
    re-open the patient-leakage hole.
    """
    s = str(site).strip().lower()
    pid = str(pseudo_patient).strip().lower()
    if s and not pid.startswith(s + "_"):
        pid = f"{s}_{pid}"
    return f"avf_{pid}_s{int(series_idx):02d}"


def frame_stem(prefix, frame_idx):
    """``<prefix>_<FFFFF>`` -- the logical stem of one frame (what group_key/split_of see)."""
    return f"{prefix}_{int(frame_idx):05d}"


def write_sidecar(out_root, prefix, meta):
    """Write ``<out_root>/sidecar/<prefix>.json`` atomically. Returns the path."""
    from src.ingest.manifest import provenance, write_json_atomic

    d = os.path.join(str(out_root), "sidecar")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{prefix}.json")
    payload = {"schema": SIDECAR_SCHEMA,
               "stem_prefix": prefix,
               "deid_method": DEID_METHOD,
               "crop": {"applied": False, "reason": CROP_DEFERRAL}}
    payload.update(dict(meta or {}))
    payload.setdefault("tool", provenance("src.ingest.extract"))
    write_json_atomic(path, {k: _jsonable(v) for k, v in payload.items()})
    return path
```

- [ ] **Step 4: Run the windowing and naming tests.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_extract.py -q
```

Expected: PASS (7 passed)

- [ ] **Step 5: Add the `extract_series` tests.** Replace the `src.ingest.extract` import line with the first block, then append the second block.

```python
from src.ingest.extract import (extract_series, frame_stem, stem_prefix, to_8bit, write_sidecar)
```

```python
# --- extract_series: ordered PNGs, masking, sidecar ---------------------------------------------

def _set_pixels(ds, arr):
    """Replace the fixture's pixel data and drop pydicom's decode cache."""
    ds.PixelData = np.ascontiguousarray(arr.astype(np.uint16)).tobytes()
    ds._pixel_array = None
    ds._pixel_id = None
    return ds


def _barcode(ds, n_frames, rows, cols):
    """Overwrite the cine with an unambiguous per-frame 'barcode': frame i has i+1 saturated px."""
    arr = np.zeros((n_frames, rows, cols), np.uint16)
    for i in range(n_frames):
        arr[i, 2, 0:i + 1] = 4095
    return _set_pixels(ds, arr)


def test_extract_series_writes_one_zero_padded_png_per_frame(tmp_path):
    ds = make_xa_dataset(n_frames=8, rows=ROWS, cols=COLS)

    res = extract_series(ds, str(tmp_path), site="inu", pseudo_patient=PID, series_idx=1)

    assert res["stem_prefix"] == "avf_inu_3f9c21b04e_s01"
    assert res["n_frames"] == 8
    expected_dir = os.path.join(str(tmp_path), "frames", "avf_inu_3f9c21b04e_s01")
    assert res["dir"] == expected_dir
    names = sorted(os.listdir(expected_dir))
    assert names == [f"f{i:05d}.png" for i in range(8)], f"unexpected frame names: {names}"


def test_extract_series_preserves_temporal_order_on_disk(tmp_path):
    # Sorted filenames must reconstruct the acquisition order -- later temporal work depends on it.
    import cv2
    n = 6
    ds = _barcode(make_xa_dataset(n_frames=n, rows=ROWS, cols=COLS), n, ROWS, COLS)

    res = extract_series(ds, str(tmp_path), site="inu", pseudo_patient=PID, series_idx=2)

    counts = []
    for name in sorted(os.listdir(res["dir"])):
        png = cv2.imread(os.path.join(res["dir"], name), cv2.IMREAD_UNCHANGED)
        assert png.ndim == 2 and png.dtype == np.uint8, "frames are single-channel 8-bit"
        counts.append(int((png[2] == 255).sum()))
    assert counts == [i + 1 for i in range(n)], f"frames written out of order: {counts}"


def test_extract_series_applies_mask_boxes_to_every_frame(tmp_path):
    import cv2
    ds = make_xa_dataset(n_frames=5, rows=ROWS, cols=COLS, burned_in=True)

    res = extract_series(ds, str(tmp_path), site="inu", pseudo_patient=PID, series_idx=1,
                         mask_boxes=[(0, 0, COLS, 8)])

    for name in sorted(os.listdir(res["dir"])):
        png = cv2.imread(os.path.join(res["dir"], name), cv2.IMREAD_UNCHANGED)
        assert int(png[0:8, :].max()) == 0, f"{name}: banner survived masking"
        assert int(png[8:, :].max()) > 0, f"{name}: masking blanked the whole frame"
    assert res["review_required"] is True


def test_extract_series_sidecar_records_ids_geometry_fps_hash_and_crop_deferral(tmp_path):
    ds = make_xa_dataset(n_frames=4, rows=ROWS, cols=COLS, manufacturer="Siemens")
    ds.FrameTime = "33.333"

    res = extract_series(ds, str(tmp_path), site="inu", pseudo_patient=PID, series_idx=3)

    meta = json.load(open(os.path.join(str(tmp_path), "sidecar", res["stem_prefix"] + ".json")))
    assert meta["pseudo_patient"] == PID
    assert meta["pseudo_series"] and meta["pseudo_sop"], "pseudo UIDs must be traceable"
    assert meta["manufacturer"] == "Siemens" and meta["modality"] == "XA"
    assert meta["rows"] == ROWS and meta["columns"] == COLS and meta["n_frames"] == 4
    assert abs(meta["frame_time_ms"] - 33.333) < 1e-3
    assert abs(meta["fps"] - 30.0) < 0.01, f"fps must derive from FrameTime, got {meta['fps']}"
    assert re.fullmatch(r"[0-9a-f]{64}", meta["source_sha256"]), "source hash must be sha256 hex"
    assert meta["provenance"] == "dicom" and meta["dicom_metadata"] is True
    assert meta["crop"]["applied"] is False and meta["crop"]["reason"]
    assert meta["deid_method"] == "tag:src.ingest.deid + pixel-screen:src.ingest.pixel_deid"
    assert "source_name" not in meta, "handover filenames carry PHI -- only the hash is kept"
    assert meta["frame_stem_pattern"] == frame_stem(res["stem_prefix"], 0)[:-5] + "%05d"


def test_extract_series_flags_review_when_the_burned_in_tag_is_set(tmp_path):
    # No boxes passed in at all -- the tag alone must still escalate (fail-safe).
    ds = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS, burned_in=True)
    res = extract_series(ds, str(tmp_path), site="inu", pseudo_patient=PID, series_idx=1)
    assert res["review_required"] is True

    clean = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS, burned_in=False)
    clean.BurnedInAnnotation = "NO"
    res2 = extract_series(clean, str(tmp_path), site="inu", pseudo_patient=PID, series_idx=9)
    assert res2["review_required"] is False
```

- [ ] **Step 6: Run and confirm the expected failure.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_extract.py -q
```

Expected: FAIL — `ImportError while importing test module '.../tests/test_ingest_extract.py' ... ImportError: cannot import name 'extract_series' from 'src.ingest.extract'` (1 error).

- [ ] **Step 7: Implement `extract_series` and the CLI.** Append to `src/ingest/extract.py`.

```python
def extract_series(ds, out_root, *, site, pseudo_patient, series_idx, mask_boxes=None):
    """Write every frame of ``ds`` as an ordered PNG under ``<out_root>/frames/<stem_prefix>/``.

    ``mask_boxes`` (from ``pixel_deid.detect_text_regions``) is applied to EVERY frame -- an overlay
    banner is burned into the whole cine, not one frame of it. Returns
    ``{"stem_prefix", "n_frames", "dir", "review_required"}``.
    """
    import cv2
    import numpy as np

    from src.ingest.pixel_deid import mask_regions, needs_review

    prefix = stem_prefix(site, pseudo_patient, series_idx)
    frames_dir = os.path.join(str(out_root), "frames", prefix)
    os.makedirs(frames_dir, exist_ok=True)

    arr = np.asarray(ds.pixel_array)
    stack = arr if arr.ndim == 3 else arr[None, ...]
    boxes = [tuple(int(v) for v in b) for b in (mask_boxes or ())]

    n = 0
    for i in range(stack.shape[0]):
        frame = to_8bit(stack[i], ds)
        if boxes:
            frame = mask_regions(frame, boxes)
        dest = os.path.join(frames_dir, FRAME_PATTERN % i)
        if not cv2.imwrite(dest, frame):
            raise IOError(f"failed to write frame {i} of {prefix} to {dest}")
        n += 1

    review = bool(needs_review(ds, boxes))
    frame_time = _as_float(getattr(ds, "FrameTime", None))
    meta = {
        "provenance": "dicom",
        "dicom_metadata": True,
        "site": str(site),
        "pseudo_patient": str(pseudo_patient),
        "pseudo_study": _jsonable(getattr(ds, "StudyInstanceUID", None)),
        "pseudo_series": _jsonable(getattr(ds, "SeriesInstanceUID", None)),
        "pseudo_sop": _jsonable(getattr(ds, "SOPInstanceUID", None)),
        "series_idx": int(series_idx),
        "modality": _jsonable(getattr(ds, "Modality", None)),
        "manufacturer": _jsonable(getattr(ds, "Manufacturer", None)),
        "photometric_interpretation": _jsonable(getattr(ds, "PhotometricInterpretation", None)),
        "bits_stored": _jsonable(getattr(ds, "BitsStored", None)),
        "window_center": _as_float(getattr(ds, "WindowCenter", None)),
        "window_width": _as_float(getattr(ds, "WindowWidth", None)),
        "rows": int(stack.shape[1]),
        "columns": int(stack.shape[2]),
        "n_frames": n,
        "frame_time_ms": frame_time,
        "fps": round(1000.0 / frame_time, 3) if frame_time else None,
        "frame_pattern": FRAME_PATTERN,
        "frame_stem_pattern": prefix + "_%05d",
        "frames_dir": os.path.relpath(frames_dir, str(out_root)),
        "source_sha256": _sha256_bytes(np.ascontiguousarray(stack).tobytes()),
        "mask_boxes": [list(b) for b in boxes],
        "burned_in_tag": _jsonable(getattr(ds, "BurnedInAnnotation", None)),
        "review_required": review,
    }
    write_sidecar(out_root, prefix, meta)
    return {"stem_prefix": prefix, "n_frames": n, "dir": frames_dir, "review_required": review}


def main(argv=None):
    """CLI: de-identify one DICOM, screen its pixels, and write frames + sidecar."""
    import argparse
    import json

    import numpy as np
    import pydicom

    from src.ingest.clearance import require_clearance
    from src.ingest.deid import deid_dataset, load_or_create_salt
    from src.ingest.pixel_deid import detect_text_regions

    ap = argparse.ArgumentParser(description="Extract de-identified PNG frames from one DICOM.")
    ap.add_argument("source", help="path to a DICOM file")
    ap.add_argument("--out-root", required=True, help="output root: <clean_root>/<site>")
    ap.add_argument("--site", default="inu", help="site code used in the stem grammar")
    ap.add_argument("--salt", default="secrets/deid_salt.bin", help="path to the de-id salt file")
    ap.add_argument("--series-idx", type=int, default=1)
    ap.add_argument("--clearance", default=None, help="path to the signed clearance record")
    args = ap.parse_args(argv)
    require_clearance("read", **({"clearance_path": args.clearance} if args.clearance else {}))

    salt = load_or_create_salt(args.salt)
    ds, ids = deid_dataset(pydicom.dcmread(args.source), salt, site=args.site)
    arr = np.asarray(ds.pixel_array)
    stack = arr if arr.ndim == 3 else arr[None, ...]
    probe = sorted({0, stack.shape[0] // 2, stack.shape[0] - 1})   # overlays are static; sample 3
    boxes = sorted({b for i in probe for b in detect_text_regions(to_8bit(stack[i], ds))})
    res = extract_series(ds, args.out_root, site=args.site,
                         pseudo_patient=ids["pseudo_patient"], series_idx=args.series_idx,
                         mask_boxes=boxes)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run the full module test file.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_extract.py -q
```

Expected: PASS (12 passed)

- [ ] **Step 9: Smoke-test the CLI and commit.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && \
python -m src.ingest.extract --help && \
git add src/ingest/extract.py tests/test_ingest_extract.py && \
git commit -m "feat(ingest): VOI-LUT windowing + multi-frame DICOM -> ordered PNG frames

to_8bit applies the DICOM VOI LUT (the omission that washes out angio frames),
inverts MONOCHROME1, min-max scales to uint8, and degrades a constant frame to
flat mid-grey instead of dividing by zero. extract_series writes ordered
f00000.png.. under frames/<stem_prefix>/, applies Task 9 mask boxes to every
frame, and records provenance in a sidecar. The B3 segment-of-interest crop is
deferred to labelling (Task 13) and recorded as such.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: `--help` prints the usage line; commit succeeds.

---

### Task 11: `extract.py` — the exported-video path

Not all of the handover arrives as DICOM. A meaningful slice of any institutional fistulography export is already-flattened video: AVI or MP4 clips a technician saved out of the review workstation, sometimes because the archive would not release the originals, sometimes because that is simply what got burned to the drive. These files have **no DICOM metadata at all** — no patient ID, no UIDs, no `PhotometricInterpretation`, no `WindowCenter`, no `BurnedInAnnotation`. They are pixels and nothing else. Discarding them would throw away real studies; treating them as equivalent to DICOM-derived frames would let metadata-free images silently masquerade as fully-provenanced ones.

So `extract_video` decodes with `cv2.VideoCapture` into the **identical** `frames/<stem_prefix>/f00000.png` layout — same directory grammar, same zero padding, same ordering guarantee — converts each frame to greyscale (the workstation re-encoded a greyscale angiogram into a 3-channel stream; the colour channels carry no information and tripling the store size for them is pure waste), and writes a sidecar tagged `provenance="video"` with `dicom_metadata=false`. That pair of fields is the whole point: downstream code, and any human auditing the store, can tell at a glance which frames have real acquisition metadata behind them and which have only pixels, without having to infer it from what is missing. For the same reason `review_required` is unconditionally `True` on this path — there is no `BurnedInAnnotation` to consult and exported clips are the *most* likely to carry a burned-in header, since burning the overlay in is often exactly why the clip was exported.

It must **fail loudly** when the file cannot be opened. `cv2.VideoCapture` does not raise on a missing file, a truncated file, or a codec the build cannot decode — it returns a capture object whose `isOpened()` is `False`, and a naive read loop then exits immediately having written zero frames. A zero-frame extraction that returns normally is indistinguishable, downstream, from "this study genuinely had no images", and the study quietly vanishes from the corpus. So an unopenable source raises with the path in the message, and a source that opens but yields nothing raises too.

**Files:**
- Modify: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/ingest/extract.py` (append `VIDEO_EXTS`, `extract_video`; replace `main`)
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_extract.py` (append)

**Interfaces:**
- Consumes: `stem_prefix`, `write_sidecar`, `_sha256_file`, `FRAME_PATTERN` (Task 10); `src/ingest/deid.pseudo_id(salt, real_id, *, site, kind="P")`, `src/ingest/deid.load_or_create_salt(path)`; `src/ingest/clearance.require_clearance(mode, clearance_path=...)`
- Produces:
  - `VIDEO_EXTS: tuple[str, ...]`
  - `extract_video(path, out_root, *, site, pseudo_patient, series_idx) -> dict` with keys `stem_prefix`, `n_frames`, `dir`, `review_required`; raises `IOError` when the source cannot be opened or yields no frames
  - `main(argv=None) -> int` (now dispatches DICOM vs video on extension)

- [ ] **Step 1: Append the exported-video tests.** Replace the `src.ingest.extract` import line with the first block, then append the second block to `tests/test_ingest_extract.py`.

```python
import pytest

from src.ingest.extract import (extract_series, extract_video, frame_stem, stem_prefix, to_8bit,
                                write_sidecar)
```

```python
# --- extract_video: already-flattened AVI/MP4 handover -------------------------------------------

def _tiny_video(path, n_frames=7, w=32, h=24, fps=15.0):
    """Write a real, decodable MJPG/AVI clip. Synthetic frames only -- no patient data."""
    import cv2
    fourcc = (getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc)(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert vw.isOpened(), f"cv2.VideoWriter could not open {path} (codec unavailable)"
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, :, :] = 10 + i * 20
        vw.write(frame)
    vw.release()
    return str(path)


def test_extract_video_uses_the_same_frame_layout_as_the_dicom_path(tmp_path):
    src = _tiny_video(tmp_path / "clip.avi", n_frames=7)

    res = extract_video(src, str(tmp_path / "out"), site="inu", pseudo_patient=PID, series_idx=4)

    assert res["stem_prefix"] == "avf_inu_3f9c21b04e_s04"
    assert res["n_frames"] == 7
    expected_dir = os.path.join(str(tmp_path / "out"), "frames", "avf_inu_3f9c21b04e_s04")
    assert res["dir"] == expected_dir
    assert sorted(os.listdir(expected_dir)) == [f"f{i:05d}.png" for i in range(7)]


def test_extract_video_frames_are_single_channel_greyscale(tmp_path):
    import cv2
    src = _tiny_video(tmp_path / "clip.avi", n_frames=4, w=32, h=24)

    res = extract_video(src, str(tmp_path / "out"), site="inu", pseudo_patient=PID, series_idx=1)

    for name in sorted(os.listdir(res["dir"])):
        png = cv2.imread(os.path.join(res["dir"], name), cv2.IMREAD_UNCHANGED)
        assert png.ndim == 2, f"{name}: colour channels carry no angiographic information"
        assert png.dtype == np.uint8 and png.shape == (24, 32)


def test_extract_video_sidecar_marks_video_provenance(tmp_path):
    src = _tiny_video(tmp_path / "clip.avi", n_frames=5, w=32, h=24, fps=15.0)
    out = str(tmp_path / "out")

    res = extract_video(src, out, site="inu", pseudo_patient=PID, series_idx=2)

    meta = json.load(open(os.path.join(out, "sidecar", res["stem_prefix"] + ".json")))
    assert meta["provenance"] == "video", "video frames must be distinguishable from DICOM frames"
    assert meta["dicom_metadata"] is False
    assert meta["n_frames"] == 5 and meta["rows"] == 24 and meta["columns"] == 32
    assert meta["modality"] is None and meta["pseudo_sop"] is None, "there is no header to read"
    assert re.fullmatch(r"[0-9a-f]{64}", meta["source_sha256"])
    assert "source_name" not in meta, "exported clip filenames routinely contain the patient name"
    assert meta["review_required"] is True and meta["review_reason"]
    assert res["review_required"] is True


def test_extract_video_raises_when_the_source_cannot_be_opened(tmp_path):
    # A silent zero-frame extraction reads downstream as "this study had no images".
    missing = str(tmp_path / "does_not_exist.avi")
    with pytest.raises(IOError, match="does_not_exist.avi"):
        extract_video(missing, str(tmp_path / "out"), site="inu", pseudo_patient=PID, series_idx=1)

    garbage = tmp_path / "garbage.avi"
    garbage.write_bytes(b"this is not a video container" * 32)
    with pytest.raises(IOError, match="garbage.avi"):
        extract_video(str(garbage), str(tmp_path / "out"), site="inu", pseudo_patient=PID,
                      series_idx=1)

    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "frames")), \
        "a failed open must not leave an empty frames/ directory behind"
```

- [ ] **Step 2: Run and confirm the expected failure.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_extract.py -q
```

Expected: FAIL — `ImportError while importing test module '.../tests/test_ingest_extract.py' ... ImportError: cannot import name 'extract_video' from 'src.ingest.extract'` (1 error).

- [ ] **Step 3: Implement the video path.** Append `VIDEO_EXTS` and `extract_video` to `src/ingest/extract.py`, and replace the existing `main` with the dispatching version below.

```python
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mpg", ".mpeg", ".mkv", ".wmv", ".m4v")


def extract_video(path, out_root, *, site, pseudo_patient, series_idx):
    """Decode an exported AVI/MP4 clip into the SAME frame layout as ``extract_series``.

    Part of the handover is already-flattened video with no DICOM metadata at all. Those frames go
    to the same ``frames/<stem_prefix>/f00000.png`` tree, but the sidecar is tagged
    ``provenance="video"`` / ``dicom_metadata=false`` so downstream code can tell them apart.

    ``review_required`` is unconditionally True: there is no ``BurnedInAnnotation`` to consult, and
    an exported clip is the MOST likely place to find a burned-in header.

    Raises ``IOError`` when the source cannot be opened or decodes to zero frames --
    ``cv2.VideoCapture`` reports a missing file/unknown codec only via ``isOpened()``, and a silent
    empty extraction would read downstream as "this study had no images".
    """
    import cv2

    src = str(path)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"cannot open video for extraction: {src}")

    prefix = stem_prefix(site, pseudo_patient, series_idx)
    frames_dir = os.path.join(str(out_root), "frames", prefix)
    os.makedirs(frames_dir, exist_ok=True)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    n, rows, cols = 0, None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            rows, cols = int(grey.shape[0]), int(grey.shape[1])
            dest = os.path.join(frames_dir, FRAME_PATTERN % n)
            if not cv2.imwrite(dest, grey):
                raise IOError(f"failed to write frame {n} of {prefix} to {dest}")
            n += 1
    finally:
        cap.release()

    if n == 0:
        raise IOError(f"video opened but decoded zero frames: {src}")

    meta = {
        "provenance": "video",
        "dicom_metadata": False,
        "site": str(site),
        "pseudo_patient": str(pseudo_patient),
        "pseudo_study": None,
        "pseudo_series": None,
        "pseudo_sop": None,
        "series_idx": int(series_idx),
        "modality": None,
        "manufacturer": None,
        "photometric_interpretation": None,
        "rows": rows,
        "columns": cols,
        "n_frames": n,
        "frame_time_ms": round(1000.0 / fps, 3) if fps > 0 else None,
        "fps": fps if fps > 0 else None,
        "frame_pattern": FRAME_PATTERN,
        "frame_stem_pattern": prefix + "_%05d",
        "frames_dir": os.path.relpath(frames_dir, str(out_root)),
        "source_sha256": _sha256_file(src),
        "container": os.path.splitext(src)[1].lower().lstrip("."),
        "mask_boxes": [],
        "burned_in_tag": None,
        "review_required": True,
        "review_reason": ("exported video: no DICOM header to check for burned-in annotation; "
                          "overlay text is common on workstation exports"),
    }
    write_sidecar(out_root, prefix, meta)
    return {"stem_prefix": prefix, "n_frames": n, "dir": frames_dir, "review_required": True}


def main(argv=None):
    """CLI: extract one study file -- DICOM (de-identified + pixel-screened) or exported video."""
    import argparse
    import json

    from src.ingest.clearance import require_clearance
    from src.ingest.deid import load_or_create_salt

    ap = argparse.ArgumentParser(
        description="Extract de-identified PNG frames from one DICOM or exported video.")
    ap.add_argument("source", help="path to a DICOM file or an exported .avi/.mp4 clip")
    ap.add_argument("--out-root", required=True, help="output root: <clean_root>/<site>")
    ap.add_argument("--site", default="inu", help="site code used in the stem grammar")
    ap.add_argument("--salt", default="secrets/deid_salt.bin", help="path to the de-id salt file")
    ap.add_argument("--series-idx", type=int, default=1)
    ap.add_argument("--clearance", default=None, help="path to the signed clearance record")
    args = ap.parse_args(argv)
    require_clearance("read", **({"clearance_path": args.clearance} if args.clearance else {}))

    salt = load_or_create_salt(args.salt)
    if os.path.splitext(args.source)[1].lower() in VIDEO_EXTS:
        from src.ingest.deid import pseudo_id
        # No header to hash: the clip's CONTENT hash is the only stable, PHI-free identifier.
        pid = pseudo_id(salt, _sha256_file(args.source), site=args.site)
        res = extract_video(args.source, args.out_root, site=args.site, pseudo_patient=pid,
                            series_idx=args.series_idx)
    else:
        import numpy as np
        import pydicom

        from src.ingest.deid import deid_dataset
        from src.ingest.pixel_deid import detect_text_regions

        ds, ids = deid_dataset(pydicom.dcmread(args.source), salt, site=args.site)
        arr = np.asarray(ds.pixel_array)
        stack = arr if arr.ndim == 3 else arr[None, ...]
        probe = sorted({0, stack.shape[0] // 2, stack.shape[0] - 1})
        boxes = sorted({b for i in probe for b in detect_text_regions(to_8bit(stack[i], ds))})
        res = extract_series(ds, args.out_root, site=args.site,
                             pseudo_patient=ids["pseudo_patient"], series_idx=args.series_idx,
                             mask_boxes=boxes)
    print(json.dumps(res, indent=2))
    return 0
```

- [ ] **Step 4: Run the full module test file.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_extract.py -q
```

Expected: PASS (16 passed)

- [ ] **Step 5: Confirm the ingest suite is still green end to end.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && \
python -m pytest tests/test_ingest_extract.py tests/test_ingest_pixel_deid.py -q && \
python -m src.ingest.extract --help
```

Expected: PASS (27 passed) and the `--help` usage line showing the `source` positional plus `--out-root`.

- [ ] **Step 6: Commit.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && \
git add src/ingest/extract.py tests/test_ingest_extract.py && \
git commit -m "feat(ingest): extract frames from exported AVI/MP4 handover clips

Part of the handover is already-flattened video with no DICOM metadata.
extract_video decodes to the identical frames/<stem_prefix>/f00000.png layout,
converts to greyscale, and writes a sidecar tagged provenance=video /
dicom_metadata=false so these frames stay distinguishable from DICOM-derived
ones. Unopenable or zero-frame sources raise with the path instead of silently
producing an empty extraction. review_required is unconditionally true.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: commit succeeds.

---

### Task 12: The patient-grouping leakage guard

This is the highest-risk integration point in the entire pipeline, and it is worth being blunt about why. Everything Tasks 1–11 produce is a stream of PNG frames named `avf_inu_3f9c21b04e_s01_00012`. Those frames are then consumed by the existing data-prep machinery, and the moment they enter it they pass through `src/data_prep/io_utils.split_of`, which is what decides whether a frame is a training example or a held-out validation example. `split_of` does not hash the frame name — it hashes `group_key(name)`. `group_key` exists to collapse every frame of one source sequence to a single split group, so that a patient lands *entirely* on one side of the split. **Dialygo B5 requires the split to be by patient, never by image**, and `group_key` is the only place in the codebase where that rule is actually enforced.

`group_key`'s existing regexes cover exactly three naming conventions: Danilov (`<site>_<patient>_<seq>_<frame>` → `<site>_<patient>`), CADICA (`p<patient>_v<video>_<frame>` → `p<patient>`), and CathAction (`<clip>_img-<seg>-<frame>` → `<clip>`). A new AVF stem matches **none** of them — `_PATIENT_RE` needs a leading digit run, `_CADICA_RE` needs a leading `p<digits>`, `_CLIP_RE` needs the literal `_img-`. So `group_key("avf_inu_3f9c21b04e_s01_00012")` falls through to its final `return name` and hands back the whole stem. Every frame becomes its own group. `split_of` then hashes each frame independently, and a patient's cine — a few hundred near-identical images taken milliseconds apart, same anatomy, same catheter, same contrast bolus — is scattered across train *and* val. The model memorises frame 11 and is graded on frame 12.

That is not a hypothetical. It is the exact failure that produced this project's fake F1. From `docs/PROJECT_TRACKER.md`, changelog **2026-07-12(a)**: the `arcade+danilov_yolo11s_768_e150` run scored **F1 0.885 / mAP50 0.87** and was flagged as leakage-inflated because Danilov video frames were split per-frame with every patient in both train and val; the honest, patient-grouped re-split dropped the same pipeline to **F1 0.214**. A four-fold collapse. Shipping AVF frames through an unaware `group_key` would reproduce that failure precisely, on a clinical triage aid, with nothing in the output to indicate anything was wrong — an inflated number looks exactly like a good number.

The fix is three lines of production code and a wall of tests, and that ratio is correct. Add `_AVF_RE = re.compile(r"^(avf_[a-z0-9]+_[0-9a-f]{10})_s\d+_\d+$")` alongside the existing patterns, add the AVF row to the naming table so the next person can see all four conventions in one place, and match it in `group_key`. The regex is deliberately tight — it anchors both ends, pins the pseudo-ID to exactly ten hex characters, and requires both the `_s<NN>` series segment and the trailing frame index — so it collapses real stems and refuses near-misses rather than quietly over-collapsing unrelated names into one giant group. Because this modifies a module imported by every converter, trainer, and split auditor in the repo, the final step runs the **full** suite, not just the new file.

**Files:**
- Modify: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/src/data_prep/io_utils.py`
- Test: `/Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline/tests/test_ingest_group_key.py`

**Interfaces:**
- Consumes: `src/data_prep/io_utils.group_key(name) -> str`, `src/data_prep/io_utils.split_of(name, val_frac=0.15) -> str`
- Produces: `_AVF_RE` (module-private); `group_key` gains the AVF branch — `group_key("avf_inu_3f9c21b04e_s01_00012") == "avf_inu_3f9c21b04e"`. Signatures and every existing return value are unchanged.

- [ ] **Step 1: Write the leakage guard tests.**

```python
# tests/test_ingest_group_key.py
"""AVF patient grouping: the guard that keeps one patient out of both train and val.

Dialygo B5 splits BY PATIENT, never by image. group_key() is the single place that rule is
enforced: split_of() hashes group_key(stem), so if group_key does not collapse an AVF cine's
frames to one key, a patient's near-identical frames scatter across train AND val.

That is the documented failure behind this project's fake F1 -- PROJECT_TRACKER changelog
2026-07-12(a): F1 0.885 on a per-frame split, 0.214 after the honest patient-grouped re-split.
"""
import hashlib

from src.data_prep.io_utils import group_key, split_of

PID = "avf_inu_3f9c21b04e"


def _patients(n=12):
    """Deterministic synthetic pseudo-patients in the locked stem grammar (no real IDs)."""
    return [f"avf_inu_{hashlib.md5(f'avf-test-patient-{i}'.encode()).hexdigest()[:10]}"
            for i in range(n)]


def _frames(pid, n_series=4, n_frames=6):
    return [f"{pid}_s{s:02d}_{f:05d}" for s in range(1, n_series + 1) for f in range(n_frames)]


def test_avf_frame_stem_collapses_to_the_patient_group():
    stem = "avf_inu_3f9c21b04e_s01_00012"
    assert group_key(stem) == PID, \
        f"AVF frame must collapse to its patient, got {group_key(stem)!r}"


def test_all_frames_of_one_patient_across_series_share_one_key():
    keys = {group_key(s) for s in _frames(PID, n_series=4, n_frames=25)}
    assert keys == {PID}, f"one patient must yield exactly one group, got {sorted(keys)}"


def test_two_avf_patients_get_distinct_keys():
    a, b = _patients(2)
    assert group_key(f"{a}_s01_00000") != group_key(f"{b}_s01_00000")
    assert group_key(f"{a}_s01_00000") == a and group_key(f"{b}_s01_00000") == b


def test_every_frame_of_a_patient_lands_in_the_same_split():
    # The B5 guarantee, over a corpus-shaped set of stems.
    stems = {pid: _frames(pid, n_series=4, n_frames=6) for pid in _patients(12)}
    total = sum(len(v) for v in stems.values())
    assert total >= 200, f"exercise at least 200 frames, got {total}"

    for pid, frames in stems.items():
        sides = {split_of(f) for f in frames}
        assert len(sides) == 1, f"patient {pid} leaked across {sorted(sides)}"


def test_split_of_agrees_for_a_bare_group_key_and_its_frames():
    # Converters call split_of(<patient>) directly in places and split_of(<frame stem>) in others;
    # the two must never disagree.
    for pid in _patients(12):
        assert split_of(pid) == split_of(f"{pid}_s02_00031"), f"{pid}: group/frame split disagree"


def test_the_grouped_split_is_not_degenerate():
    # Guards against a vacuous pass: if everything hashed to 'train', the test above proves nothing.
    assert {split_of(pid) for pid in _patients(12)} == {"train", "val"}


def test_near_miss_avf_stems_are_not_collapsed():
    # The regex is deliberately tight: over-collapsing unrelated names into one group would be the
    # mirror-image bug (a giant fake 'patient' swallowing the whole corpus).
    assert group_key("avf_inu_3f9c21b04e_s01") == "avf_inu_3f9c21b04e_s01"      # no frame index
    assert group_key("avf_inu_XYZ_s01_00012") == "avf_inu_XYZ_s01_00012"        # id is not hex10
    assert group_key("avf_inu_3f9c21b04_s01_00012") == "avf_inu_3f9c21b04_s01_00012"   # 9 hex
    assert group_key("notavf_inu_3f9c21b04e_s01_00012") == "notavf_inu_3f9c21b04e_s01_00012"


def test_existing_dataset_groupings_are_unchanged():
    # Regression: this module is imported by every converter, trainer and split auditor.
    assert group_key("14_002_5_0016") == "14_002"                       # Danilov
    assert group_key("14_002_8_0001") == "14_002"                       # Danilov, other sequence
    assert group_key("p12_v3_00045") == "p12"                           # CADICA
    assert group_key("p1_v1_0") == "p1"                                 # CADICA
    assert group_key("JFQ_j3383201_img-00000-0042") == "JFQ_j3383201"   # CathAction
    assert split_of("p12") == split_of("p12_v3_00045")                  # CADICA converter path


def test_unmatched_stems_still_return_themselves():
    assert group_key("800") == "800"              # ARCADE bare stem
    assert group_key("train_5") == "train_5"      # ARCADE-disambiguated stem
    assert group_key("") == ""
```

- [ ] **Step 2: Run it and confirm the expected failure.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_group_key.py -q
```

Expected: FAIL — 4 failed, 5 passed; first failure `AssertionError: AVF frame must collapse to its patient, got 'avf_inu_3f9c21b04e_s01_00012'` (the other three are `test_all_frames_of_one_patient_across_series_share_one_key`, `test_every_frame_of_a_patient_lands_in_the_same_split`, and `test_split_of_agrees_for_a_bare_group_key_and_its_frames`).

- [ ] **Step 3: Teach `group_key` the AVF grammar.** In `src/data_prep/io_utils.py`, replace the naming-table comment and the three regex definitions (lines 12–20) with this block:

```python
# Video-derived frames are near-identical between consecutive frames, so a per-frame split leaks
# the same sequence into train AND val. Collapse every frame of one source sequence to a single
# group so it lands entirely on one side (honest holdout):
#   Danilov    <site>_<patient>_<seq>_<frame>  (e.g. 14_002_5_0016)          -> <site>_<patient>
#   CADICA     p<patient>_v<video>_<frame>     (e.g. p12_v3_00045)           -> p<patient>
#   CathAction <clip>_img-<seg>-<frame>        (e.g. JFQ_j3383201_img-00000-0042) -> <clip>
#   AVF        avf_<site>_<pid10>_s<NN>_<FFFFF> (e.g. avf_inu_3f9c21b04e_s01_00012)
#                                                                            -> avf_<site>_<pid10>
_PATIENT_RE = re.compile(r"^(\d+_\d+)_\d+_\d+$")
_CADICA_RE = re.compile(r"^(p\d+)_v\d+_\d+")   # CADICA pXX_vYY_NNNNN -> patient pXX
_CLIP_RE = re.compile(r"^(.+?)_img-\d+-\d+$")
# Dialygo AVF frames (src/ingest/extract.stem_prefix + frame_stem). Anchored at both ends with the
# pseudo-id pinned to exactly 10 hex chars: tight enough that a near-miss stem falls through to
# itself rather than over-collapsing unrelated names into one giant fake 'patient'.
_AVF_RE = re.compile(r"^(avf_[a-z0-9]+_[0-9a-f]{10})_s\d+_\d+$")
```

Then replace `group_key` with:

```python
def group_key(name):
    """Split-group key: collapse a source sequence's frames to one key; else the name itself.

    Dialygo B5 splits BY PATIENT, never by image -- split_of() hashes THIS value, so any stem that
    falls through to ``return name`` gets one group per frame and scatters a patient's
    near-identical frames across train and val (the F1 0.885 -> 0.214 failure, PROJECT_TRACKER
    2026-07-12(a)).

        Danilov    <site>_<patient>_<seq>_<frame>   14_002_5_0016                 -> 14_002
        CADICA     p<patient>_v<video>_<frame>      p12_v3_00045                  -> p12
        CathAction <clip>_img-<seg>-<frame>         JFQ_j3383201_img-00000-0042   -> JFQ_j3383201
        AVF        avf_<site>_<pid10>_s<NN>_<FFFFF> avf_inu_3f9c21b04e_s01_00012  -> avf_inu_3f9c21b04e
    """
    m = _AVF_RE.match(name)
    if m:
        return m.group(1)
    m = _PATIENT_RE.match(name)
    if m:
        return m.group(1)
    m = _CADICA_RE.match(name)
    if m:
        return m.group(1)
    m = _CLIP_RE.match(name)
    if m:
        return m.group(1)
    return name
```

- [ ] **Step 4: Run the new test file.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/test_ingest_group_key.py -q
```

Expected: PASS (9 passed)

- [ ] **Step 5: Run the pre-existing split-grouping guards untouched.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && \
python -m pytest tests/test_split_grouping.py tests/test_val_by_source.py tests/test_io_collision.py -q
```

Expected: PASS — every Danilov / CADICA / CathAction / ARCADE grouping and every `audit_split_leakage` assertion still holds; 0 failed.

- [ ] **Step 6: Run the FULL suite — `io_utils` is imported by every converter, trainer and auditor.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && python -m pytest tests/ -q
```

Expected: PASS — 0 failed. The whole suite: the 374-test baseline plus every test added in Tasks 1–12 (11 from Task 9, 16 from Tasks 10–11, 9 from Task 12).

- [ ] **Step 7: Commit.**

```bash
cd /Users/jugalmodi/Projects/Workspace/INU/Med/interventional-imaging-pipeline && \
git add src/data_prep/io_utils.py tests/test_ingest_group_key.py && \
git commit -m "fix(data_prep): group AVF frame stems by patient in group_key

AVF stems (avf_inu_3f9c21b04e_s01_00012) matched none of the Danilov/CADICA/
CathAction regexes, so group_key returned the stem itself -- one group per
frame -- and split_of scattered a patient's near-identical cine frames across
train and val. That is the exact failure behind F1 0.885 -> 0.214
(PROJECT_TRACKER 2026-07-12(a)).

Adds _AVF_RE (anchored, pid pinned to 10 hex), the AVF row in the naming table,
and the matching branch in group_key. Existing groupings unchanged; full suite
green.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: commit succeeds.
### Task 13: `labels.py` — clinician label adapters and the join

The clinical partner's labels do not arrive in one shape. Batch one is a per-study spreadsheet exported
from Excel (`StudyInstanceUID, Segment, Label, Impression`). Batch two may be an annotation-tool export
(COCO JSON from a box-drawing tool). Batch three may be a directory of PNG masks named after our own
frame stems. A fourth form — free-text radiology reports — arrives too, and this module refuses to parse
it: **narrative prose is the densest PHI carrier in any label export** (age, sex, admission history,
referring clinician, sometimes the patient's name mid-sentence). Under Dialygo B5 nothing in this repo
may become a route by which report text lands in a training artifact, so any column whose name looks
narrative is dropped before the row is constructed and its name is recorded in a `quarantined` audit
field. The four locked keys (`key`, `segment`, `label`, `source`) are always present; `quarantined` is
the fifth, always present, empty when the export was clean.

**The join from a clinician spreadsheet to a DICOM series is where ingest bugs actually live.** Not in
the DICOM reader, not in the PNG writer — in the moment somebody types a study UID into a spreadsheet
cell and Excel reformats it, or labels a study that was never exported, or exports a study nobody
labelled. A join that silently drops those rows produces a training set that is quietly smaller and
quietly biased than the one the clinical lead thinks they labelled. So `join_labels` returns three
lists, not one: the matches, **and** the label rows that hit nothing, **and** the index rows nothing
covered. The caller is expected to treat a non-empty `unmatched_labels` as a **blocking condition, not
a warning** — a clinician spent time on that row and it went nowhere; that is a data-entry or export
bug to be resolved with the clinical lead, not a number to log and move past. `main()` therefore exits
non-zero when `unmatched_labels` is non-empty. `unmatched_index` is the softer signal (unlabelled
studies are normal mid-annotation), reported but not fatal.

Per Dialygo B7 the engineer does not define what counts as abnormal. `normalize_label` strips
surrounding whitespace and lowercases — nothing else. It does not map `"50-70%"` onto `"significant"`,
does not collapse `"moderate"` and `"severe"` into a binary, does not apply a threshold. Whatever
vocabulary Dr. Reddy's protocol uses travels through this module verbatim and is interpreted downstream
against the ground-truth protocol doc (T1.2), not here. Fail-safe defaults hold throughout: a missing
or unreadable label file returns `[]` rather than raising, which makes every index row surface in
`unmatched_index` and every downstream consumer see an empty label set — visibly wrong, never a
confident-looking partial success. Rows with a blank key or a blank label are never matched; they land
in `unmatched_labels` where they block.

**Files:**
- Create: `src/ingest/labels.py`
- Test: `tests/test_ingest_labels.py`

**Interfaces:**
- Consumes: `src.ingest.manifest.append_jsonl(path, obj)`, `src.ingest.manifest.read_jsonl(path)`,
  `src.ingest.manifest.provenance(tool, **extra)`, `src.ingest.clearance.require_clearance(mode)`;
  index rows from `src.ingest.index_dicom` carrying `path, PatientID, StudyInstanceUID,
  SeriesInstanceUID, SOPInstanceUID, Modality, NumberOfFrames, Manufacturer, StudyDate`;
  frame stems from `src.ingest.extract` in the locked grammar `avf_<site>_<pid_hex10>_s<NN>_<FFFFF>`.
- Produces:
  - `load_csv_labels(path) -> list[dict]` — `{"key","segment","label","source","quarantined"}`
  - `load_coco_labels(path) -> list[dict]` — `{"key","frame","bbox","label","source"}`
  - `load_mask_dir_labels(dirpath) -> list[dict]` — `{"key","frame","mask_path","source"}`
  - `normalize_label(value) -> str`
  - `is_narrative_column(name) -> bool`
  - `split_stem(stem) -> tuple[str, str]` — `("avf_inu_3f9c21b04e_s01", "00012")`
  - `join_labels(index_rows, label_rows, *, key) -> tuple[list, list, list]`
  - `write_labels_jsonl(path, matched) -> str`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing loader tests.**

```python
# tests/test_ingest_labels.py
"""Label adapters + the index<->label join (Dialygo B7: labels pass through verbatim)."""
import json
from pathlib import Path

import pytest

from src.ingest.labels import (
    is_narrative_column,
    load_coco_labels,
    load_csv_labels,
    load_mask_dir_labels,
    normalize_label,
    split_stem,
)

CSV_LINES = [
    "StudyInstanceUID,Segment,Label,Impression",
    '1.2.840.1,Juxta-Anastomotic,  Significant Stenosis (>50%) ,'
    '"68F on HD via left radiocephalic AVF; poor thrill, referred by Dr K"',
    '1.2.840.2,juxta_anastomotic,Normal,"no significant lesion"',
]


def _write_csv(tmp_path: Path) -> Path:
    p = tmp_path / "batch1_labels.csv"
    p.write_text("\n".join(CSV_LINES) + "\n", encoding="utf-8")
    return p


def test_load_csv_labels_reads_header_and_normalizes(tmp_path):
    rows = load_csv_labels(_write_csv(tmp_path))
    assert len(rows) == 2
    assert rows[0]["key"] == "1.2.840.1"
    assert rows[0]["segment"] == "juxta-anastomotic"
    assert rows[0]["label"] == "significant stenosis (>50%)"
    assert rows[0]["source"].endswith("batch1_labels.csv")
    assert rows[1]["key"] == "1.2.840.2"
    assert rows[1]["label"] == "normal"


def test_load_csv_labels_quarantines_narrative_columns(tmp_path):
    rows = load_csv_labels(_write_csv(tmp_path))
    assert rows[0]["quarantined"] == ["Impression"]
    assert "impression" not in rows[0]
    # the densest PHI carrier must not survive anywhere in the emitted rows
    blob = json.dumps(rows)
    assert "68F" not in blob
    assert "Dr K" not in blob


def test_is_narrative_column_flags_report_prose():
    for name in ("Impression", "report", "Findings_Text", "clinical notes",
                 "History", "Indication", "Conclusion", "Remarks"):
        assert is_narrative_column(name) is True
    for name in ("StudyInstanceUID", "Segment", "Label", "PatientID", "key"):
        assert is_narrative_column(name) is False


def test_normalize_label_is_verbatim_passthrough():
    # B7: strip + lowercase ONLY. No threshold, no vocabulary mapping.
    assert normalize_label("  Moderate ") == "moderate"
    assert normalize_label("50-70%") == "50-70%"
    assert normalize_label("Significant Stenosis (>50%)") == "significant stenosis (>50%)"
    assert normalize_label(None) == ""
    assert normalize_label("") == ""


def test_split_stem_splits_frame_suffix():
    assert split_stem("avf_inu_3f9c21b04e_s01_00012") == ("avf_inu_3f9c21b04e_s01", "00012")
    assert split_stem("avf_inu_3f9c21b04e_s01") == ("avf_inu_3f9c21b04e_s01", "")


def test_load_coco_labels_reads_bbox_and_category(tmp_path):
    doc = {
        "images": [{"id": 1, "file_name": "avf_inu_3f9c21b04e_s01_00012.png"}],
        "categories": [{"id": 7, "name": "Significant Stenosis"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7,
                         "bbox": [10, 20, 30, 40]}],
    }
    p = tmp_path / "export.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    rows = load_coco_labels(p)
    assert len(rows) == 1
    assert rows[0]["key"] == "avf_inu_3f9c21b04e_s01"
    assert rows[0]["frame"] == "00012"
    assert rows[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert rows[0]["label"] == "significant stenosis"
    assert rows[0]["source"].endswith("export.json")


def test_load_mask_dir_labels_lists_masks(tmp_path):
    d = tmp_path / "masks"
    d.mkdir()
    (d / "avf_inu_3f9c21b04e_s01_00012.png").write_bytes(b"")
    (d / "avf_inu_3f9c21b04e_s01_00013.png").write_bytes(b"")
    rows = load_mask_dir_labels(d)
    assert [r["frame"] for r in rows] == ["00012", "00013"]
    assert {r["key"] for r in rows} == {"avf_inu_3f9c21b04e_s01"}
    assert rows[0]["mask_path"].endswith("avf_inu_3f9c21b04e_s01_00012.png")


def test_missing_inputs_degrade_to_empty_list(tmp_path):
    # fail-safe: an absent/garbled export yields nothing, so every index row
    # surfaces as unmatched instead of a quiet partial success.
    assert load_csv_labels(tmp_path / "nope.csv") == []
    assert load_coco_labels(tmp_path / "nope.json") == []
    assert load_mask_dir_labels(tmp_path / "nope") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_coco_labels(bad) == []
```

- [ ] **Step 2: Run it and watch it fail.**

```bash
python -m pytest tests/test_ingest_labels.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest.labels'` (collection error, 0 tests run).

- [ ] **Step 3: Implement the loaders.**

```python
# src/ingest/labels.py
"""Clinician label adapters + the index<->label join.

Dialygo B7: label semantics belong to the clinical lead. Nothing in this module
maps a value onto a clinical threshold; labels travel through verbatim.
Dialygo B5: free-text reports are quarantined, never parsed — narrative prose is
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
```

- [ ] **Step 4: Run the loader tests.**

```bash
python -m pytest tests/test_ingest_labels.py -q
```

Expected: PASS (8 passed)

- [ ] **Step 5: Write the failing join / write / CLI tests.**

Append to `tests/test_ingest_labels.py`:

```python
from src.ingest import labels as labels_mod
from src.ingest.labels import join_labels, main, write_labels_jsonl
from src.ingest.manifest import append_jsonl, read_jsonl


def _index_row(uid, series="1.9.1"):
    return {
        "path": f"/vol/drive/{uid}.dcm", "PatientID": "P1",
        "StudyInstanceUID": uid, "SeriesInstanceUID": series,
        "SOPInstanceUID": series + ".1", "Modality": "XA",
        "NumberOfFrames": 30, "Manufacturer": "Siemens", "StudyDate": "20260714",
    }


def test_join_labels_matches_on_key():
    index_rows = [_index_row("1.2.840.1"), _index_row("1.2.840.2", "1.9.2")]
    label_rows = [
        {"key": "1.2.840.1", "segment": "juxta_anastomotic", "label": "significant",
         "source": "b1.csv", "quarantined": []},
        {"key": "1.2.840.2", "segment": "juxta_anastomotic", "label": "normal",
         "source": "b1.csv", "quarantined": []},
    ]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert len(matched) == 2
    assert unmatched_labels == []
    assert unmatched_index == []
    assert matched[0]["key"] == "1.2.840.1"
    assert matched[0]["index_row"]["SeriesInstanceUID"] == "1.9.1"
    assert matched[0]["label_row"]["label"] == "significant"


def test_join_labels_reports_label_row_that_matches_nothing():
    index_rows = [_index_row("1.2.840.1")]
    label_rows = [{"key": "1.2.840.999", "segment": "", "label": "significant",
                   "source": "b1.csv", "quarantined": []}]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert matched == []
    assert len(unmatched_labels) == 1
    assert unmatched_labels[0]["key"] == "1.2.840.999"


def test_join_labels_reports_index_row_no_label_covers():
    index_rows = [_index_row("1.2.840.1"), _index_row("1.2.840.2", "1.9.2")]
    label_rows = [{"key": "1.2.840.1", "segment": "", "label": "normal",
                   "source": "b1.csv", "quarantined": []}]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert len(matched) == 1
    assert unmatched_labels == []
    assert [r["StudyInstanceUID"] for r in unmatched_index] == ["1.2.840.2"]


def test_join_labels_returns_both_unmatched_lists_populated():
    # The failure this module exists to catch: the spreadsheet and the drive
    # disagree in BOTH directions and the join still reports, never drops.
    index_rows = [_index_row("1.2.840.1"), _index_row("1.2.840.2", "1.9.2")]
    label_rows = [
        {"key": "1.2.840.1", "segment": "", "label": "normal",
         "source": "b1.csv", "quarantined": []},
        {"key": "1.2.840.777", "segment": "", "label": "significant",
         "source": "b1.csv", "quarantined": []},
    ]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert len(matched) == 1
    assert len(unmatched_labels) == 1 and len(unmatched_index) == 1
    assert unmatched_labels[0]["key"] == "1.2.840.777"
    assert unmatched_index[0]["StudyInstanceUID"] == "1.2.840.2"


def test_join_labels_rejects_blank_key_or_blank_label():
    index_rows = [_index_row("1.2.840.1")]
    label_rows = [
        {"key": "", "segment": "", "label": "normal", "source": "b1.csv",
         "quarantined": []},
        {"key": "1.2.840.1", "segment": "", "label": "  ", "source": "b1.csv",
         "quarantined": []},
    ]
    matched, unmatched_labels, unmatched_index = join_labels(
        index_rows, label_rows, key="StudyInstanceUID")
    assert matched == []
    assert len(unmatched_labels) == 2
    assert len(unmatched_index) == 1


def test_write_labels_jsonl_roundtrip(tmp_path):
    matched = [{"key": "1.2.840.1", "index_row": _index_row("1.2.840.1"),
                "label_row": {"key": "1.2.840.1", "label": "normal"}}]
    out = write_labels_jsonl(tmp_path / "sub" / "labels.jsonl", matched)
    assert Path(out).is_file()
    back = read_jsonl(out)
    assert len(back) == 1
    assert back[0]["key"] == "1.2.840.1"
    assert back[0]["label_row"]["label"] == "normal"
    assert "provenance" in back[0]
    # empty match set still produces the artifact, so doctor can see it
    out2 = write_labels_jsonl(tmp_path / "empty.jsonl", [])
    assert Path(out2).is_file()
    assert read_jsonl(out2) == []


def test_main_exits_nonzero_when_labels_unmatched(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(labels_mod, "require_clearance", lambda *a, **k: None)
    index_path = tmp_path / "dicom_index.jsonl"
    append_jsonl(index_path, _index_row("1.2.840.1"))
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "StudyInstanceUID,Segment,Label\n1.2.840.999,juxta,significant\n",
        encoding="utf-8")
    rc = main(["--index", str(index_path), "--labels", str(csv_path),
               "--kind", "csv", "--key", "StudyInstanceUID",
               "--out", str(tmp_path / "labels.jsonl"), "--mode", "synthetic"])
    assert rc != 0
    assert "BLOCKING" in capsys.readouterr().out
```

- [ ] **Step 6: Run and watch the new tests fail.**

```bash
python -m pytest tests/test_ingest_labels.py -q
```

Expected: FAIL — `ImportError: cannot import name 'join_labels' from 'src.ingest.labels'` (collection error, 0 tests run).

- [ ] **Step 7: Implement the join, the writer, and the CLI.**

Append to `src/ingest/labels.py`:

```python
def join_labels(index_rows, label_rows, *, key):
    """Join clinician labels to DICOM index rows on `key`.

    Returns (matched, unmatched_labels, unmatched_index). Nothing is dropped:
    a label row that hits no series and an index row no label covers are both
    returned. Callers MUST treat a non-empty `unmatched_labels` as blocking —
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
              f"series. Resolve with the clinical lead before training — do not "
              f"proceed on a silently smaller label set.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run the full file.**

```bash
python -m pytest tests/test_ingest_labels.py -q
```

Expected: PASS (15 passed)

- [ ] **Step 9: Commit.**

```bash
git add src/ingest/labels.py tests/test_ingest_labels.py
git commit -m "feat(ingest): clinician label adapters + reporting join

CSV/COCO/mask-dir loaders normalize to a common row shape. normalize_label
strips and lowercases only — label semantics stay with the clinical lead (B7).
Narrative columns (report/impression/history/...) are quarantined, never
parsed: free text is the densest PHI carrier in a label export (B5).
join_labels returns unmatched labels AND unmatched index rows so a
spreadsheet-to-series mismatch blocks instead of silently shrinking the set.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: `link.py` — Phase 5, symlink the clean tree into the repo

The repo holds a **symlink and never real files**. Under Dialygo B5 the patient data stays inside the
environment the institutional data-use agreement governs — the external drive and the clean tree beside
it — and the repo only points at it. `data/raw/` is already gitignored, so a symlink there cannot drag
PHI into version control: git records nothing, and even if the ignore rule were removed a symlink
commits as a 40-byte path string, not as pixels. The moment somebody copies frames into `data/raw/`
instead, the repo becomes a second uncontrolled copy of patient data living on a laptop, outside the
agreement's custody chain, one `git add -f` away from a remote. That is the exact mistake this design
prevents, so `link_site` **raises rather than clobbers** when it finds a real directory or a real file
at the destination. Deleting it automatically would be worse than the bug: it would silently destroy
whatever somebody put there while hiding that they had done it. An existing *symlink*, by contrast, is
replaced idempotently — re-pointing at a new clean root is a routine operation, and re-running the
phase must converge rather than accumulate.

`verify_link` never raises. A dangling link is the normal state of this project: the drive is external
and usually unplugged, so `data/raw/avf_fistulography` points at nothing for most of any working day.
That is a fact to report, not an exception to throw — `doctor` (Task 15) calls this on every run and
needs a dict back regardless of what it finds. It returns `resolves=False` with `exists=True` and
`is_symlink=True` for a dangling link, and all-false with an empty target for a path that is not there
at all. `unlink_site` removes only the link; if the path is not a symlink it raises rather than
recursing into a delete, for the same reason `link_site` refuses to overwrite — this module must never
be the thing that deletes patient data.

**Files:**
- Create: `src/ingest/link.py`
- Test: `tests/test_ingest_link.py`

**Interfaces:**
- Consumes: `src.ingest.clearance.require_clearance(mode)`; the frames root produced by
  `src.ingest.extract` at `<clean_root>/<site>/frames`.
- Produces:
  - `link_site(clean_frames_dir, repo_data_raw, name) -> str` — path of the created symlink
  - `unlink_site(path) -> None`
  - `verify_link(path) -> dict` — `{"exists","is_symlink","resolves","target"}`
  - `LinkError(RuntimeError)`
  - `main(argv=None) -> int`
- Canonical result: `<repo>/data/raw/avf_fistulography -> <clean_root>/<site>/frames`

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_ingest_link.py
"""Phase 5: the repo points at the clean tree, it never holds a copy (B5)."""
import os
from pathlib import Path

import pytest

from src.ingest import link as link_mod
from src.ingest.link import LinkError, link_site, main, unlink_site, verify_link


def _frames(tmp_path: Path, name="clean") -> Path:
    d = tmp_path / name / "inu" / "frames"
    (d / "avf_inu_3f9c21b04e_s01").mkdir(parents=True)
    (d / "avf_inu_3f9c21b04e_s01" / "f00000.png").write_bytes(b"")
    return d


def test_link_site_creates_symlink(tmp_path):
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    out = link_site(frames, data_raw, "avf_fistulography")
    p = Path(out)
    assert p.name == "avf_fistulography"
    assert p.is_symlink()
    assert (p / "avf_inu_3f9c21b04e_s01" / "f00000.png").is_file()


def test_link_site_creates_parent_dirs(tmp_path):
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    assert not data_raw.exists()
    link_site(frames, data_raw, "avf_fistulography")
    assert data_raw.is_dir()


def test_link_site_is_idempotent(tmp_path):
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    first = link_site(frames, data_raw, "avf_fistulography")
    second = link_site(frames, data_raw, "avf_fistulography")
    assert first == second
    assert Path(second).is_symlink()
    assert sorted(p.name for p in data_raw.iterdir()) == ["avf_fistulography"]


def test_link_site_repoints_existing_symlink(tmp_path):
    old = _frames(tmp_path, "clean_old")
    new = _frames(tmp_path, "clean_new")
    data_raw = tmp_path / "repo" / "data" / "raw"
    link_site(old, data_raw, "avf_fistulography")
    out = link_site(new, data_raw, "avf_fistulography")
    assert Path(os.readlink(out)) == new.resolve()


def test_link_site_refuses_to_replace_real_directory(tmp_path):
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    real = data_raw / "avf_fistulography"
    real.mkdir(parents=True)
    (real / "someone_copied_this.png").write_bytes(b"x")
    with pytest.raises(LinkError):
        link_site(frames, data_raw, "avf_fistulography")
    # nothing was clobbered
    assert (real / "someone_copied_this.png").is_file()
    assert not real.is_symlink()


def test_link_site_refuses_to_replace_real_file(tmp_path):
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    data_raw.mkdir(parents=True)
    (data_raw / "avf_fistulography").write_bytes(b"x")
    with pytest.raises(LinkError):
        link_site(frames, data_raw, "avf_fistulography")
    assert (data_raw / "avf_fistulography").read_bytes() == b"x"


def test_verify_link_reports_resolves_false_for_dangling(tmp_path):
    # the external drive is unplugged: a normal condition, not an exception
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    out = link_site(frames, data_raw, "avf_fistulography")
    for child in sorted(frames.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    frames.rmdir()
    info = verify_link(out)
    assert info["exists"] is True
    assert info["is_symlink"] is True
    assert info["resolves"] is False
    assert info["target"].endswith("frames")


def test_verify_link_on_missing_path_does_not_raise(tmp_path):
    info = verify_link(tmp_path / "nothing" / "here")
    assert info == {"exists": False, "is_symlink": False, "resolves": False,
                    "target": ""}


def test_unlink_site_removes_link_not_target(tmp_path):
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    out = link_site(frames, data_raw, "avf_fistulography")
    unlink_site(out)
    assert not os.path.lexists(out)
    assert (frames / "avf_inu_3f9c21b04e_s01" / "f00000.png").is_file()
    unlink_site(out)  # idempotent no-op


def test_unlink_site_refuses_real_directory(tmp_path):
    real = tmp_path / "repo" / "data" / "raw" / "avf_fistulography"
    real.mkdir(parents=True)
    (real / "keep.png").write_bytes(b"x")
    with pytest.raises(LinkError):
        unlink_site(real)
    assert (real / "keep.png").is_file()


def test_main_links_then_verifies(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(link_mod, "require_clearance", lambda *a, **k: None)
    frames = _frames(tmp_path)
    data_raw = tmp_path / "repo" / "data" / "raw"
    rc = main(["--clean-frames", str(frames), "--data-raw", str(data_raw),
               "--name", "avf_fistulography", "--mode", "synthetic"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "resolves=True" in out
    assert (data_raw / "avf_fistulography").is_symlink()
```

- [ ] **Step 2: Run it and watch it fail.**

```bash
python -m pytest tests/test_ingest_link.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest.link'` (collection error, 0 tests run).

- [ ] **Step 3: Implement `link.py`.**

```python
# src/ingest/link.py
"""Phase 5 — point the repo at the clean frame store. Never copy into it.

Dialygo B5: patient data stays inside the environment the institutional
data-use agreement governs. The repo holds a symlink under the already-gitignored
data/raw/, so no pixel can reach version control. A real directory at the link
destination means somebody copied data into the repo — that raises, it is never
overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.ingest.clearance import require_clearance

DEFAULT_LINK_NAME = "avf_fistulography"


class LinkError(RuntimeError):
    """Raised when the link destination holds something that is not a symlink."""


def link_site(clean_frames_dir, repo_data_raw, name) -> str:
    """Create/replace `<repo_data_raw>/<name>` -> `<clean_frames_dir>`.

    Idempotent for an existing symlink. Refuses to touch a real file/dir.
    """
    target = Path(clean_frames_dir).expanduser()
    try:
        target = target.resolve()
    except OSError:
        target = target.absolute()
    root = Path(repo_data_raw).expanduser()
    dest = root / str(name)

    root.mkdir(parents=True, exist_ok=True)

    if os.path.lexists(dest):
        if not dest.is_symlink():
            raise LinkError(
                f"refusing to replace non-symlink path {dest}: real data in the "
                f"repo is exactly what this phase prevents (B5). Move or delete "
                f"it by hand, then re-run."
            )
        dest.unlink()

    dest.symlink_to(target, target_is_directory=True)
    return str(dest)


def unlink_site(path) -> None:
    """Remove the symlink. Never removes the target, never removes a real dir."""
    p = Path(path)
    if not os.path.lexists(p):
        return
    if not p.is_symlink():
        raise LinkError(
            f"refusing to remove non-symlink path {p}: this module never deletes "
            f"real data."
        )
    p.unlink()


def verify_link(path) -> dict:
    """Report link health. Never raises — a dangling link is a normal state."""
    p = Path(path)
    exists = bool(os.path.lexists(p))
    is_symlink = bool(p.is_symlink())
    target = ""
    if is_symlink:
        try:
            target = str(os.readlink(p))
        except OSError:
            target = ""
    resolves = False
    if exists:
        try:
            resolves = bool(p.exists())
        except OSError:
            resolves = False
    return {"exists": exists, "is_symlink": is_symlink,
            "resolves": resolves, "target": target}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.link",
        description="Symlink the clean frame store into the repo (Dialygo B5).")
    ap.add_argument("--clean-frames", help="<clean_root>/<site>/frames")
    ap.add_argument("--data-raw", default="data/raw")
    ap.add_argument("--name", default=DEFAULT_LINK_NAME)
    ap.add_argument("--unlink", action="store_true",
                    help="remove the symlink (never the target)")
    ap.add_argument("--verify", action="store_true",
                    help="report link health and exit without changing anything")
    ap.add_argument("--mode", default="synthetic",
                    help="synthetic until the B5 data-use agreement executes")
    args = ap.parse_args(argv)

    dest = Path(args.data_raw) / args.name

    if args.verify:
        info = verify_link(dest)
        print(f"[link] {dest} exists={info['exists']} "
              f"is_symlink={info['is_symlink']} resolves={info['resolves']} "
              f"target={info['target'] or '-'}")
        return 0 if info["resolves"] else 1

    require_clearance(args.mode)

    if args.unlink:
        unlink_site(dest)
        print(f"[link] removed {dest} (target untouched)")
        return 0

    if not args.clean_frames:
        ap.error("--clean-frames is required unless --unlink/--verify is given")

    out = link_site(args.clean_frames, args.data_raw, args.name)
    info = verify_link(out)
    print(f"[link] {out} -> {info['target'] or '-'} resolves={info['resolves']}")
    if not info["resolves"]:
        print("[link] WARNING: link is dangling — the external drive is not "
              "mounted. Re-run `make ingest-doctor` once it is plugged in.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests.**

```bash
python -m pytest tests/test_ingest_link.py -q
```

Expected: PASS (11 passed)

- [ ] **Step 5: Prove the module is import-light and the CLI is wired.**

```bash
python -c "import sys, src.ingest.link; assert 'torch' not in sys.modules and 'cv2' not in sys.modules; print('clean import ok')"
python -m src.ingest.link --help
```

Expected: PASS — prints `clean import ok`, then the argparse usage block listing
`--clean-frames --data-raw --name --unlink --verify --mode`.

- [ ] **Step 6: Commit.**

```bash
git add src/ingest/link.py tests/test_ingest_link.py
git commit -m "feat(ingest): phase 5 symlink clean frame store into data/raw

The repo points at the clean tree and never holds a copy: under B5 the patient
data stays inside the environment the data-use agreement governs, and data/raw/
is already gitignored so a symlink cannot carry PHI into version control.
link_site refuses to overwrite a real directory (that means someone copied data
into the repo — raise, do not clobber), replaces an existing symlink
idempotently, and creates parent dirs. verify_link reports resolves=False for a
dangling link without raising: the drive is external and usually unplugged, and
doctor must be able to report that as a normal condition.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 15: `doctor.py` — the health check

This is what you run when something looks wrong. Four checks, in the order you would ask the questions
yourself. **`check_mounted`** — are the configured drive roots present? The drive is external and
usually unplugged, and "the pipeline produced nothing" almost always means "the drive is not there".
An empty `drive_roots` list is *not* a failure: it is the expected in-repo state while the B5 data-use
agreement is unexecuted, so it reports ok with that reason spelled out. **`check_links`** — do the
`data/raw/` symlinks resolve, or are they dangling? **`check_manifest`** — do the working-directory
artifacts exist and parse? A work dir that has not been created yet is fine (no ingest run has
happened); a work dir that exists but is missing or has a truncated `files.jsonl` /`scan_state.json` /
`dicom_index.jsonl` is not — a half-written manifest is precisely the condition worth surfacing loudly.
**`check_no_phi_in_repo`** is the safety check: no `.dcm`, no `*.crosswalk.csv`, no `*.salt` or
`salt.bin` may exist as a **real file** anywhere under the repo root (symlinks are fine — pointing at
patient data is the design; holding it is the bug), and `data/raw/` must be gitignored, verified by
shelling out to `git check-ignore -q data/raw` rather than by parsing `.gitignore` ourselves. Git is
the authority on what git ignores; a hand-rolled pattern matcher would eventually disagree with it, and
disagreeing on this particular question is how PHI gets committed.

Note on that shell-out, verified against this repo: `git check-ignore -q data/raw` returns **1** when
`data/raw` does not exist on disk, because `.gitignore:5` is the directory pattern `data/raw/` and git
will not match a directory pattern against a path it cannot see is a directory. `git check-ignore -q
data/raw/` returns 0 either way. The check therefore tries both spellings and passes if either matches
— the ignore rule is genuinely in force in both cases.

`run_doctor` aggregates into `{"ok": all-passed, "checks": [...]}` and **never raises**: every check
runs inside a try/except that converts an exception into a failed check with the exception text as its
detail. A health check that crashes tells you nothing — the one situation where you most need output is
the situation most likely to make a check throw. It also does not call `require_clearance`: doctor is
read-only, must run in any legal state, and refusing to run diagnostics because the paperwork is not
signed would be the wrong kind of safe. `main()` prints one line per check and exits non-zero when
`ok` is False, so `make ingest-doctor` is usable in a shell `&&` chain.

**Files:**
- Create: `src/ingest/doctor.py`
- Test: `tests/test_ingest_doctor.py`

**Interfaces:**
- Consumes: `src.ingest.link.verify_link(path)`; the canonical layout
  `<repo>/.ingest/<site>/{files.jsonl,scan_state.json,dicom_index.jsonl}` and
  `<repo>/data/raw/<name> -> <clean_root>/<site>/frames`; `configs/ingest_sites.yaml` (Task 16).
- Produces:
  - `check_mounted(paths) -> dict` — `{"name","ok","detail"}`
  - `check_links(data_raw) -> dict`
  - `check_manifest(work_dir) -> dict`
  - `check_no_phi_in_repo(repo_root) -> dict`
  - `run_doctor(cfg) -> dict` — `{"ok": bool, "checks": [{"name","ok","detail"}, ...]}`
  - `MANIFEST_ARTIFACTS = ("files.jsonl", "scan_state.json", "dicom_index.jsonl")`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing per-check tests.**

```python
# tests/test_ingest_doctor.py
"""Health check: the thing you run when something looks wrong."""
import json
import subprocess
from pathlib import Path

import pytest

from src.ingest import doctor as doctor_mod
from src.ingest.doctor import (
    MANIFEST_ARTIFACTS,
    check_links,
    check_manifest,
    check_mounted,
    check_no_phi_in_repo,
    run_doctor,
)
from src.ingest.link import link_site


def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text(
        "data/raw/\ndata/processed/\n.ingest/\n*.crosswalk.csv\n*.salt\nsalt.bin\n",
        encoding="utf-8")
    (root / "data" / "raw").mkdir(parents=True)
    return root


def _work_dir(tmp_path: Path) -> Path:
    work = tmp_path / ".ingest" / "inu"
    work.mkdir(parents=True)
    (work / "files.jsonl").write_text(
        json.dumps({"path": "/vol/drive/a.dcm", "size": 12}) + "\n", encoding="utf-8")
    (work / "scan_state.json").write_text(
        json.dumps({"site": "inu", "n_files": 1}), encoding="utf-8")
    (work / "dicom_index.jsonl").write_text(
        json.dumps({"path": "/vol/drive/a.dcm", "Modality": "XA"}) + "\n",
        encoding="utf-8")
    return work


def test_check_mounted_ok_when_roots_present(tmp_path):
    (tmp_path / "drive").mkdir()
    res = check_mounted([tmp_path / "drive"])
    assert res["ok"] is True
    assert res["name"] == "mounted"


def test_check_mounted_fails_when_root_missing(tmp_path):
    (tmp_path / "drive").mkdir()
    res = check_mounted([tmp_path / "drive", tmp_path / "gone"])
    assert res["ok"] is False
    assert "gone" in res["detail"]


def test_check_mounted_ok_when_no_roots_configured():
    # the expected in-repo state while B5 is unexecuted
    res = check_mounted([])
    assert res["ok"] is True
    assert "B5" in res["detail"]


def test_check_links_ok_for_resolving_symlink(tmp_path):
    frames = tmp_path / "clean" / "inu" / "frames"
    frames.mkdir(parents=True)
    data_raw = tmp_path / "repo" / "data" / "raw"
    link_site(frames, data_raw, "avf_fistulography")
    res = check_links(data_raw)
    assert res["ok"] is True
    assert res["name"] == "links"


def test_check_links_fails_for_dangling_symlink(tmp_path):
    frames = tmp_path / "clean" / "inu" / "frames"
    frames.mkdir(parents=True)
    data_raw = tmp_path / "repo" / "data" / "raw"
    link_site(frames, data_raw, "avf_fistulography")
    frames.rmdir()
    res = check_links(data_raw)
    assert res["ok"] is False
    assert "avf_fistulography" in res["detail"]


def test_check_links_ok_when_nothing_linked_yet(tmp_path):
    assert check_links(tmp_path / "repo" / "data" / "raw")["ok"] is True


def test_check_manifest_ok_when_artifacts_parse(tmp_path):
    res = check_manifest(_work_dir(tmp_path))
    assert res["ok"] is True
    assert res["name"] == "manifest"


def test_check_manifest_fails_on_malformed_jsonl(tmp_path):
    work = _work_dir(tmp_path)
    (work / "dicom_index.jsonl").write_text('{"path": "a.dcm"\n', encoding="utf-8")
    res = check_manifest(work)
    assert res["ok"] is False
    assert "dicom_index.jsonl" in res["detail"]


def test_check_manifest_fails_when_artifact_missing(tmp_path):
    work = _work_dir(tmp_path)
    (work / "scan_state.json").unlink()
    res = check_manifest(work)
    assert res["ok"] is False
    assert "scan_state.json" in res["detail"]


def test_check_manifest_ok_when_no_run_has_happened(tmp_path):
    res = check_manifest(tmp_path / ".ingest" / "inu")
    assert res["ok"] is True
    assert set(MANIFEST_ARTIFACTS) == {"files.jsonl", "scan_state.json",
                                       "dicom_index.jsonl"}


def test_check_no_phi_in_repo_ok_on_clean_repo(tmp_path):
    res = check_no_phi_in_repo(_git_repo(tmp_path))
    assert res["ok"] is True, res["detail"]
    assert res["name"] == "no_phi_in_repo"


def test_check_no_phi_in_repo_catches_planted_dcm(tmp_path):
    root = _git_repo(tmp_path)
    (root / "notes").mkdir()
    (root / "notes" / "IM000001.dcm").write_bytes(b"DICM")
    res = check_no_phi_in_repo(root)
    assert res["ok"] is False
    assert "IM000001.dcm" in res["detail"]


def test_check_no_phi_in_repo_catches_crosswalk_and_salt(tmp_path):
    root = _git_repo(tmp_path)
    (root / "inu.crosswalk.csv").write_text("pseudo,real\n", encoding="utf-8")
    (root / "salt.bin").write_bytes(b"\x00" * 16)
    (root / "inu.salt").write_bytes(b"\x00" * 16)
    res = check_no_phi_in_repo(root)
    assert res["ok"] is False
    for expected in ("inu.crosswalk.csv", "salt.bin", "inu.salt"):
        assert expected in res["detail"]


def test_check_no_phi_in_repo_allows_symlinked_dcm(tmp_path):
    # pointing at patient data is the design; holding it is the bug
    root = _git_repo(tmp_path)
    real = tmp_path / "outside" / "IM000001.dcm"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"DICM")
    (root / "IM000001.dcm").symlink_to(real)
    res = check_no_phi_in_repo(root)
    assert res["ok"] is True, res["detail"]


def test_check_no_phi_in_repo_fails_when_data_raw_not_gitignored(tmp_path):
    root = _git_repo(tmp_path)
    (root / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    res = check_no_phi_in_repo(root)
    assert res["ok"] is False
    assert "check-ignore" in res["detail"]


def test_run_doctor_ok_on_healthy_tree(tmp_path):
    root = _git_repo(tmp_path)
    frames = tmp_path / "clean" / "inu" / "frames"
    frames.mkdir(parents=True)
    link_site(frames, root / "data" / "raw", "avf_fistulography")
    work = root / ".ingest" / "inu"
    work.mkdir(parents=True)
    (work / "files.jsonl").write_text("", encoding="utf-8")
    (work / "scan_state.json").write_text("{}", encoding="utf-8")
    (work / "dicom_index.jsonl").write_text("", encoding="utf-8")
    report = run_doctor({"site": "inu", "drive_roots": [], "repo_root": str(root),
                         "data_raw": str(root / "data" / "raw"),
                         "work_dir": str(work)})
    assert report["ok"] is True, report["checks"]
    assert [c["name"] for c in report["checks"]] == [
        "mounted", "links", "manifest", "no_phi_in_repo"]


def test_run_doctor_returns_ok_false_instead_of_raising(tmp_path, monkeypatch):
    def boom(_paths):
        raise RuntimeError("drive enumeration exploded")

    monkeypatch.setattr(doctor_mod, "check_mounted", boom)
    report = run_doctor({"site": "inu", "repo_root": str(_git_repo(tmp_path))})
    assert report["ok"] is False
    bad = [c for c in report["checks"] if c["name"] == "mounted"][0]
    assert "drive enumeration exploded" in bad["detail"]


def test_main_exits_nonzero_when_not_ok(tmp_path, capsys):
    root = _git_repo(tmp_path)
    (root / "IM000001.dcm").write_bytes(b"DICM")
    cfg = tmp_path / "sites.yaml"
    cfg.write_text(f"site: inu\ndrive_roots: []\nwork_dir: {root}/.ingest/inu\n"
                   f"data_raw: {root}/data/raw\n", encoding="utf-8")
    rc = doctor_mod.main(["--config", str(cfg), "--repo-root", str(root)])
    out = capsys.readouterr().out
    assert rc != 0
    assert "FAIL" in out
    assert "ok=False" in out
```

- [ ] **Step 2: Run it and watch it fail.**

```bash
python -m pytest tests/test_ingest_doctor.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest.doctor'` (collection error, 0 tests run).

- [ ] **Step 3: Implement `doctor.py`.**

```python
# src/ingest/doctor.py
"""Ingest health check — what you run when something looks wrong.

Never raises: a health check that crashes tells you nothing, and the moment you
most need output is the moment a check is most likely to throw. Read-only, so it
deliberately does NOT call require_clearance — diagnostics must run in any legal
state.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from src.ingest.link import verify_link

MANIFEST_ARTIFACTS = ("files.jsonl", "scan_state.json", "dicom_index.jsonl")

PHI_SUFFIXES = (".dcm", ".crosswalk.csv", ".salt")
PHI_NAMES = ("salt.bin",)
SKIP_DIRS = (".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv")


def _result(name, ok, detail) -> dict:
    return {"name": name, "ok": bool(ok), "detail": str(detail)}


def check_mounted(paths) -> dict:
    """Are the configured drive roots present? The drive is usually unplugged."""
    roots = [str(p) for p in (paths or [])]
    if not roots:
        return _result(
            "mounted", True,
            "no drive_roots configured — expected while the B5 data-use "
            "agreement is unexecuted (synthetic data only)")
    missing = sorted(p for p in roots if not Path(p).is_dir())
    detail = f"{len(roots) - len(missing)}/{len(roots)} drive root(s) present"
    if missing:
        detail += (f"; missing: {', '.join(missing)} "
                   f"— external drive not mounted?")
    return _result("mounted", not missing, detail)


def check_links(data_raw) -> dict:
    """Do the data/raw/ symlinks resolve, or are they dangling?"""
    root = Path(data_raw)
    if not root.is_dir():
        return _result("links", True, f"{root} does not exist yet — nothing linked")
    links = sorted(p for p in root.iterdir() if p.is_symlink())
    if not links:
        return _result("links", True, f"no symlinks under {root}")
    dangling = []
    for p in links:
        info = verify_link(p)
        if not info["resolves"]:
            dangling.append(f"{p.name} -> {info['target'] or '?'}")
    detail = f"{len(links) - len(dangling)}/{len(links)} symlink(s) resolve"
    if dangling:
        detail += (f"; dangling: {', '.join(dangling)} "
                   f"— external drive not mounted?")
    return _result("links", not dangling, detail)


def check_manifest(work_dir) -> dict:
    """Do the working-directory artifacts exist and parse?"""
    work = Path(work_dir)
    if not work.is_dir():
        return _result("manifest", True,
                       f"{work} does not exist yet — no ingest run has happened")
    problems = []
    for name in MANIFEST_ARTIFACTS:
        p = work / name
        if not p.is_file():
            problems.append(f"{name} missing")
            continue
        try:
            text = p.read_text(encoding="utf-8")
            if name.endswith(".jsonl"):
                for i, line in enumerate(text.splitlines(), start=1):
                    if line.strip():
                        json.loads(line)
            elif text.strip():
                json.loads(text)
        except (OSError, UnicodeDecodeError) as exc:
            problems.append(f"{name} unreadable ({type(exc).__name__})")
        except ValueError as exc:
            problems.append(f"{name} does not parse ({exc})")
    detail = (f"{len(MANIFEST_ARTIFACTS) - len(problems)}/"
              f"{len(MANIFEST_ARTIFACTS)} artifact(s) ok in {work}")
    if problems:
        detail += "; " + "; ".join(problems)
    return _result("manifest", not problems, detail)


def _is_phi_file(name: str) -> bool:
    lowered = name.lower()
    return lowered in PHI_NAMES or lowered.endswith(PHI_SUFFIXES)


def _git_check_ignore(root: Path, rel: str) -> bool:
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", rel], cwd=str(root),
                              capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def check_no_phi_in_repo(repo_root) -> dict:
    """The safety check: no real PHI-bearing file may live under the repo.

    Symlinks are fine — pointing at patient data is the design (B5); holding a
    copy is the bug. `data/raw/` must be gitignored, and git is asked directly
    rather than by re-implementing its pattern matching.
    """
    root = Path(repo_root)
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS
                       and not os.path.islink(os.path.join(dirpath, d))]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.is_symlink():
                continue
            if _is_phi_file(fn):
                offenders.append(str(p))
    # `data/raw` alone returns 1 when the dir does not exist on disk, because the
    # rule is the directory pattern `data/raw/`. Try both spellings.
    ignored = (_git_check_ignore(root, "data/raw")
               or _git_check_ignore(root, "data/raw/"))
    parts = []
    if offenders:
        parts.append(f"{len(offenders)} PHI-bearing real file(s): "
                     + ", ".join(sorted(offenders)))
    if not ignored:
        parts.append("`git check-ignore -q data/raw` did not pass — data/raw is "
                     "NOT ignored (or this is not a git work tree)")
    if not parts:
        parts.append("no real .dcm/.crosswalk.csv/.salt/salt.bin under the repo; "
                     "data/raw is gitignored")
    return _result("no_phi_in_repo", not offenders and ignored, "; ".join(parts))


def run_doctor(cfg) -> dict:
    """Run every check. Aggregates; never raises."""
    cfg = dict(cfg or {})
    repo_root = str(cfg.get("repo_root") or ".")
    site = str(cfg.get("site") or "inu")
    data_raw = str(cfg.get("data_raw") or Path(repo_root) / "data" / "raw")
    work_dir = str(cfg.get("work_dir") or Path(repo_root) / ".ingest" / site)
    plan = (
        ("mounted", lambda: check_mounted(cfg.get("drive_roots") or [])),
        ("links", lambda: check_links(data_raw)),
        ("manifest", lambda: check_manifest(work_dir)),
        ("no_phi_in_repo", lambda: check_no_phi_in_repo(repo_root)),
    )
    checks = []
    for name, fn in plan:
        try:
            res = fn() or {}
            checks.append(_result(res.get("name", name), res.get("ok", False),
                                  res.get("detail", "")))
        except Exception as exc:  # a crashing check must still report
            checks.append(_result(name, False,
                                  f"check raised {type(exc).__name__}: {exc}"))
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def main(argv=None) -> int:
    import argparse

    import yaml

    ap = argparse.ArgumentParser(
        prog="python -m src.ingest.doctor",
        description="Ingest health check (read-only; runs in any legal state).")
    ap.add_argument("--config", default="configs/ingest_sites.yaml")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            print(f"[doctor] config {cfg_path} unreadable ({exc}) — "
                  f"falling back to defaults")
            cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("repo_root", args.repo_root)

    report = run_doctor(cfg)
    for c in report["checks"]:
        flag = "ok  " if c["ok"] else "FAIL"
        print(f"[{flag}] {c['name']}: {c['detail']}")
    print(f"[doctor] ok={report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests.**

```bash
python -m pytest tests/test_ingest_doctor.py -q
```

Expected: PASS (18 passed)

- [ ] **Step 5: Run the doctor against this repo for real.**

```bash
python -m src.ingest.doctor --repo-root . ; echo "exit=$?"
```

Expected: PASS — four `[ok  ]` lines and `[doctor] ok=True`.
`mounted` reports "no drive_roots configured"; `links` reports "no symlinks under data/raw" (or that
`data/raw` does not exist yet); `manifest` reports the work dir does not exist yet;
`no_phi_in_repo` reports no real PHI-bearing files and that `data/raw` is gitignored.
If `no_phi_in_repo` fails here, stop — that is a real finding, not a test bug.

- [ ] **Step 6: Commit.**

```bash
git add src/ingest/doctor.py tests/test_ingest_doctor.py
git commit -m "feat(ingest): doctor health check with a PHI-in-repo assertion

check_mounted (external drive usually unplugged; empty drive_roots is the
expected pre-B5 state), check_links (dangling symlinks), check_manifest
(artifacts exist and parse), and check_no_phi_in_repo — no real .dcm,
*.crosswalk.csv, *.salt or salt.bin anywhere under the repo root (symlinks are
fine) plus git check-ignore on data/raw, asking git rather than re-implementing
its pattern matching. run_doctor aggregates and never raises: a health check
that crashes tells you nothing. main() prints a readable report and exits
non-zero when ok is False.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 16: Wiring — configs, Makefile, docs

The pipeline exists; nothing points at it yet. This task gives it a config file, six `make` targets in
the shape the rest of the repo already uses, and a paper trail in the tracker. Two things carry weight
beyond convenience. First, `configs/ingest_sites.yaml` ships with `drive_roots: []` — **empty, on
purpose**. Populating it is the single action that turns this pipeline from a synthetic-data exercise
into real-patient processing, so the empty list is the B5 gate expressed in a file somebody has to
deliberately edit, with the reason written next to it. Second, `MODE ?= synthetic` in the Makefile
means the safe mode is what you get when you type nothing; processing real data requires typing
`MODE=cleared` and having `require_clearance` accept it against `configs/ingest_clearance.yaml`. A
default that is safe when you are not paying attention is the only kind worth having here.

The changelog entry must state plainly that no real patient data was processed and that both gates
remain closed — B5 (institutional data-use agreement) and B9 (IP/engagement agreement). Six months
from now the question "was any of this built against real fistulography?" needs an answer that does not
depend on anyone's memory.

**Files:**
- Create: `configs/ingest_sites.yaml`
- Modify: `Makefile`, `docs/PROJECT_TRACKER.md`,
  `docs/superpowers/plans/2026-08-01-dialygo-realignment.md`
- Test: `tests/test_ingest_doctor.py` (append one config-contract test)

**Interfaces:**
- Consumes: `src.ingest.doctor.main` (reads `configs/ingest_sites.yaml`); the module CLIs from
  Tasks 8–15, all of which take `--mode` and the phase paths below.
- Produces:
  - `configs/ingest_sites.yaml` — `site, drive_roots, clean_root, work_dir, data_raw, link_name`
  - `make ingest-scan | ingest-index | ingest-deid | ingest-extract | ingest-link | ingest-doctor`
  - Makefile variables `SITE ?= inu`, `MODE ?= synthetic`, `SRC ?=`, `WORK ?= .ingest/$(SITE)`,
    `CLEAN_ROOT ?=`, `LINK_NAME ?= avf_fistulography`, `INGEST_CFG ?= configs/ingest_sites.yaml`

- [ ] **Step 1: Write the failing config-contract test.**

Append to `tests/test_ingest_doctor.py`:

```python
def test_ingest_sites_config_is_b5_safe():
    """The shipped site config must not point at any real drive."""
    import yaml

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "ingest_sites.yaml"
    assert cfg_path.is_file(), f"{cfg_path} missing"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["site"] == "inu"
    assert cfg["drive_roots"] == [], (
        "drive_roots must ship EMPTY — populating it starts real-patient "
        "processing and is gated on the B5 data-use agreement")
    assert cfg["work_dir"] == ".ingest/inu"
    assert cfg["data_raw"] == "data/raw"
    assert cfg["link_name"] == "avf_fistulography"
    assert "clean_root" in cfg
```

- [ ] **Step 2: Run it and watch it fail.**

```bash
python -m pytest tests/test_ingest_doctor.py::test_ingest_sites_config_is_b5_safe -q
```

Expected: FAIL — `AssertionError: .../configs/ingest_sites.yaml missing` (1 failed).

- [ ] **Step 3: Create `configs/ingest_sites.yaml`.**

```yaml
# Dialygo ingest — institutional fistulography site config (src/ingest/).
# One site today (Institute of Nephro-Urology). Copy this block per site when B6
# external validation brings a second one online.
site: inu                                                        # short site slug; appears in every stem: avf_<site>_<pid_hex10>_s<NN>_<FFFFF>
drive_roots: []                                                  # HARD GATE (B5): empty until the institutional data-use agreement executes. Populating this list is the action that starts processing real patient data — do not add a path to satisfy a test, a demo, or a deadline. Until then: synthetic DICOM only (see tests/fixtures). B9 (IP/engagement agreement) must also be executed before real-data development begins.
clean_root: ~/dialygo_clean                                      # de-identified tree, OUTSIDE the repo and inside the environment the B5 agreement governs: <clean_root>/<site>/{dicom,frames,sidecar,_keys,_manifest}/
work_dir: .ingest/inu                                            # in-repo working artifacts (gitignored): files.jsonl, scan_state.json, dicom_index.jsonl, phi_audit.md, qa_review.jsonl
data_raw: data/raw                                               # where the symlink lands; already gitignored (.gitignore:5) so a link cannot carry PHI into version control
link_name: avf_fistulography                                     # data/raw/avf_fistulography -> <clean_root>/<site>/frames  (matches configs/avf_fistulography.yaml data.root)
mode: synthetic                                                  # synthetic|cleared — checked by src/ingest/clearance.require_clearance against configs/ingest_clearance.yaml; 'cleared' requires the executed agreement
# Second site (B6 external validation, >=1 non-source site) — uncomment and fill when identified:
# site: ext1
# drive_roots: []
# clean_root: ~/dialygo_clean_ext1
# work_dir: .ingest/ext1
```

- [ ] **Step 4: Run the config test.**

```bash
python -m pytest tests/test_ingest_doctor.py -q
```

Expected: PASS (19 passed)

- [ ] **Step 5: Add the Makefile targets.**

Append to `Makefile` (matching the existing `$(PY) -m ...` style and the `# --- section ---` headers):

```make

# --- Dialygo ingest (institutional fistulography DICOM -> de-identified PNG frames) ---
# HARD GATE (Dialygo B5): real patient data may not be processed until the institutional
# data-use agreement executes, and B9 (IP/engagement agreement) must execute before
# real-data development begins. MODE defaults to `synthetic` so the safe path is the one
# you get by typing nothing; `MODE=cleared` is checked by src/ingest/clearance.py against
# configs/ingest_clearance.yaml and will refuse if the paperwork is not recorded there.
SITE       ?= inu
MODE       ?= synthetic
SRC        ?=                       # drive root to scan; leave empty until B5 clears
WORK       ?= .ingest/$(SITE)
CLEAN_ROOT ?= $(HOME)/dialygo_clean
LINK_NAME  ?= avf_fistulography
INGEST_CFG ?= configs/ingest_sites.yaml

ingest-scan:              # SRC=/Volumes/<drive> — walk the drive, record every candidate file
	$(PY) -m src.ingest.scan --src $(SRC) --work $(WORK) --site $(SITE) --mode $(MODE)
ingest-index:             # read DICOM headers -> dicom_index.jsonl (patient/study/series/SOP)
	$(PY) -m src.ingest.index_dicom --work $(WORK) --site $(SITE) --mode $(MODE)
ingest-deid:              # pseudonymize + residual-PHI audit -> <clean_root>/<site>/dicom + _keys
	$(PY) -m src.ingest.deid --work $(WORK) --clean-root $(CLEAN_ROOT) --site $(SITE) --mode $(MODE)
ingest-extract:           # frames -> <clean_root>/<site>/frames/<stem_prefix>/fNNNNN.png + sidecars
	$(PY) -m src.ingest.extract --work $(WORK) --clean-root $(CLEAN_ROOT) --site $(SITE) --mode $(MODE)
ingest-labels:            # LABELS=<csv|coco.json|maskdir> KIND=csv|coco|mask_dir — join clinician labels (B7)
	$(PY) -m src.ingest.labels --index $(WORK)/dicom_index.jsonl --labels $(LABELS) \
		--kind $(KIND) --key StudyInstanceUID --out $(WORK)/labels.jsonl --mode $(MODE)
ingest-link:              # data/raw/$(LINK_NAME) -> $(CLEAN_ROOT)/$(SITE)/frames  (symlink, never a copy)
	$(PY) -m src.ingest.link --clean-frames $(CLEAN_ROOT)/$(SITE)/frames \
		--data-raw data/raw --name $(LINK_NAME) --mode $(MODE)
ingest-doctor:            # health check: drives mounted, links resolve, manifests parse, no PHI in repo
	$(PY) -m src.ingest.doctor --config $(INGEST_CFG) --repo-root .
```

Verify the targets expand and the two CLIs this plan section owns are wired:

```bash
make -n ingest-doctor ingest-link
python -m src.ingest.doctor --help | head -5
python -m src.ingest.link --help | head -5
```

Expected: PASS — `make -n` prints the two `python -m src.ingest....` command lines with `MODE=synthetic`
substituted, and both `--help` blocks print their usage lines.

- [ ] **Step 6: Prove `make ingest-doctor` reports ok against a synthetic fixture tree.**

```bash
FIX=$(mktemp -d)/clean/inu/frames
mkdir -p "$FIX/avf_inu_3f9c21b04e_s01"
: > "$FIX/avf_inu_3f9c21b04e_s01/f00000.png"
make ingest-link CLEAN_ROOT="$(dirname "$(dirname "$FIX")")"
make ingest-doctor ; echo "doctor exit=$?"
git check-ignore -q data/raw ; echo "check-ignore exit=$?"
git status --porcelain data/raw
```

Expected: PASS — `ingest-link` prints `resolves=True`; `ingest-doctor` prints four `[ok  ]` lines,
`[doctor] ok=True` and `doctor exit=0`; `check-ignore exit=0` (it now returns 0 because
`make ingest-link` created the `data/raw` directory — before that it returns 1 against the
`data/raw/` directory pattern, which is why `check_no_phi_in_repo` tries both spellings);
`git status --porcelain data/raw` prints nothing.
Tear down the fixture link with `python -m src.ingest.link --unlink --data-raw data/raw --name avf_fistulography`
if you do not want it left behind — it points into a temp dir, so it will dangle after a reboot and
`make ingest-doctor` will (correctly) start reporting it.

- [ ] **Step 7: Update the docs.**

Edit `docs/superpowers/plans/2026-08-01-dialygo-realignment.md` — tick T1.7 in the Track 1 queue:

```markdown
- [x] **T1.7** `src/ingest/` — DICOM → de-identified cropped PNG frame pipeline, built and tested
      against **synthetic DICOM only**, so it is ready the moment Track 0 clears. **Done 2026-08-02**:
      all six phases + label adapters shipped with CLIs (`make ingest-scan|index|deid|extract|link|doctor`),
      `configs/ingest_sites.yaml` ships with `drive_roots: []`, no real patient data processed.
```

Edit `docs/PROJECT_TRACKER.md` §2 "Code inventory → Implemented (real code)" — add after the
`src/serve/...` line:

```markdown
- [x] `src/ingest/` (2026-08-02) — Dialygo institutional ingest, **synthetic DICOM only**:
  `clearance.py` (B5/B9 mode gate), `manifest.py` (jsonl/atomic-json/provenance/sha256),
  `scan.py`, `index_dicom.py`, `deid.py` (pseudonymize + residual-PHI audit + crosswalk),
  `extract.py` (frames → `<clean_root>/<site>/frames/<stem_prefix>/fNNNNN.png` + sidecars),
  `labels.py` (CSV/COCO/mask-dir adapters + reporting join, B7 verbatim passthrough),
  `link.py` (`data/raw/avf_fistulography` → clean tree, symlink never a copy),
  `doctor.py` (mounted / links / manifest / **no-PHI-in-repo** health check).
  All modules import torch- and cv2-free, expose `main()`, run as `python -m src.ingest.<module>`.
```

Edit `docs/PROJECT_TRACKER.md` §8 Changelog — insert as the new top entry (above the existing
`- **2026-08-02** — **Documentation consistency pass**` line):

```markdown
- **2026-08-02 (b)** — **`src/ingest/` landed — Dialygo realignment T1.7 done.** Institutional
  fistulography DICOM → de-identified, patient-grouped PNG frame store, built and tested end-to-end
  against **synthetic DICOM only**. Phases: `clearance` (mode gate) → `scan` → `index_dicom` →
  `deid` (pseudonymize + residual-PHI audit + gitignored crosswalk) → `extract` (frames + sidecars,
  stem grammar `avf_<site>_<pid_hex10>_s<NN>_<FFFFF>`) → `labels` (clinician CSV/COCO/mask-dir
  adapters + a join that reports unmatched rows on **both** sides, so a spreadsheet-to-series
  mismatch blocks instead of silently shrinking the label set; narrative columns quarantined, never
  parsed — free text is the densest PHI carrier) → `link` (`data/raw/avf_fistulography` symlinked to
  the clean tree; a real directory there raises rather than being clobbered) → `doctor` (drives
  mounted / links resolve / manifests parse / **no real `.dcm`, `*.crosswalk.csv`, `*.salt` or
  `salt.bin` anywhere under the repo** + `git check-ignore -q data/raw`; aggregates and never raises).
  Wiring: `configs/ingest_sites.yaml` (ships with `drive_roots: []`), six `make ingest-*` targets with
  `MODE ?= synthetic` so the safe mode is the default. Tests **+42 in this batch**
  (`test_ingest_labels.py` 15, `test_ingest_link.py` 11, `test_ingest_doctor.py` 19 incl. the shipped
  config contract) on top of the Tasks 1–12 ingest tests; full `pytest tests/` green, 0 failed.
  **No real patient data was processed at any point.** Both gates remain CLOSED: **B5** (institutional
  data-use agreement) unexecuted → `drive_roots` stays empty and only synthetic DICOM has touched this
  code; **B9** (IP/engagement agreement) unexecuted → real-data development has not begun. Per B7 the
  label path carries clinician vocabulary through verbatim (strip + lowercase only) and defines no
  threshold of its own. Per B6 the site config is shaped to take a second, non-source site block for
  external validation.
```

Also update the `**Last updated:**` line at `docs/PROJECT_TRACKER.md:4` to `2026-08-02` if it is not
already, and leave the §1 status table alone — ingest is infrastructure for Stage 4, not a stage gate.

- [ ] **Step 8: Run the full suite.**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: PASS (0 failed) — the ingest additions from Tasks 13–15 contribute exactly +42
(`test_ingest_labels.py` 15, `test_ingest_link.py` 11, `test_ingest_doctor.py` 19). If
`test_train_seg.py` fails here, re-run it in isolation (`python -m pytest tests/test_train_seg.py -q`):
that is the known pre-existing torch-in-`sys.modules` order-pollution failure recorded in the
2026-07-17 changelog entry, unrelated to this work.

- [ ] **Step 9: Commit.**

```bash
git add configs/ingest_sites.yaml Makefile tests/test_ingest_doctor.py \
        docs/PROJECT_TRACKER.md docs/superpowers/plans/2026-08-01-dialygo-realignment.md
git commit -m "chore(ingest): wire configs, make targets and docs for src/ingest

configs/ingest_sites.yaml ships with drive_roots: [] — populating it is the
action that starts real-patient processing and is gated on the B5 data-use
agreement, so the empty list is the gate expressed in a file somebody has to
deliberately edit. Six make targets (scan/index/deid/extract/link/doctor, plus
labels) in the existing \$(PY) -m style with MODE ?= synthetic, so the safe mode
is what you get by typing nothing. Tracker gets a src/ingest/ code-inventory
entry and a 2026-08-02 changelog entry recording that no real patient data was
processed and that B5/B9 remain closed; realignment T1.7 ticked.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 10: Definition of done — the whole ingest pipeline.**

Walk this list and tick each item from actual output, not from memory:

```bash
# 1. every phase has a working CLI
for m in clearance manifest scan index_dicom deid extract labels link doctor; do
  python -m src.ingest.$m --help >/dev/null 2>&1 && echo "ok   $m" || echo "FAIL $m"
done
# 2. the full suite is green
python -m pytest tests/ -q 2>&1 | tail -3
# 3. doctor reports ok on a synthetic fixture tree
make ingest-doctor ; echo "doctor exit=$?"
# 4. the repo cannot carry PHI into git
git check-ignore -q data/raw ; echo "check-ignore exit=$?"
git status --porcelain
```

- [ ] Every phase has a CLI — all nine lines print `ok` (`clearance` and `manifest` are library
      modules with a `main()` that self-reports; the seven pipeline phases run as
      `python -m src.ingest.<module>`).
- [ ] Full suite green — `pytest tests/` reports `0 failed`.
- [ ] `make ingest-doctor` reports `[doctor] ok=True` and exits 0 against the synthetic fixture tree
      built in Step 6.
- [ ] `git check-ignore -q data/raw` exits 0, and `git status --porcelain` shows no `data/raw` entry
      and no `.ingest/` entry — the frame store and the working artifacts are invisible to git.
- [ ] `configs/ingest_sites.yaml` still has `drive_roots: []`. If it does not, real-data processing
      has begun and both **B5** and **B9** must be executed and recorded in
      `configs/ingest_clearance.yaml` first — that is a legal gate, not an engineering one.
- [ ] No real patient data has touched this machine. The only DICOM this pipeline has processed is the
      synthetic fixture set.
