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


def read_crosswalk(path):
    """Read a crosswalk CSV back into {real_id: pseudo_id}. Missing file -> {}.

    The inverse of write_crosswalk, and the reason the driver can delete its plaintext scratch
    log after every run: the 0600 CSV is the single accumulated store, so a later handover batch
    reloads the earlier batches' mapping from the one file that is actually protected instead of
    from a second, unprotected copy kept alive purely for resume.

    Rows that are short or malformed are skipped rather than raising -- losing one row costs one
    re-derivable mapping, whereas raising loses the whole crosswalk.
    """
    path = Path(path)
    if not path.exists():
        return {}
    mapping = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if i == 0 and row[:2] == ["real_id", "pseudo_id"]:
                continue
            if len(row) >= 2 and row[0] and row[1]:
                mapping[row[0]] = row[1]
    return mapping


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
    ap.add_argument(
        "--salt",
        default="_keys/salt.bin",
        help="path to the 0600 salt file on the secure drive (default: _keys/salt.bin)",
    )
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


#: PS3.15 Annex E "Z" action: the element stays, its value is emptied. Emptying rather than
#: deleting keeps the dataset conformant and makes the absence explicit to a downstream reader.
#: Times are emptied outright -- a time of day is no clinical use here and is a re-identification
#: handle when combined with a hospital schedule. StudyDescription and SeriesDescription are
#: emptied because in practice they carry patient names typed in at booking and at the cath-lab
#: console -- both are free text a human types under time pressure, and "AVF RUN 3 REDDY" is a
#: routine value. SeriesDescription used to sit in KEEP_TAGS while phi_audit.md told the human
#: reviewer it was scrubbed; resolved towards the stronger claim, since nothing downstream reads
#: it (Modality/Manufacturer/geometry carry the acquisition metadata the models use).
#: PatientID is NOT here: it is replaced with the pseudonym so the de-identified file stays
#: self-describing.
REMOVE_TAGS = (
    "PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "PatientBirthName", "PatientMotherBirthName",
    "AccessionNumber", "StudyID",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "NameOfPhysiciansReadingStudy", "PhysiciansOfRecord",
    "RequestingPhysician", "OperatorsName",
    "StudyDescription", "SeriesDescription",
    "AdmissionID", "PatientComments", "ImageComments",
    "DeviceSerialNumber", "StationName",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
    # added per audit P0.8
    "AdditionalPatientHistory",
)

#: Clinically or methodologically required -- these must survive the scrub untouched.
#: Manufacturer and ManufacturerModelName are retained deliberately (Dialygo B6): vendor identity
#: is not patient identity, and leave-one-site-out external validation cannot be run without it.
KEEP_TAGS = (
    "Modality", "Manufacturer", "ManufacturerModelName",
    "PatientSex",
    "KVP", "ExposureTime", "DistanceSourceToDetector", "DistanceSourceToPatient",
    "PositionerPrimaryAngle", "ImagerPixelSpacing", "CineRate", "FrameTime",
    "Rows", "Columns", "NumberOfFrames", "BitsAllocated", "BitsStored",
    "PhotometricInterpretation", "WindowCenter", "WindowWidth",
)

#: Shifted by the patient's stable offset, never deleted -- intervals are the clinical signal.
DATE_TAGS = ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate")

# Multi-valued LO: each component must stay within the 64-char VR limit.
DEIDENTIFICATION_METHOD = (
    "Dialygo ingest: PS3.15 Annex E basic profile",
    "private tags and 60xx overlays removed",
    "identifying elements emptied; dates shifted per-patient",
    "UIDs remapped under 2.25.",
    "Manufacturer/ModelName retained for external validation",
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
    ds.DeidentificationMethod = list(DEIDENTIFICATION_METHOD)

    return ds, {
        "real_patient": real_patient,
        "pseudo_patient": pseudo_patient,
        "pseudo_study": pseudo_study,
        "pseudo_series": pseudo_series,
        "pseudo_sop": pseudo_sop,
    }


if __name__ == "__main__":
    raise SystemExit(main())
