"""DICOM cine -> ordered de-identified PNG frames + sidecar.

Dialygo B5: synthetic fixture only. The VOI LUT tests are the ones that matter -- skipping the
DICOM window is why angio frames come out washed out.
"""
import json
import os
import re

import numpy as np
import pytest

from src.ingest.extract import (extract_series, extract_video, frame_stem, stem_prefix, to_8bit,
                                write_sidecar)

from tests.fixtures.synthetic_dicom import make_xa_dataset, write_dataset

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
    ds2 = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS)
    ds2.PhotometricInterpretation = "MONOCHROME2"
    ds1 = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS)
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


# --- P0.1: module-level clearance-gate hardening on the CLI --------------------------------------


def test_main_requires_mode(tmp_path):
    """Fix #1: --mode has no default -- omitting it must fail argument parsing."""
    from src.ingest.extract import main as extract_main

    ds_path = write_dataset(make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS), tmp_path / "case.dcm")

    with pytest.raises(SystemExit) as ei:
        extract_main([str(ds_path), "--out-root", str(tmp_path / "out")])
    assert ei.value.code == 2
    assert not (tmp_path / "out").exists()


def test_main_smoke_extracts_a_dicom_series(tmp_path, capsys):
    from src.ingest.extract import main as extract_main

    ds_path = write_dataset(make_xa_dataset(n_frames=3, rows=ROWS, cols=COLS), tmp_path / "case.dcm")
    out_root = tmp_path / "out"

    rc = extract_main([str(ds_path), "--out-root", str(out_root), "--mode", "synthetic",
                       "--salt", str(tmp_path / "salt.bin")])

    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_frames"] == 3


def test_main_clearance_override_for_tests_flag_is_honoured(tmp_path):
    """Fix #2: the renamed flag still works, so tests can point real-mode runs at a fixture."""
    from src.ingest.extract import main as extract_main

    ds_path = write_dataset(make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS), tmp_path / "case.dcm")
    marker = tmp_path / "clearance.yaml"
    marker.write_text("data_agreement_executed: true\nip_agreement_executed: true\n")

    rc = extract_main([str(ds_path), "--out-root", str(tmp_path / "out"), "--mode", "real",
                       "--salt", str(tmp_path / "salt.bin"),
                       "--clearance-override-for-tests", str(marker)])

    assert rc == 0


def test_main_rejects_the_old_bare_clearance_flag(tmp_path):
    """The old --clearance flag name must be REJECTED outright, not silently accepted as an
    argparse abbreviation of --clearance-override-for-tests."""
    from src.ingest.extract import main as extract_main

    ds_path = write_dataset(make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS), tmp_path / "case.dcm")
    marker = tmp_path / "clearance.yaml"
    marker.write_text("data_agreement_executed: true\nip_agreement_executed: true\n")

    with pytest.raises(SystemExit) as ei:
        extract_main([str(ds_path), "--out-root", str(tmp_path / "out"), "--mode", "real",
                     "--salt", str(tmp_path / "salt.bin"), "--clearance", str(marker)])
    assert ei.value.code == 2


def test_main_real_mode_ignores_a_cwd_relative_clearance_marker(tmp_path, monkeypatch):
    """Fix #2: without the override flag, the marker must resolve from the repo root, never from
    a marker that happens to sit at a cwd-relative 'configs/ingest_clearance.yaml'."""
    from src.ingest.clearance import ClearanceError
    from src.ingest.extract import main as extract_main

    fake_cwd = tmp_path / "fake_cwd"
    (fake_cwd / "configs").mkdir(parents=True)
    (fake_cwd / "configs" / "ingest_clearance.yaml").write_text(
        "data_agreement_executed: true\nip_agreement_executed: true\n")
    ds_path = write_dataset(make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS),
                            tmp_path / "case.dcm")
    monkeypatch.chdir(fake_cwd)

    with pytest.raises(ClearanceError):
        extract_main([str(ds_path), "--out-root", str(fake_cwd / "out"), "--mode", "real",
                     "--salt", str(fake_cwd / "salt.bin")])
    assert not (fake_cwd / "out").exists()
