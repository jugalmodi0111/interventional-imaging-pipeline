"""cadica_to_yolo.main must gate its own output with audit_split_leakage.

danilov_to_yolo.main already ends with an audit_split_leakage() call, so `make prep-stenosis`
is gated for the ARCADE+Danilov half of the corpus. cadica_to_yolo.main had no such call: it
wrote CADICA frames into the SAME data/processed/stenosis tree after the Danilov audit had
already run, so nothing ever audited the combined corpus. That is how the 2026-07-16 run
(experiments/stenosis_arcade+cadica+danilov_yolo11s_768_e150) printed "LEAKAGE CHECK PASSED"
over a split whose 3996 CADICA frames counted as 3996 separate patient groups.
"""
import os

import cv2
import numpy as np
import pytest

from src.data_prep import cadica_to_yolo


def _write_cadica_tree(root, patients, videos=2, frames=5, gt_line="100 100 40 40 p20_50"):
    """Materialize a minimal real CADICA layout: <root>/<patient>/<video>/{input,groundtruth}.

    ``videos=None`` writes the flat variant (``input``/``groundtruth`` directly under the patient
    dir), used by the negative test so each patient still gets a distinct split key even when
    ``cadica_patient_of`` cannot parse a ``p<digits>`` component."""
    for p in patients:
        vids = [None] if videos is None else [f"v{v}" for v in range(1, videos + 1)]
        for vid in vids:
            vdir = os.path.join(root, p) if vid is None else os.path.join(root, p, vid)
            indir, gtdir = os.path.join(vdir, "input"), os.path.join(vdir, "groundtruth")
            os.makedirs(indir, exist_ok=True)
            os.makedirs(gtdir, exist_ok=True)
            for f in range(frames):
                stem = f"{p}_{f:05d}" if vid is None else f"{p}_{vid}_{f:05d}"
                cv2.imwrite(os.path.join(indir, stem + ".png"),
                            np.full((512, 512), 128, dtype=np.uint8))
                open(os.path.join(gtdir, stem + ".txt"), "w").write(gt_line + "\n")
    return root


def _cfg(root):
    return {"datasets": {"cadica": {"root": root}}, "model": {"imgsz": 64}}


def test_main_audits_its_own_output_and_reports_cadica_grouping(tmp_path, monkeypatch, capsys):
    # 8 patients so the deterministic hash split lands some on each side (a one-sided split would
    # make the auditor's empty-split assertion fire for reasons unrelated to grouping).
    src = _write_cadica_tree(str(tmp_path / "raw"), [f"p{i}" for i in range(1, 9)])
    out = str(tmp_path / "proc")
    monkeypatch.setattr(cadica_to_yolo, "OUT", out)

    cadica_to_yolo.main(_cfg(src))

    printed = capsys.readouterr().out
    assert "LEAKAGE CHECK PASSED" in printed, printed
    assert "8 patient groups" in printed, printed


def test_main_raises_when_written_stems_defeat_group_key(tmp_path, monkeypatch):
    # Patient dirs that are not 'p<digits>' -> cadica_patient_of() returns None -> the output stems
    # are not 'p<patient>_v<video>_<frame>' -> group_key cannot collapse them -> per-frame split.
    # main() must refuse to produce a data.yaml for that, not print a success line over it.
    src = _write_cadica_tree(str(tmp_path / "raw"), [f"case{i}" for i in range(1, 9)], videos=None)
    out = str(tmp_path / "proc")
    monkeypatch.setattr(cadica_to_yolo, "OUT", out)

    with pytest.raises(AssertionError, match="UNGROUPED CADICA"):
        cadica_to_yolo.main(_cfg(src))
