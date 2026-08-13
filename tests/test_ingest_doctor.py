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


def test_ingest_sites_config_is_b5_safe():
    """The shipped site config must not point at any real drive.

    Adapted from the Task 16 plan: VALID_MODES = ("synthetic", "real") in
    src/ingest/clearance.py -- "cleared" is not a mode, so this checks
    mode == "synthetic" rather than the plan's original "cleared" claim
    (audit 2026-08-03 P1, Task 16 row).
    """
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
    assert cfg["mode"] == "synthetic"
    assert "clean_root" in cfg
