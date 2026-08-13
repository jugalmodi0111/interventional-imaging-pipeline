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
