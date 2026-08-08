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
