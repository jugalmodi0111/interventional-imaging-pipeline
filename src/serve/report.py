"""Output contract for the diagnostic orchestrator. A StudyReport is what a clinician sees: modality,
per-finding screening flags with calibrated confidence, and an explicit study-level defer. `to_dict`
is JSON-safe for the /analyze endpoint: box tuples -> lists, and every numpy scalar anywhere in the
payload -> its built-in Python equivalent. The latter is not cosmetic: infer.py's CoreML box parser
emits numpy.float32 coordinates/confidences, and json.dumps raises TypeError on any numpy scalar --
which turned EVERY positive finding into an HTTP 500 at serialization time (2026-08-03 audit, P3
critical 3). A positive finding must never be the one case the endpoint cannot serialize.
"""
from dataclasses import dataclass, field, asdict


def _jsonable(v):
    """Recursively convert a value to built-in JSON-safe types: dict/list/tuple containers are
    rebuilt, and any numpy scalar (float32 coords, int64 counts, bool_ flags) collapses to its
    Python builtin. Duck-typed via `.item()` so this module never has to import numpy -- it stays
    dependency-free and import-safe like the rest of the serve layer's pure modules."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, float):                     # builtin float AND numpy.float64 (a float subclass)
        return float(v)
    item = getattr(v, "item", None)              # numpy scalar (float32/int64/bool_) -> builtin
    if callable(item):
        try:
            return _jsonable(v.item())
        except (TypeError, ValueError):          # .item() exists but isn't a 1-element scalar
            pass
    return v


@dataclass
class Finding:
    label: str
    display_name: str
    confidence: float
    deferred: bool
    reason: str
    severity: str | None = None
    boxes: list = field(default_factory=list)


@dataclass
class StudyReport:
    modality: str
    view: str | None
    quality_ok: bool
    findings: list
    deferred: bool
    defer_reason: str
    frames_analyzed: int
    model_versions: dict

    def to_dict(self):
        """JSON-safe dict: every numpy scalar anywhere in the payload becomes a Python builtin
        (see `_jsonable`), and every box coordinate/confidence is additionally forced to a builtin
        float -- `json.dumps(report.to_dict())` must never raise, least of all on a positive
        finding."""
        d = _jsonable(asdict(self))
        for f in d["findings"]:
            f["boxes"] = [[float(v) for v in b] for b in f["boxes"]]
        return d
