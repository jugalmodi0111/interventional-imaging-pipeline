"""Input validity gate (Dialygo B3) — the single-modality replacement for the modality router.

B3 requires the service to "reject any input that is not a valid vascular-access angiogram (wrong
modality, corrupt or unrelated image) rather than attempting to read it". Model One is a
single-modality product, so there is nothing to *route* between; what the request path actually
needs is a gate that decides whether this frame may be read at all.

**What this gate honestly does — and does not do.** It is a deterministic
plausibility/acquisition-quality check on pixel statistics: shape, channel count, size, dynamic
range and endpoint clipping. It catches corrupt files, blank/black frames, blown-out acquisitions,
colour images, screenshots and wrong-shaped tensors. It **cannot** tell a coronary angiogram from a
chest X-ray — discriminating imaging modality requires a learned OOD head trained on real
in-distribution data, which does not exist yet (no cleared AVF data; see PROJECT_TRACKER §7). So
this closes the "corrupt or unrelated image" half of B3 and leaves the "wrong modality" half open,
tracked as such rather than quietly claimed. `assess_frame` is the seam a learned scorer plugs into
later without touching the orchestrator.

**Protocol.** `ValidityGate.classify(frame)` returns a `ModalityDecision`, the same shape the
orchestrator already consumes, so the decision layer's dispatch and every C2/C3/C4 fail-safe stay
byte-identical. A valid frame resolves to the one configured modality; anything else resolves to
`"unknown"` with `deferred=True`, which the orchestrator turns into a deferred study — a rejected
frame can therefore never reach a disease model.

**Safety default is REJECT.** Every failure mode — unreadable input, a surprising type, an
exception inside a check — collapses to a deferred decision. The gate sits in the request path and
must never raise; numpy-only, no torch, no lazy heavy imports.
"""
from dataclasses import dataclass

import numpy as np

# Defaults. Tuned to be permissive about *content* (that is not this gate's job) and strict about
# the failure modes that actually reach a service: corrupt reads, blank frames, clipped
# acquisitions, colour screenshots.
MIN_SIDE = 128            # px; below this no angiographic detail survives
REJECT_RANGE = 20         # p99-p1 in 8-bit levels; below this the frame carries no structure
ACCEPT_RANGE = 60         # p99-p1 at/above which contrast is unambiguously angiogram-like
MAX_CLIPPED = 0.90        # max fraction of pixels pinned at 0 or 255
ACCEPT_SCORE = 0.50       # score at/above which the gate resolves instead of deferring


@dataclass
class ModalityDecision:
    """The decision the orchestrator consumes. Field-compatible with the router's decision so the
    dispatch path, its tests and the event topics are unchanged by the router's removal."""
    modality: str
    view: str | None
    quality_ok: bool
    confidence: float
    deferred: bool
    reason: str


def _as_gray(frame):
    """Coerce input to a 2-D uint8-ish array, or return (None, reason). Never raises."""
    if frame is None:
        return None, "no-frame"
    try:
        a = np.asarray(frame)
    except Exception:
        return None, "unreadable-frame"
    if a.dtype == object or not np.issubdtype(a.dtype, np.number):
        return None, "unreadable-frame"
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        # 3-channel colour, a batched tensor, a scalar: none of these is a single angiographic frame.
        return None, "not-grayscale" if a.ndim == 3 else "unreadable-frame"
    if a.size == 0:
        return None, "unreadable-frame"
    return a, ""


def assess_frame(frame, *, min_side=MIN_SIDE, reject_range=REJECT_RANGE,
                 accept_range=ACCEPT_RANGE, max_clipped=MAX_CLIPPED):
    """Frame -> ``(ok, score, reason)``. ``score`` in [0,1] rises with angiogram-like contrast.

    ``ok`` is True only for a frame that passes every structural check AND clears
    ``reject_range``; the caller applies its own accept threshold to ``score`` to decide whether a
    passing-but-marginal frame should still defer. Any failure returns ``(False, 0.0, <reason>)``.
    Wrapped end-to-end so a surprising input defers instead of raising into the request path.
    """
    try:
        a, why = _as_gray(frame)
        if a is None:
            return False, 0.0, why
        if min(a.shape) < min_side:
            return False, 0.0, "too-small"

        f = a.astype(np.float64)
        lo, hi = np.percentile(f, 1), np.percentile(f, 99)
        rng = float(hi - lo)

        # Domain detection, so a float [0,1] image is judged on the same scale as an 8-bit one.
        scale = 1.0 if float(f.max()) <= 1.0 else 255.0
        rng_norm = rng / scale * 255.0
        if rng_norm < reject_range:
            return False, 0.0, "degenerate-contrast"

        # Clipping means pinned at the SENSOR endpoints (0 / full scale) -- a blown-out or crushed
        # acquisition. It must NOT be measured against the frame's own min/max: by construction
        # every frame attains those, so a legitimate low-contrast frame would read as 100% clipped.
        clipped = float(np.mean((f <= 0.0) | (f >= scale)))
        if clipped > max_clipped:
            return False, 0.0, "saturated"

        span = max(accept_range - reject_range, 1e-9)
        score = float(np.clip((rng_norm - reject_range) / span, 0.0, 1.0))
        return True, score, "valid"
    except Exception:
        return False, 0.0, "gate-error"


class ValidityGate:
    """Router-shaped adapter: gates a frame for ONE deployed modality.

    ``modality`` is the single modality this deployment serves. A frame that clears the gate
    resolves to it; anything else resolves to ``"unknown"`` + ``deferred=True`` so the orchestrator
    defers the study and no disease model runs.
    """

    VERSION = "validity-gate/heuristic-v1"

    def __init__(self, modality, *, accept_score=ACCEPT_SCORE, view=None, **assess_kwargs):
        self.modality = modality
        self.accept_score = accept_score
        self.view = view
        self.assess_kwargs = assess_kwargs

    @property
    def weights(self):
        """Version string for the audit trail / `versions` map. Named to match what the
        orchestrator reads off the decision source; this gate has no weights file."""
        return self.VERSION

    def classify(self, frame_gray):
        ok, score, reason = assess_frame(frame_gray, **self.assess_kwargs)
        if not ok:
            return ModalityDecision("unknown", self.view, False, score, True, reason)
        if score < self.accept_score:
            # Structurally fine but marginal: the honest answer is "uncertain", and B3 says an
            # uncertain read defers rather than resolving either way.
            return ModalityDecision("unknown", self.view, False, score, True, "validity-uncertain")
        return ModalityDecision(self.modality, self.view, True, score, False, "valid")
