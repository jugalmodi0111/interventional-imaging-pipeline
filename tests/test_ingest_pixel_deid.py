"""Burned-in overlay text: geometric detection, masking, and the fail-safe review flag.

Dialygo B5: every test here runs on the SYNTHETIC DICOM fixture only. No real study is ever
opened by the test suite.
"""
import numpy as np

from src.ingest.pixel_deid import (SCREEN_FRACTION, detect_text_regions, mask_regions,
                                   needs_review)

from tests.fixtures.synthetic_dicom import make_xa_dataset

ROWS = COLS = 64


def _u8(frame):
    """Linear 12-bit -> 8-bit scale (no VOI LUT).

    The fixture's burned-in band is written at 4000 (near the 12-bit ceiling of 4095, see
    tests/fixtures/synthetic_dicom._BANNER), so ``>> 4`` puts it at 250 -- comfortably above the
    SATURATION_FRACTION threshold -- while leaving the vessel (3000 >> 4 == 187) below saturation.
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
    ds = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS, burned_in=True)
    top_end, bot_start = _bands(ROWS)
    for x, y, w, h in detect_text_regions(_u8(ds.pixel_array[0])):
        assert 0 <= x and x + w <= COLS, "box must stay inside the frame horizontally"
        assert y + h <= top_end or y >= bot_start, "only the top/bottom bands are screened"


def test_malformed_input_degrades_to_no_boxes():
    # Fail-safe: a colour/3-D array or an empty array must not raise -- it returns nothing to mask
    # and needs_review (below) is what keeps such a frame out of the clean store.
    assert detect_text_regions(np.zeros((0, 0), np.uint8)) == []
    assert detect_text_regions(np.zeros((8, 8, 3), np.uint8)) == []


# --- masking: destroy the pixels, keep the anatomy, never touch the caller's array --------------

def test_mask_zeroes_the_burned_in_band_and_keeps_the_vessel():
    ds = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS, burned_in=True)
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
    ds = make_xa_dataset(n_frames=2, rows=ROWS, cols=COLS, burned_in=True)
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
