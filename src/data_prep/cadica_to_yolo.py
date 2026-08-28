"""CADICA (2024) -> YOLO single class 'stenosis'. Patient-diverse coronary angiography w/ lesion boxes.

CADICA layout (verified against Mendeley p9bpx9ctcv):
    selectedVideos/pX/vY/input/<frame>.png            # ALL angiography frames (10 fps)
    selectedVideos/pX/vY/groundtruth/<frame>.txt      # one lesion box per line; keyframes only,
                                                      # and ONLY for lesion videos
NB the GT dir is lowercase ``groundtruth`` on Mendeley (some mirrors camel-case it) and Kaggle's
filesystem is case-SENSITIVE, so the sibling lookup below is case-insensitive.

Each groundtruth line is ``x y w h [label]`` where (x, y) is the ABSOLUTE-pixel TOP-LEFT corner,
(w, h) the box size in pixels, and the optional trailing ``label`` a severity string (e.g. ``p20_50``);
every severity collapses to the single YOLO class 0 ('stenosis'), and a 4-field line (no label) also
works. Patients are p1..p42; the split GROUP KEY is the PATIENT (``pX``) so all of a patient's frames
land on one side of the train/val split (no frame leakage), mirroring the Danilov converter. Only
keyframes carry a groundtruth .txt, so a frame is converted iff it has one (annotation-driven).

Heavy deps (cv2) are imported lazily inside the conversion functions, so the pure helpers below
(`cadica_boxes_to_yolo_lines`, `parse_cadica_gt`, `cadica_patient_of`, `cadica_video_label`,
`_evenly_spaced`, `_select_negatives`, `_cap_records`) can be imported and unit-tested without cv2
installed. Writes to the SHARED stenosis OUT dir like Danilov.

Optional per-patient frame cap (config `datasets.cadica.max_frames_per_patient`, default None = no
cap): CADICA ships ~100 near-duplicate keyframes/patient, which dilutes the split with redundant
frames; capping keeps an evenly-spaced subset per patient (see `_cap_records` / `io_utils.
cap_frames_per_patient`).

NEGATIVE (background) frames — config `datasets.cadica.negatives_per_positive`, default 0.25, `0`
disables. Added 2026-08-16 after the architecture audit (docs/STENOSIS_ARCHITECTURE_AUDIT.md A1)
found the stenosis corpus contained ZERO label-less images: this converter's annotation-driven walk
dropped every non-lesion video whole (they ship no `groundtruth` dir), and ARCADE/Danilov annotate
every frame, so CADICA was the only source that HELD negatives and it threw them away. A
single-class detector trained purely on positives has no gradient teaching it not to fire; it
learned "always propose a box" (per-video false-flag rate ~1.0, negative Youden J). Ultralytics'
own guidance is ~0-10% background images; this corpus had 0%.

The correctness constraint (audit A1a) governs the whole implementation: ONLY frames from NON-LESION
VIDEOS are valid negatives. An unannotated frame inside a LESION video is NOT a negative — the
lesion is physically present, CADICA merely annotates keyframes — so using it as background teaches
the model to suppress true positives and destroys recall. The partition is therefore on the VIDEO's
lesion/non-lesion label (`cadica_video_label`: CADICA's per-patient `lesionVideos.txt` /
`nonlesionVideos.txt` manifests first, groundtruth-dir presence only as a fallback), NEVER on
per-frame annotation presence.
"""
import argparse, glob, os, re, yaml

OUT = "data/processed/stenosis"
NAMES = ("stenosis",)
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-testable without cv2)
# --------------------------------------------------------------------------------------------------
def cadica_boxes_to_yolo_lines(boxes_xywh_abs, W, H):
    """CADICA GT boxes (absolute top-left x,y,w,h in px) -> YOLO lines, all class 0 'stenosis'.

    Each box is normalized by image (``W``, ``H``) and converted from top-left to CENTER form. Every
    CADICA severity label collapses to the single class 0. Returns ``["0 cx cy w h", ...]`` (6-dp),
    one line per box (empty list -> a negative/background frame with an empty label file).
    """
    lines = []
    for box in boxes_xywh_abs:
        x, y, w, h = (float(v) for v in box[:4])
        lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    return lines


def parse_cadica_gt(text):
    """Parse a CADICA groundTruth file body -> ``[(x, y, w, h), ...]`` floats (label dropped).

    Each nonblank line is ``x y w h label``; lines with fewer than 4 leading numeric fields are
    skipped (robust to blank lines / malformed rows)."""
    boxes = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            x, y, w, h = (float(p) for p in parts[:4])
        except ValueError:
            continue
        boxes.append((x, y, w, h))
    return boxes


def cadica_patient_of(path):
    """Return the CADICA patient id (``pXX``) found in ``path``, else None.

    The patient is the train/val split GROUP KEY (all of ``pXX``'s frames land on one side)."""
    for part in os.path.normpath(path).split(os.sep):
        if re.fullmatch(r"p\d+", part):
            return part
    return None


def _sibling_dir(parent, name):
    """Case-insensitive lookup of subdir ``name`` under ``parent``. CADICA ships lowercase
    ``groundtruth`` but mirrors camel-case it, and Kaggle's filesystem is case-SENSITIVE, so a
    hardcoded name silently finds nothing there. Returns the matching path or None."""
    low = name.lower()
    try:
        for e in sorted(os.listdir(parent)):
            if e.lower() == low and os.path.isdir(os.path.join(parent, e)):
                return os.path.join(parent, e)
    except OSError:
        pass
    return None


def _out_stem(patient, video, frame_stem):
    """Globally-unique output stem so frames from different patients/videos never clobber one path.

    Real CADICA frames are already named ``pXX_vYY_NNNNN``; if the frame stem doesn't already start
    with the patient id, prefix ``<patient>_<video>_``."""
    if patient and frame_stem.startswith(patient + "_"):
        return frame_stem
    return "_".join([p for p in (patient, video) if p] + [frame_stem])


def _manifest_videos(patient_dir, name):
    """Case-insensitive read of a CADICA per-patient manifest -> set of entries, or None if absent.

    ``name`` is compared lowercased against the patient dir's entries because Mendeley ships
    ``lesionVideos.txt``/``nonlesionVideos.txt`` camel-cased while mirrors (and Kaggle's
    case-SENSITIVE filesystem) vary. ``None`` means "no such manifest" and is deliberately distinct
    from ``set()`` ("manifest present but empty") — the caller's priority order depends on it.
    """
    low = name.lower()
    try:
        entries = sorted(os.listdir(patient_dir))
    except OSError:
        return None
    for e in entries:
        p = os.path.join(patient_dir, e)
        if e.lower() == low and os.path.isfile(p):
            try:
                with open(p, errors="replace") as f:
                    return {ln.strip() for ln in f if ln.strip()}
            except OSError:
                return None
    return None


def _video_keys(patient, video):
    """Both spellings CADICA's manifests use for one video: bare ``v1`` and prefixed ``p3_v1``."""
    keys = {video}
    if patient:
        keys.add(f"{patient}_{video}")
    return {k.lower() for k in keys}


def cadica_video_label(patient_dir, patient, video, has_gt, cache=None):
    """Lesion/non-lesion label for ONE CADICA video -> ``(label, source)``.

    ``label`` is ``'lesion'`` / ``'nonlesion'`` / ``'unknown'``; ``source`` is ``'manifest'`` or
    ``'groundtruth'`` so a caller can report honestly which evidence it acted on.

    Priority (audit A1a — getting this wrong destroys recall):
      1. CADICA's per-patient ``lesionVideos.txt`` / ``nonlesionVideos.txt``. AUTHORITATIVE: a lesion
         video that ships no groundtruth dir still has a lesion in it, and the fallback below would
         mislabel it non-lesion and hand real lesions to the model as background.
      2. Fallback ONLY when neither manifest exists: a video with no groundtruth dir — or one whose
         groundtruth dir holds no ``.txt`` (a mirror artifact) — is non-lesion. Caller passes that
         as ``has_gt``.

    When the manifests ARE present but list this video in neither, the label is ``'unknown'``: the
    video is not *proven* lesion-free, so it fails closed (never sampled as a negative) rather than
    silently inheriting the weaker fallback rule.

    ``cache`` is an optional dict reused across the videos of one patient so the manifests are read
    once per patient dir rather than once per video.
    """
    if cache is None or patient_dir not in cache:
        pair = (_manifest_videos(patient_dir, "lesionvideos.txt"),
                _manifest_videos(patient_dir, "nonlesionvideos.txt"))
        if cache is not None:
            cache[patient_dir] = pair
    else:
        pair = cache[patient_dir]
    lesion, nonlesion = pair
    if lesion is not None or nonlesion is not None:
        keys = _video_keys(patient, video)
        if lesion and keys & {e.lower() for e in lesion}:
            return "lesion", "manifest"
        if nonlesion and keys & {e.lower() for e in nonlesion}:
            return "nonlesion", "manifest"
        return "unknown", "manifest"
    return ("lesion" if has_gt else "nonlesion"), "groundtruth"


def _evenly_spaced(seq, k):
    """``k`` items of ``seq`` at evenly-spaced indices spanning the WHOLE range (both endpoints).

    Same arithmetic as ``io_utils.cap_frames_per_patient`` so negatives are chosen the way positives
    are capped: strictly better temporal coverage than first-``k``, and fully deterministic (no RNG —
    converters in this repo must reproduce byte-identically across runs). ``k >= len(seq)`` keeps
    everything; ``k <= 0`` keeps nothing.
    """
    m = len(seq)
    if k <= 0:
        return []
    if k >= m:
        return list(seq)
    if k == 1:
        return [seq[m // 2]]
    return [seq[round(i * (m - 1) / (k - 1))] for i in range(k)]


def _select_negatives(records, n):
    """Pick exactly ``n`` of ``records`` (``(patient, video, img_path, None)``) as background frames.

    Two properties the sampling must have, both deterministic and RNG-free:

    * SPREAD ACROSS PATIENTS — quotas are handed out one at a time, round-robin over the videos
      ordered patient-major-interleaved (every patient's 1st video, then every patient's 2nd, ...).
      Drawing ~2000 negatives from a single patient's clips would teach the model one patient's
      background statistics, not "no lesion".
    * EVENLY SPACED WITHIN A VIDEO — each video's quota is taken at spread indices
      (``_evenly_spaced``), so a clip contributes frames across its whole temporal range instead of
      a burst of near-identical consecutive frames from its start.

    Returns fewer than ``n`` only when the non-lesion pool is smaller than ``n``.
    """
    if n <= 0 or not records:
        return []
    by_video, by_patient = {}, {}
    for rec in records:
        key = (rec[0], rec[1])
        if key not in by_video:
            by_video[key] = []
            by_patient.setdefault(rec[0], []).append(key)
        by_video[key].append(rec)
    order = []
    for vids in by_patient.values():
        vids.sort()
    patients = sorted(by_patient)
    for i in range(max(len(v) for v in by_patient.values())):
        for p in patients:
            if i < len(by_patient[p]):
                order.append(by_patient[p][i])
    quota = dict.fromkeys(order, 0)
    remaining = n
    while remaining > 0:
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if quota[key] < len(by_video[key]):
                quota[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:                      # whole non-lesion pool already taken
            break
    picked = []
    for key in order:
        frames = sorted(by_video[key], key=lambda r: r[2])
        picked.extend(_evenly_spaced(frames, quota[key]))
    return sorted(picked, key=lambda r: r[2])


def _cap_records(records, k):
    """Select which ``(patient, video, img_path, gt_path)`` records survive a per-patient frame cap.

    ``records`` is the materialized ``_iter_frames`` output. Builds each record's OUTPUT stem (via
    ``_out_stem``, same as the real conversion path) and delegates the actual selection to
    ``io_utils.cap_frames_per_patient`` keyed by ``group_key`` on that stem, so capping groups by the
    SAME patient key the split/leakage accounting uses (evenly-spaced picks per patient, not first-k).
    ``k=None`` keeps every record (sorted by out-stem, for determinism).

    ``io_utils`` (and the cv2/numpy it imports at module load) is only imported inside this function,
    not at module top, so importing ``cadica_to_yolo`` itself still doesn't require cv2 -- only
    calling this (or ``_convert``) does.
    """
    from src.data_prep.io_utils import cap_frames_per_patient, group_key
    by_stem = {}
    for rec in records:
        patient, video, ip, gp = rec
        frame_stem = os.path.splitext(os.path.basename(ip))[0]
        by_stem[_out_stem(patient, video, frame_stem)] = rec
    kept_stems = cap_frames_per_patient(by_stem.keys(), k, key_fn=group_key)
    return [by_stem[s] for s in kept_stems]


# --------------------------------------------------------------------------------------------------
# Discovery (stdlib) + conversion (needs cv2 / io_utils, imported inside the functions)
# --------------------------------------------------------------------------------------------------
def _iter_videos(root):
    """Yield ``(patient, video, patient_dir, input_dir, gt_dir)`` for every CADICA video under root.

    Discovers every ``input`` dir (tolerating whether the ``selectedVideos`` wrapper is present);
    ``gt_dir`` is the sibling ``groundTruth`` looked up case-insensitively (lowercase on Mendeley,
    camel-cased on some mirrors, and Kaggle's filesystem is case-SENSITIVE) or None. ``patient_dir``
    is the video dir's parent, where CADICA's per-patient manifests live. Deterministic (sorted)."""
    for input_dir in sorted(glob.glob(os.path.join(root, "**", "input"), recursive=True)):
        if not os.path.isdir(input_dir):
            continue
        video_dir = os.path.dirname(input_dir)
        yield (cadica_patient_of(input_dir) or os.path.basename(video_dir),
               os.path.basename(video_dir), os.path.dirname(video_dir), input_dir,
               _sibling_dir(video_dir, "groundtruth"))


def _iter_images(input_dir):
    for ip in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if os.path.splitext(ip)[1].lower() in _IMG_EXTS:
            yield ip


def _gt_has_txt(gt_dir):
    """True iff ``gt_dir`` exists and holds at least one ``.txt``. A groundtruth dir with no .txt is
    a mirror artifact, not evidence that the video contains a lesion."""
    if not gt_dir:
        return False
    try:
        return any(f.lower().endswith(".txt") for f in os.listdir(gt_dir))
    except OSError:
        return False


def _iter_frames(root):
    """Yield ``(patient, video, img_path, gt_path)`` for every CADICA frame that has a GT .txt.

    POSITIVES only, unchanged: a video with no groundtruth dir is skipped, and within a lesion video
    only the annotated keyframes are converted. The frames this discards are handled separately —
    and on the VIDEO's label, never per-frame — by ``_negative_records``. Robust to missing dirs
    (skip, don't crash). Deterministic (sorted)."""
    for patient, video, _pdir, input_dir, gt_dir in _iter_videos(root):
        if not gt_dir:
            continue
        for ip in _iter_images(input_dir):
            stem = os.path.splitext(os.path.basename(ip))[0]
            gp = os.path.join(gt_dir, stem + ".txt")
            if not os.path.isfile(gp):
                continue
            yield patient, video, ip, gp


def _negative_records(root):
    """Every candidate background frame -> ``([(patient, video, img_path, None), ...], source)``.

    A frame is a candidate iff its VIDEO is labelled non-lesion by ``cadica_video_label`` (manifests
    first, groundtruth-dir presence only as a fallback). ``'unknown'`` videos — present in the
    manifests' patient but listed in neither file — are excluded: unproven is not lesion-free.

    ``source`` is ``'manifest'`` / ``'groundtruth'`` / ``'mixed'`` (or None for an empty tree), the
    aggregate over the videos actually inspected, so ``main`` can print which evidence it acted on.
    """
    records, sources, cache = [], set(), {}
    for patient, video, patient_dir, input_dir, gt_dir in _iter_videos(root):
        label, source = cadica_video_label(patient_dir, patient, video, _gt_has_txt(gt_dir), cache)
        sources.add(source)
        if label != "nonlesion":
            continue
        records.extend((patient, video, ip, None) for ip in _iter_images(input_dir))
    if not sources:
        return records, None
    return records, (sources.pop() if len(sources) == 1 else "mixed")


def _write_frame(out_dir, size, patient, stem, gray, lines, cv2, io):
    """One YOLO (image, label) pair. ``lines=[]`` writes a 0-byte label = an ultralytics BACKGROUND.

    Split is ``io.split_of(patient)`` for positives AND negatives alike, so a patient's negatives
    always land on the same side as its positives — splitting negatives independently would put the
    same patient's frames on both sides and re-open the leak this converter exists to avoid."""
    sp = io.split_of(patient)                               # PATIENT-grouped split -> no frame leak
    io.ensure(os.path.join(out_dir, "images", sp), os.path.join(out_dir, "labels", sp))
    cv2.imwrite(os.path.join(out_dir, "images", sp, stem + ".png"),
                cv2.resize(io.clahe_unsharp(gray), (size, size)))
    with open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w") as f:
        f.write("\n".join(lines))


def _convert(root, out_dir, size, max_per_patient=None, negatives_per_positive=0.25):
    """CLAHE+resize each CADICA frame, write YOLO images/labels/{train,val} split by PATIENT.

    If ``max_per_patient`` is set, the record list is capped per patient (``_cap_records``, evenly
    spaced across each patient's frames) BEFORE any image is read/written, so the near-duplicate
    keyframes that get dropped never touch disk.

    ``negatives_per_positive`` (0 = off, the pre-2026-08-16 behaviour) additionally samples
    ``round(ratio * positives_written)`` BACKGROUND frames from NON-LESION videos only — same
    CLAHE+resize preprocessing, same patient-grouped split, but an EMPTY label file, which is exactly
    what ultralytics counts as a background. Sampling is proportional to the positives ACTUALLY
    written (i.e. after ``max_per_patient``), so the ratio holds whatever the cap is.

    Returns ``{"positives": [...stems], "negatives": [...stems], "label_source": str|None,
    "negatives_available": int, "drops": [row, ...]}``. ``main`` hands positives+negatives to
    ``audit_split_leakage(cadica_stems=...)`` so the grouping is proven independently of
    ``group_key``'s regex rather than assumed, and uses ``negatives_available`` to decide whether a
    zero-background result is a bug (there were non-lesion videos) or just a lesion-only tree.

    ``drops`` is the audit-B2 record of frames that could not be converted (an undecodable file ->
    ``unreadable``), each also appended to ``<out_dir>/convert_errors.jsonl``. It used to be a bare
    ``continue``: a corrupt mirror could silently halve a patient's frames -- positives AND
    backgrounds alike -- and the printed count looked perfectly healthy."""
    import cv2
    from src.data_prep import io_utils as io
    records = list(_iter_frames(root))
    if max_per_patient is not None:
        records = _cap_records(records, max_per_patient)
    written, drops = [], []
    for patient, video, ip, gp in records:
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            io.record_drop(out_dir, ip, "unreadable", drops=drops)
            continue
        H, W = g.shape
        with open(gp) as f:
            boxes = parse_cadica_gt(f.read())
        lines = cadica_boxes_to_yolo_lines(boxes, W, H)     # normalized by ORIGINAL W,H (pre-resize)
        stem = _out_stem(patient, video, os.path.splitext(os.path.basename(ip))[0])
        _write_frame(out_dir, size, patient, stem, g, lines, cv2, io)
        written.append(stem)

    out = {"positives": written, "negatives": [], "label_source": None, "negatives_available": 0,
           "drops": drops}
    if not (negatives_per_positive and written):
        return out
    candidates, out["label_source"] = _negative_records(root)
    out["negatives_available"] = len(candidates)
    target = int(round(len(written) * float(negatives_per_positive)))
    pos = set(written)
    for patient, video, ip, _ in _select_negatives(candidates, target):
        stem = _out_stem(patient, video, os.path.splitext(os.path.basename(ip))[0])
        if stem in pos:                 # defensive: a lesion frame must never be overwritten by a
            continue                    # background one (non-lesion videos are disjoint, so N/A)
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is None:
            io.record_drop(out_dir, ip, "unreadable", drops=drops)
            continue
        _write_frame(out_dir, size, patient, stem, g, [], cv2, io)   # [] -> 0-byte label file
        out["negatives"].append(stem)
    return out


def main(cfg):
    from src.data_prep import io_utils as io
    d = cfg.get("datasets", {}).get("cadica")
    if not d or not d.get("root"):
        print("[cadica] no 'cadica' dataset configured; skipping.")
        return 0
    root = d["root"]
    size = cfg.get("model", {}).get("imgsz", 640)
    max_per_patient = d.get("max_frames_per_patient")
    # Fallback MUST match the shipped config (configs/stenosis_yolo.yaml), or dropping the key
    # silently rebuilds a different corpus. It was 1.0 -- the swept value that cannot reach 0.90
    # per-video sensitivity at any measured threshold. 0 disables (pre-2026-08-16 behaviour).
    npp = d.get("negatives_per_positive", 0.25)
    res = _convert(root, OUT, size, max_per_patient=max_per_patient, negatives_per_positive=npp)
    stems, negs = res["positives"], res["negatives"]
    n, n_neg = len(stems), len(negs)

    # Audit B2: report the drops BEFORE the empty-corpus return, because "everything was dropped"
    # is exactly the case whose cause an operator needs, and print the count even when it is zero
    # so two runs can be diffed at all (a silent drop is how a corpus goes quietly incomplete).
    drops = res["drops"]
    msg = f"[cadica] dropped: {len(drops)} frames"
    if drops:
        msg += f" {io.drop_reason_counts(drops)} -> {io.convert_errors_path(OUT)}"
    print(msg)

    if n == 0:
        print(f"[cadica] WARNING: no CADICA frames converted under {root!r}; check layout.")
        return 0

    # Negative-sampling summary. Say which evidence the lesion/non-lesion partition rests on: the
    # manifests are authoritative, the groundtruth-presence fallback is an INFERENCE, and reading
    # "N backgrounds" without knowing which was used is how a mislabelled lesion video would slip
    # into the background pool unnoticed.
    src_desc = {"manifest": "CADICA lesionVideos.txt/nonlesionVideos.txt manifests",
                "groundtruth": "groundtruth-dir presence (manifests absent)",
                "mixed": "MIXED: manifests where present, groundtruth-dir presence elsewhere",
                }.get(res["label_source"], "n/a (no negatives sampled)")
    print(f"[cadica] negatives: {n_neg} background frames written from "
          f"{res['negatives_available']} candidate frames in non-lesion videos "
          f"(negatives_per_positive={npp}); lesion/non-lesion truth from {src_desc}")
    if npp and n_neg == 0:
        print(f"[cadica] WARNING: negative sampling is ON but no non-lesion video was found under "
              f"{root!r}, so the corpus still has ZERO background frames — the 2026-08-16 root "
              f"cause (docs/STENOSIS_ARCHITECTURE_AUDIT.md A1). Check the manifests / layout.")

    # Honesty gate, mirroring danilov_to_yolo.main. CADICA is written into the SAME OUT tree AFTER
    # the Danilov converter has already audited it, so without this call nothing ever audits the
    # combined corpus -- the 2026-07-16 hole. Passing the stems we just wrote also proves the
    # p<patient>_v<video>_<frame> collapse actually happened (a regex no-op degrades the split to
    # per-frame and makes every group/val-fraction number downstream wrong).
    #
    # require_backgrounds makes the 2026-08-16 fix SELF-VERIFYING at conversion time: with negative
    # sampling on and non-lesion videos present, a split that still contains no label-less image is
    # the bug, not a warning. It is not required when there was nothing to sample (npp=0, or a
    # lesion-only tree) -- that case is reported by the WARNING above instead of a hard failure,
    # because the corpus, not the converter, is what is missing.
    rep = io.audit_split_leakage(OUT, cadica_stems=stems + negs,
                                 require_backgrounds=bool(npp) and res["negatives_available"] > 0)
    yml = io.write_yolo_datayaml(OUT, names=NAMES)
    print(f"CADICA -> {OUT} : {n + n_neg} frames "
          f"({n} positive + {n_neg} background) ; data cfg {yml}")
    print(f"LEAKAGE CHECK PASSED — split is patient-grouped and honest\n"
          f"  train {rep['train_imgs']} imgs / {rep['train_groups']} groups | "
          f"val {rep['val_imgs']} imgs / {rep['val_groups']} groups "
          f"(val ~{rep['val_frac_by_group']:.0%} by group)\n"
          f"  backgrounds: {rep['train_backgrounds']} train / {rep['val_backgrounds']} val "
          f"({rep['background_frac']:.1%} of the split; ultralytics guidance is 0-10%)\n"
          f"  cadica: {rep['cadica']['cadica_frames']} frames -> "
          f"{rep['cadica']['patient_groups']} patient groups, "
          f"{rep['cadica']['ungrouped']} ungrouped ({rep['cadica']['ungrouped_frac']:.0%})")
    return n + n_neg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    a = ap.parse_args(); main(yaml.safe_load(open(a.config)))
