"""Train coronary segmentation: nnU-Net teacher (CLI) -> knowledge-distill TinyU-Net student.

Library entrypoint `train(cfg)` — call it from the Colab/Kaggle GPU notebook. Heavy: pulls
torch + the nnU-Net CLI, so those imports are LAZY (inside the functions that use them) and this
module imports fine on a torch-less box. The pure config->argv/kwargs helpers below are unit-tested.
"""
import argparse, glob, os, subprocess, sys, yaml


def student_kwargs(cfg):
    """cfg['student'].{base_ch,depth} -> TinyUNet/to_coreml kwargs {base, depth} (defaults 16/4)."""
    s = cfg.get("student", {})
    return {"base": s.get("base_ch", 16), "depth": s.get("depth", 4)}


def distill_kwargs(cfg):
    """Only the kwargs distill() accepts: temperature->T, alpha, + train epochs/lr/amp."""
    d, tr = cfg.get("distill", {}), cfg.get("train", {})
    return {"alpha": d.get("alpha", 0.5), "T": d.get("temperature", 2.0),
            "epochs": tr.get("epochs", 200), "lr": tr.get("lr", 1e-3), "amp": tr.get("amp", True),
            "clgeo_weight": d.get("clgeodice_weight", 0.0), "clgeo_r_th": d.get("clgeodice_r_th", 8)}


def dataset_id_and_name(cfg, did=1):
    """Deterministic nnU-Net dataset id + folder name, e.g. (1, 'Dataset001_Coronary').

    Matches the name dca1_to_nnunet.py writes so ARCADE + DCA1 land in the same raw dataset."""
    short = (cfg.get("task", "coronary").split("_")[0] or "coronary").capitalize()
    return did, f"Dataset{did:03d}_{short}"


_RESENC_PLANS = {"ResEncM": "nnUNetResEncUNetMPlans", "ResEncL": "nnUNetResEncUNetLPlans",
                 "ResEncXL": "nnUNetResEncUNetXLPlans"}


def _plans(cfg):
    """Map teacher preset -> nnU-Net plans identifier; None means the default nnUNetPlans."""
    return _RESENC_PLANS.get(cfg.get("teacher", {}).get("preset"))


def nnunet_train_cmd(dataset_id, cfg, fold=0):
    """argv for `nnUNetv2_train <id> <config> <fold> [-p <plans>]` (one fold per call)."""
    t = cfg.get("teacher", {})
    cmd = ["nnUNetv2_train", str(dataset_id), t.get("config", "2d"), str(fold)]
    p = _plans(cfg)
    return cmd + (["-p", p] if p else [])


def nnunet_predict_cmd(dataset_id, in_dir, out_dir, cfg):
    """argv for `nnUNetv2_predict` with --save_probabilities (the teacher cache) across all folds."""
    t = cfg.get("teacher", {})
    cmd = ["nnUNetv2_predict", "-i", in_dir, "-o", out_dir, "-d", str(dataset_id),
           "-c", t.get("config", "2d"), "--save_probabilities",
           "-f", *[str(f) for f in range(t.get("folds", 5))]]
    p = _plans(cfg)
    return cmd + (["-p", p] if p else [])


def qualifies(scores, cfg, teacher_scores=None):
    """True once the student clears BOTH gates. PURE.

    - Dice floor: scores['dice'] >= cfg['target']['dice'] (default 0.75).
    - clDice: (a) optional absolute floor cfg['target'].get('cldice') — skipped when unset; AND
              (b) when teacher_scores is given, the student's clDice must stay within
                  cfg['target'].get('cldice_rel_teacher', 0.03) of the TEACHER's clDice
                  (scores['cldice'] >= teacher_scores['cldice'] - tol). Distillation that keeps
                  Dice but drops connectivity (thin vessels) must NOT qualify.

    Backward compatible: teacher_scores=None -> just the Dice floor + optional absolute clDice
    floor (an absent cfg['target']['cldice'] leaves the old Dice-only behaviour untouched)."""
    tgt = cfg.get("target", {})
    if scores.get("dice", 0.0) < tgt.get("dice", 0.75):
        return False
    cl = scores.get("cldice", 0.0)
    floor = tgt.get("cldice")
    if floor is not None and cl < floor:
        return False
    if teacher_scores is not None:
        tol = tgt.get("cldice_rel_teacher", 0.03)
        if cl < teacher_scores.get("cldice", 0.0) - tol:
            return False
    return True


def split_stems(all_stems, val_frac=0.2):
    """PURE patient-grouped train/val split of image stems -> (train_stems, val_stems).

    Reuses io_utils.split_of (which hashes group_key(stem)) so EVERY frame of one patient/clip
    sequence lands on ONE side — no per-frame leakage — and the result is deterministic across
    runs/processes. Does NOT reimplement the hashing. Returns two sorted, non-overlapping lists;
    an empty input yields ([], [])."""
    from src.data_prep import io_utils as io
    train, val = [], []
    for s in sorted(set(all_stems)):
        (val if io.split_of(s, val_frac) == "val" else train).append(s)
    return train, val


def _scores(student, loader, device=None):
    """Mean Dice/clDice of the (thresholded) student over a loader. Lazy torch."""
    import torch
    from src.eval import metrics
    device = device or next(student.parameters()).device   # else x.to(None) no-op vs a cuda model -> crash
    student.eval(); ds, cs, gs = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            pred = (torch.sigmoid(student(x.to(device))) > 0.5).cpu().numpy()
            for pi, gi in zip(pred, y.numpy()):
                ds.append(metrics.dice(pi[0], gi[0])); cs.append(metrics.cldice(pi[0], gi[0]))
                gs.append(metrics.clgeodice(pi[0], gi[0]))
    student.train()
    # metrics return NaN on empty-GT frames (trivial, no vessel to score); drop them so a single
    # empty frame can't poison the mean into NaN and false-pass the gate (nan<0.75 is False).
    def mean(a):
        v = [x for x in a if x == x]
        return sum(v) / len(v) if v else 0.0
    return {"dice": mean(ds), "cldice": mean(cs), "clgeodice": mean(gs)}


def train(cfg, processed_dir="data/processed/coronary",
          teacher_cache="runs/coronary/teacher_cache", ckpt="runs/coronary/student.pt",
          device=None, run_teacher=True, export_model=True):
    """Full driver: prep -> nnU-Net teacher (CLI) -> distill TinyU-Net -> eval -> optional CoreML.

    Heavy path (torch + nnU-Net CLI) only runs on the GPU box; every heavy import is lazy so this
    module stays importable without them."""
    import torch
    from torch.utils.data import DataLoader
    from src.models.seg_student import TinyUNet
    from src.models.distill import TeacherCacheDataset, distill
    from src.data_prep import io_utils as io, dca1_to_nnunet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg.get("preprocess", {}).get("size", 512)
    did, name = dataset_id_and_name(cfg)
    raw = os.path.join(os.environ.get("nnUNet_raw", "data/nnUNet_raw"), name)
    ds = cfg.get("datasets", {})

    # 1) prep: ARCADE COCO -> (img,msk) pairs + nnU-Net raw, then append DCA1, then dataset.json
    if ds.get("arcade"):
        io.coco_seg_to_pairs(ds["arcade"]["root"], processed_dir, size=size, raw_dir=raw)
    if ds.get("dca1"):
        dca1_to_nnunet.main(cfg)
    n = len(glob.glob(os.path.join(raw, "imagesTr", "*_0000.png")))
    io.write_nnunet_datasetjson(raw, n)

    # 2) nnU-Net teacher: train each fold, then cache per-case probabilities for distillation
    if run_teacher:
        for fold in range(cfg.get("teacher", {}).get("folds", 5)):
            subprocess.run(nnunet_train_cmd(did, cfg, fold=fold), check=True)
        subprocess.run(nnunet_predict_cmd(did, os.path.join(raw, "imagesTr"), teacher_cache, cfg),
                       check=True)

    # 3) patient-grouped held-out split: distill on TRAIN cases, evaluate on UNSEEN VAL cases.
    #    (distilling AND scoring on one loader over ALL cases is eval-on-train -> memorized Dice.)
    tr = cfg.get("train", {})
    all_stems = [os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(processed_dir, "img", "*"))]
    train_stems, val_stems = split_stems(all_stems, val_frac=tr.get("val_frac", 0.2))
    batch = tr.get("batch", 8)
    train_loader = DataLoader(
        TeacherCacheDataset(processed_dir, teacher_cache, size=size, stems=train_stems),
        batch_size=batch, shuffle=True)
    if val_stems:
        eval_loader = DataLoader(
            TeacherCacheDataset(processed_dir, teacher_cache, size=size, stems=val_stems),
            batch_size=batch, shuffle=False)
    else:
        print("WARNING: empty val split (tiny dataset) — scoring on the TRAIN loader; reported "
              "Dice/clDice are eval-on-train and NOT a held-out estimate.")
        eval_loader = train_loader

    # 4) distill student against cached teacher logits, evaluating on the held-out VAL loader
    student = TinyUNet(**student_kwargs(cfg))
    student = distill(student, train_loader, eval_fn=lambda m: str(_scores(m, eval_loader, device)),
                      device=device, ckpt=ckpt, **distill_kwargs(cfg))

    # 5) final gate on the held-out split + optional CoreML export (Mac only)
    scores = _scores(student, eval_loader, device)
    print("scores:", scores, "| qualifies:", qualifies(scores, cfg))
    if export_model and cfg.get("export", {}).get("coreml") and sys.platform == "darwin":
        from src.export.to_coreml import export
        export(ckpt, **student_kwargs(cfg))
        # INT8/CoreML clDice re-check must never be silently skipped: palettization can hold Dice
        # while clDice (thin-vessel connectivity) collapses. Re-score on the SAME held-out split.
        _int8_cldice_recheck(cfg, ckpt, processed_dir, val_stems, size)
    return ckpt


def _int8_cldice_recheck(cfg, ckpt, processed_dir, val_stems, size):
    """Best-effort: re-score the exported CoreML model's clDice-drop on the held-out val split via
    the existing src/export/coreml_validate.py gate. Materializes a val/ dir of ONLY the held-out
    stems so the gate scores UNSEEN cases. Guarded: on any gap/error, prints an explicit
    '[TODO] INT8 clDice re-check not run' rather than skipping it silently."""
    import shutil, tempfile
    from types import SimpleNamespace
    try:
        from src.export import coreml_validate
        coreml_path = ckpt.replace(".pt", ".mlpackage")
        if not val_stems or not os.path.isdir(coreml_path):
            print("[TODO] INT8 clDice re-check not run "
                  f"(val_stems={len(val_stems or [])}, coreml_present={os.path.isdir(coreml_path)})")
            return
        tmp = tempfile.mkdtemp(prefix="coreml_val_")
        vi, vm = os.path.join(tmp, "img"), os.path.join(tmp, "msk")
        os.makedirs(vi); os.makedirs(vm)
        n = 0
        for s in val_stems:
            si = os.path.join(processed_dir, "img", s + ".png")
            sm = os.path.join(processed_dir, "msk", s + ".png")
            if os.path.exists(si) and os.path.exists(sm):
                shutil.copyfile(si, os.path.join(vi, s + ".png"))
                shutil.copyfile(sm, os.path.join(vm, s + ".png"))
                n += 1
        if not n:
            print("[TODO] INT8 clDice re-check not run (no val img/msk pairs found on disk)")
            return
        sk = student_kwargs(cfg)
        a = SimpleNamespace(coreml=coreml_path, weights=ckpt, images=vi, masks=vm, size=size,
                            base=sk["base"], depth=sk["depth"], limit=n,
                            gate=cfg.get("target", {}).get("cldice_rel_teacher", 0.03))
        print(f"[INT8 clDice re-check] gate PASS={coreml_validate.main(a)}")
    except Exception as e:
        print(f"[TODO] INT8 clDice re-check not run (error: {e})")


def main(cfg):
    return train(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
