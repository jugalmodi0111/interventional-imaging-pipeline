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
            "epochs": tr.get("epochs", 200), "lr": tr.get("lr", 1e-3), "amp": tr.get("amp", True)}


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


def qualifies(scores, cfg):
    """True once the student clears the target Dice gate (cfg['target']['dice'])."""
    return scores.get("dice", 0.0) >= cfg.get("target", {}).get("dice", 0.75)


def _scores(student, loader, device=None):
    """Mean Dice/clDice of the (thresholded) student over a loader. Lazy torch."""
    import torch
    from src.eval import metrics
    device = device or next(student.parameters()).device   # else x.to(None) no-op vs a cuda model -> crash
    student.eval(); ds, cs = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            pred = (torch.sigmoid(student(x.to(device))) > 0.5).cpu().numpy()
            for pi, gi in zip(pred, y.numpy()):
                ds.append(metrics.dice(pi[0], gi[0])); cs.append(metrics.cldice(pi[0], gi[0]))
    student.train()
    mean = lambda a: sum(a) / max(1, len(a))
    return {"dice": mean(ds), "cldice": mean(cs)}


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

    # 3) distill student against cached teacher logits
    loader = DataLoader(TeacherCacheDataset(processed_dir, teacher_cache, size=size),
                        batch_size=cfg.get("train", {}).get("batch", 8), shuffle=True)
    student = TinyUNet(**student_kwargs(cfg))
    student = distill(student, loader, eval_fn=lambda m: str(_scores(m, loader, device)),
                      device=device, ckpt=ckpt, **distill_kwargs(cfg))

    # 4) final gate + optional CoreML export (Mac only)
    scores = _scores(student, loader, device)
    print("scores:", scores, "| qualifies:", qualifies(scores, cfg))
    if export_model and cfg.get("export", {}).get("coreml") and sys.platform == "darwin":
        from src.export.to_coreml import export
        export(ckpt, **student_kwargs(cfg))
    return ckpt


def main(cfg):
    return train(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
