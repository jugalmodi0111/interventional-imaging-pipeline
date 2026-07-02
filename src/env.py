"""Environment detection + standard paths so notebooks stay THIN and portable.

All heavy logic lives in importable `src/*` modules. A notebook is just: `env.setup()` then a few
calls into `src`. Works on Colab (Drive), Kaggle (/kaggle/working + /kaggle/input), or local.
"""
import os


def detect():
    if os.path.exists("/kaggle"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "colab"
    return "local"


def device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def setup(project="intv-img", mount_drive=True):
    """Mount persistent storage, point nnU-Net env vars at it, return standard paths + device."""
    platform = detect()
    if platform == "colab":
        if mount_drive:
            from google.colab import drive
            drive.mount("/content/drive")
        root = f"/content/drive/MyDrive/{project}"
        data_raw = f"{root}/data/raw"
    elif platform == "kaggle":
        root = f"/kaggle/working/{project}"
        data_raw = "/kaggle/input"                 # attach Kaggle Datasets; read-only
    else:
        root = os.path.abspath(project if os.path.isdir(project) else ".")
        data_raw = os.path.join(root, "data/raw")

    os.makedirs(root, exist_ok=True)
    runs = os.path.join(root, "runs"); os.makedirs(runs, exist_ok=True)
    for k, sub in [("nnUNet_raw", "nnUNet_raw"),
                   ("nnUNet_preprocessed", "nnUNet_preprocessed"),
                   ("nnUNet_results", "nnUNet_results")]:
        d = os.path.join(root, sub); os.makedirs(d, exist_ok=True); os.environ[k] = d

    info = {"platform": platform, "device": device(), "root": root,
            "runs": runs, "data_raw": data_raw,
            "teacher_cache": os.path.join(root, "teacher_cache")}
    os.makedirs(info["teacher_cache"], exist_ok=True)
    print(f"[env] {platform} | device={info['device']} | root={root}")
    if info["device"] == "cpu":
        print("[env] WARNING: no GPU. Training/eval will be very slow — use Colab/Kaggle GPU runtime.")
    return info


if __name__ == "__main__":
    print(setup(mount_drive=False))
