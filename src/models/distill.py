"""Knowledge distillation: nnU-Net teacher -> TinyU-Net student."""
import glob, os
import numpy as np
import torch, torch.nn.functional as Fn
from torch.utils.data import Dataset


def kd_loss(student_logits, teacher_logits, target, alpha=0.5, T=2.0):
    """Soft KD + hard supervision. Binary (1 channel) uses sigmoid-KD, not channel-softmax
    (softmax over a size-1 channel is degenerate -> ~zero soft loss)."""
    if student_logits.shape[1] == 1:                      # binary vessel seg
        pt = torch.sigmoid(teacher_logits / T)
        ce = -pt * Fn.logsigmoid(student_logits / T) \
             - (1 - pt) * Fn.logsigmoid(-student_logits / T)
        soft = T * T * ce.mean()
        hard = Fn.binary_cross_entropy_with_logits(student_logits, target.float())
    else:                                                 # multi-class
        soft = Fn.kl_div(Fn.log_softmax(student_logits / T, 1),
                         Fn.softmax(teacher_logits / T, 1),
                         reduction="batchmean") * (T * T)
        hard = Fn.cross_entropy(student_logits, target.long())
    return alpha * soft + (1 - alpha) * hard


class TeacherCacheDataset(Dataset):
    """Yields (image[1,H,W], mask[1,H,W], teacher_logits[1,H,W]).

    Expects standardized PNGs under {processed_dir}/{img,msk} (matching filenames) and
    per-case teacher probabilities in {teacher_cache} as .npy/.npz keyed by the same stem
    (e.g. from nnUNetv2_predict --save_probabilities). Adjust the stem mapping to match
    whatever your prep wrote.
    """
    def __init__(self, processed_dir, teacher_cache, size=512):
        import cv2
        self.cv2, self.size = cv2, size
        self.imgs = sorted(glob.glob(os.path.join(processed_dir, "img", "*")))
        self.msk_dir = os.path.join(processed_dir, "msk")
        self.cache = teacher_cache

    def __len__(self):
        return len(self.imgs)

    def _teacher_logit(self, stem, H, W):
        for ext in (".npy", ".npz"):
            p = os.path.join(self.cache, stem + ext)
            if os.path.exists(p):
                arr = np.asarray(np.load(p)["probabilities"] if ext == ".npz" else np.load(p))
                # nnU-Net 2D saves probs channel-first, often (C, 1, H, W). Take the foreground
                # channel then squeeze to a STRICT 2D (H,W) BEFORE resize (else cv2 misreads the
                # extra axis as channels and explodes the tensor -> 32 GB OOM in kd_loss).
                fg = arr[1] if arr.shape[0] >= 2 else arr[0]
                fg = np.squeeze(fg).astype(np.float32)
                if fg.ndim != 2:
                    fg = fg.reshape(fg.shape[-2], fg.shape[-1])
                fg = np.clip(fg, 1e-6, 1 - 1e-6)
                logit = np.log(fg / (1 - fg))             # prob -> logit for sigmoid-KD
                return self.cv2.resize(logit, (W, H))
        raise FileNotFoundError(f"teacher logits for {stem!r} not in {self.cache}")

    def __getitem__(self, i):
        ip = self.imgs[i]; stem = os.path.splitext(os.path.basename(ip))[0]
        im = self.cv2.resize(self.cv2.imread(ip, 0), (self.size, self.size)).astype(np.float32) / 255.0
        gt = self.cv2.resize(self.cv2.imread(os.path.join(self.msk_dir, os.path.basename(ip)), 0),
                             (self.size, self.size))
        tl = self._teacher_logit(stem, self.size, self.size)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
        return t(im), t((gt > 127).astype(np.float32)), t(tl)


def distill(student, loader, epochs=200, lr=1e-3, alpha=0.5, T=2.0, device=None,
            amp=True, eval_fn=None, ckpt="runs/coronary/student.pt"):
    """Train student against cached teacher logits. Saves a STATE_DICT (portable Mac handoff)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device == "cuda"
    student.to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for ep in range(epochs):
        tot = 0.0
        for x, y, tl in loader:
            x, y, tl = x.to(device), y.to(device), tl.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = kd_loss(student(x), tl, y, alpha, T)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        msg = f"epoch {ep + 1}/{epochs}  kd_loss {tot / max(1, len(loader)):.4f}"
        if eval_fn and (ep + 1) % 10 == 0:
            msg += "  |  " + eval_fn(student)
        print(msg)
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save(student.state_dict(), ckpt)
    print("saved", ckpt)
    return student
