"""TinyU-Net: ~2-3M param edge student for vessel segmentation."""
import torch, torch.nn as nn

def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, base=16, depth=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]
        self.enc = nn.ModuleList(); prev = in_ch
        for c in chs:
            self.enc.append(nn.Sequential(cbr(prev, c), cbr(c, c))); prev = c
        self.pool = nn.MaxPool2d(2)
        self.dec = nn.ModuleList(); self.up = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.dec.append(nn.Sequential(cbr(prev, c), cbr(c, c))); prev = c
        self.head = nn.Conv2d(prev, n_classes, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x); skips.append(x)
            if i < len(self.enc) - 1: x = self.pool(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = u(x); x = d(torch.cat([x, s], 1))
        return self.head(x)

def load_student(weights, in_ch=1, n_classes=1, base=16, depth=4, device="cpu"):
    """Load a student from a state_dict (preferred, portable handoff) or a pickled module."""
    obj = torch.load(weights, map_location=device)
    if isinstance(obj, dict) and not hasattr(obj, "forward"):
        m = TinyUNet(in_ch, n_classes, base, depth)
        m.load_state_dict(obj)
    else:
        m = obj
    return m.eval()


if __name__ == "__main__":
    m = TinyUNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"TinyUNet params: {n/1e6:.2f}M")
    print("out:", m(torch.randn(1, 1, 512, 512)).shape)
