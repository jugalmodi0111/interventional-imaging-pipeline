"""Frozen foundation backbone + lightweight trained head (Dialygo B4).

The backbone is a feature extractor and NOTHING trains inside it -- sample efficiency on a small
institutional cohort is the fixed requirement, and a frozen backbone is how the design meets it.
Real backbones (dinov2_vitb14 default; dinov3/rad-dino/biomedclip bake-off candidates per
configs/avf_fistulography.yaml) come from timm, imported lazily so this module -- and every test --
works with timm absent. 'test-tiny' is a seeded, deterministic conv backbone for offline tests:
NEVER use it for a real run.
"""
import torch
import torch.nn as nn


def make_backbone(name, imgsz=224):
    """-> (module, feat_dim). Module: [B,1,H,W] float in [0,1] -> [B, feat_dim] features."""
    if name == "test-tiny":
        gen = torch.Generator().manual_seed(1234)
        conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        with torch.no_grad():
            conv.weight.copy_(torch.rand(conv.weight.shape, generator=gen) - 0.5)
            conv.bias.zero_()
        backbone = nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        return backbone, 32
    import timm                                    # lazy: only real runs pay for this
    model = timm.create_model(name, pretrained=True, num_classes=0, in_chans=1)
    feat_dim = model.num_features
    return model, feat_dim


class FrozenBackboneClassifier(nn.Module):
    def __init__(self, backbone_name, imgsz=224):
        super().__init__()
        self.backbone_name = backbone_name
        self.imgsz = imgsz
        self.backbone, feat_dim = make_backbone(backbone_name, imgsz)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.head = nn.Linear(feat_dim, 1)

    def trainable_parameters(self):
        return self.head.parameters()

    def train(self, mode=True):
        """Head follows train/eval; the frozen backbone stays in eval so norm layers never update."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.head(feats).squeeze(-1)
