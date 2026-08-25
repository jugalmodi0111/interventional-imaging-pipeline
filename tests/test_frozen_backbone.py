"""Frozen-backbone classifier (B4): the backbone never trains, only the linear head does.
All tests use the 'test-tiny' backbone -- no timm import, no network, CPU-only."""
import pytest

torch = pytest.importorskip("torch")

from src.models.frozen_backbone import FrozenBackboneClassifier, make_backbone


def test_test_tiny_backbone_shape_and_determinism():
    b1, d1 = make_backbone("test-tiny", imgsz=32)
    b2, d2 = make_backbone("test-tiny", imgsz=32)
    x = torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(0))
    assert d1 == d2
    assert torch.equal(b1(x), b2(x))          # seeded init: same weights, same features
    assert b1(x).shape == (2, d1)


def test_backbone_is_frozen_and_head_is_trainable():
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    assert all(not p.requires_grad for p in m.backbone.parameters())
    trainable = list(m.trainable_parameters())
    assert trainable and all(p.requires_grad for p in trainable)
    assert {id(p) for p in trainable} == {id(p) for p in m.head.parameters()}


def test_forward_returns_one_logit_per_sample():
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    out = m(torch.rand(3, 1, 32, 32))
    assert out.shape == (3,)


def test_head_learns_while_backbone_stays_fixed():
    m = FrozenBackboneClassifier("test-tiny", imgsz=32)
    before = [p.clone() for p in m.backbone.parameters()]
    x, y = torch.rand(8, 1, 32, 32), torch.tensor([0., 1.] * 4)
    opt = torch.optim.SGD(m.trainable_parameters(), lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), y)
        loss.backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, m.backbone.parameters()))


# --- real-backbone plumbing: name resolution + resolution handling ------------------------------
# These pin the two defects found 2026-08-25 when the bake-off first tried a REAL backbone. Both
# lived in shipped code and neither was reachable from the test-tiny path, so the whole suite was
# green while `configs/avf_fistulography.yaml` named a model timm cannot build.

class _FakeTimm:
    """Stands in for timm so these tests need no network and no weights."""

    def __init__(self, accepts_img_size=True):
        self.accepts_img_size = accepts_img_size
        self.calls = []

    def create_model(self, name, **kw):
        self.calls.append(kw)
        if "img_size" in kw and not self.accepts_img_size:
            raise TypeError("ResNet.__init__() got an unexpected keyword argument 'img_size'")

        class M:
            num_features = 111
        return M()


def _with_fake_timm(monkeypatch, fake):
    import sys
    monkeypatch.setitem(sys.modules, "timm", fake)


def test_timm_backbone_is_built_at_the_requested_resolution(monkeypatch):
    """DINOv2 in timm defaults to 518 px and ASSERTS on a 224 input. make_backbone took an imgsz and
    silently dropped it, so the config's declared imgsz 224 could never have worked."""
    from src.models import frozen_backbone as fb
    fake = _FakeTimm(accepts_img_size=True)
    _with_fake_timm(monkeypatch, fake)
    _, dim = fb.make_backbone("vit_base_patch14_dinov2.lvd142m", imgsz=224)
    assert dim == 111
    assert fake.calls[-1]["img_size"] == 224
    assert fake.calls[-1]["in_chans"] == 1 and fake.calls[-1]["num_classes"] == 0


def test_cnn_backbones_that_reject_img_size_still_build(monkeypatch):
    """ResNet/ConvNeXt take no img_size (they are resolution-agnostic) and raise TypeError on it.
    That must degrade to a plain build, not kill the bake-off."""
    from src.models import frozen_backbone as fb
    fake = _FakeTimm(accepts_img_size=False)
    _with_fake_timm(monkeypatch, fake)
    _, dim = fb.make_backbone("resnet50.a1_in1k", imgsz=224)
    assert dim == 111
    assert "img_size" in fake.calls[0] and "img_size" not in fake.calls[-1]


def test_configured_backbone_name_exists_in_timm():
    """configs/avf_fistulography.yaml shipped `dinov2_vitb14` -- a torch.hub name, which timm does
    not know (`RuntimeError: Unknown model`). Registry lookup only; downloads nothing."""
    timm = pytest.importorskip("timm")
    yaml = pytest.importorskip("yaml")
    with open("configs/avf_fistulography.yaml") as f:
        name = yaml.safe_load(f)["model"]["backbone"]
    base = name.split(".")[0]          # list_models() reports architectures without pretrained tags
    assert timm.is_model(base), (
        f"configs/avf_fistulography.yaml names backbone {name!r}, which timm cannot build "
        f"(architecture {base!r} is not in the registry).")
    if "." in name:                    # a pretrained tag was given -- it must be a real one
        assert name in set(timm.list_models(pretrained=True)), (
            f"{name!r} names architecture {base!r} with an unknown pretrained tag. "
            f"Valid: {timm.list_models(f'{base}*', pretrained=True)[:4]}")
