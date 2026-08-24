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
