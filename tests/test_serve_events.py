"""Test the observe-only pub/sub event layer: bus semantics, sinks, orchestrator wiring, /events.

The bus mirrors the pipeline -- it never carries it. Nothing in these tests lets a subscriber
change a verdict, and the exception-isolation test proves a crashing subscriber cannot break the
clinical path. No torch import at module scope, matching the serve layer's import-safe convention.
"""
import json

import numpy as np
import pytest

from src.serve.events import EventBus, JsonlSink, RingBuffer
from src.serve.registry import TaskEntry
from src.serve.router import ModalityDecision
from src.serve.orchestrator import DiagnosticOrchestrator


class FakeRouter:
    def __init__(self, decision):
        self.d = decision

    def classify(self, frame):
        return self.d


def _reg(floor_ok=True):
    return {"coronary_angiography": TaskEntry(
        "coronary_angiography", "det", "best.pt", "Coronary angiography",
        "coronary_stenosis", "Possible coronary artery stenosis", floor_ok=floor_ok)}


def _det_factory(boxes):
    def factory(entry):
        return lambda frame: {"boxes": boxes, "top_conf": max([b[4] for b in boxes], default=0.0),
                              "deferred": False}
    return factory


FRAME = np.zeros((64, 64), dtype=np.uint8)
CONFIDENT = ModalityDecision("coronary_angiography", None, True, 0.95, False, "confident")


# --- bus semantics -------------------------------------------------------------------------------

def test_publish_delivers_to_subscribers_in_subscribe_order():
    bus = EventBus()
    got = []
    bus.subscribe("frame.received", lambda e: got.append(("a", e["topic"])))
    bus.subscribe("frame.received", lambda e: got.append(("b", e["topic"])))
    bus.publish("frame.received", input_hash="x")
    assert got == [("a", "frame.received"), ("b", "frame.received")]


def test_wildcard_pattern_matches_topic_family_only():
    bus = EventBus()
    got = []
    bus.subscribe("model.*", lambda e: got.append(e["topic"]))
    bus.publish("model.inferred", n_boxes=1)
    bus.publish("router.decided", modality="m")
    assert got == ["model.inferred"]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    got = []
    off = bus.subscribe("*", lambda e: got.append(e["topic"]))
    bus.publish("one")
    off()
    bus.publish("two")
    assert [t for t in got if not t.startswith("bus.")] == ["one"]


def test_crashing_subscriber_is_isolated_and_counted():
    bus = EventBus()
    got = []

    def boom(e):
        raise RuntimeError("subscriber bug")

    bus.subscribe("*", boom)
    bus.subscribe("*", lambda e: got.append(e["topic"]))
    bus.publish("frame.received", input_hash="x")
    assert "frame.received" in got          # second subscriber still ran
    assert bus.errors >= 1                   # the crash was counted, not raised


def test_subscribe_itself_is_published_as_an_event():
    bus = EventBus()
    got = []
    bus.subscribe("bus.subscribed", lambda e: got.append(e["data"]["pattern"]))
    bus.subscribe("model.*", lambda e: None)
    assert "model.*" in got


def test_events_carry_monotonic_seq_and_timestamp():
    bus = EventBus()
    got = []
    bus.subscribe("*", lambda e: got.append(e))
    bus.publish("one")
    bus.publish("two")
    seqs = [e["seq"] for e in got]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert all(e["ts"] for e in got)


# --- sinks ---------------------------------------------------------------------------------------

def test_ring_buffer_keeps_only_last_n():
    bus = EventBus()
    ring = RingBuffer(maxlen=2)
    bus.subscribe("*", ring)
    for topic in ("one", "two", "three"):
        bus.publish(topic)
    topics = [e["topic"] for e in ring.snapshot() if not e["topic"].startswith("bus.")]
    assert topics == ["two", "three"]


def test_jsonl_sink_writes_one_parseable_line_per_event(tmp_path):
    bus = EventBus()
    bus.subscribe("*", JsonlSink(tmp_path / "events.jsonl"))
    bus.publish("frame.received", input_hash="abc")
    bus.publish("verdict.emitted", deferred=True)
    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["topic"] for r in rows] == ["frame.received", "verdict.emitted"]
    assert rows[0]["data"] == {"input_hash": "abc"}


# --- orchestrator wiring -------------------------------------------------------------------------

def _collect(bus):
    got = []
    bus.subscribe("*", lambda e: got.append(e))
    return got


def test_confident_frame_publishes_the_full_topic_sequence(monkeypatch):
    monkeypatch.setattr("src.serve.orchestrator.record", lambda *a, **k: None)
    bus = EventBus()
    got = _collect(bus)
    orch = DiagnosticOrchestrator(FakeRouter(CONFIDENT), _reg(), _det_factory([(1, 2, 3, 4, 0.9)]),
                                  bus=bus)
    orch.analyze_frame(FRAME)
    topics = [e["topic"] for e in got if not e["topic"].startswith("bus.")]
    assert topics == ["frame.received", "router.decided", "model.inferred", "verdict.emitted"]


def test_router_unavailable_is_published(monkeypatch):
    from src.serve.orchestrator import RouterUnavailable

    class DeadRouter:
        def classify(self, frame):
            raise RouterUnavailable("no weights")

    bus = EventBus()
    got = _collect(bus)
    orch = DiagnosticOrchestrator(DeadRouter(), _reg(), _det_factory([]), bus=bus)
    report = orch.analyze_frame(FRAME)
    topics = [e["topic"] for e in got if not e["topic"].startswith("bus.")]
    assert topics == ["frame.received", "router.unavailable", "verdict.emitted"]
    assert report.defer_reason == "router-unavailable"


def test_model_unavailable_is_published():
    from src.serve.orchestrator import ModelUnavailable

    def dead_factory(entry):
        def _dead(frame):
            raise ModelUnavailable("weights missing")
        return _dead

    bus = EventBus()
    got = _collect(bus)
    orch = DiagnosticOrchestrator(FakeRouter(CONFIDENT), _reg(), dead_factory, bus=bus)
    orch.analyze_frame(FRAME)
    topics = [e["topic"] for e in got if not e["topic"].startswith("bus.")]
    assert topics == ["frame.received", "router.decided", "model.unavailable", "verdict.emitted"]


def test_orchestrator_without_bus_still_works(monkeypatch):
    monkeypatch.setattr("src.serve.orchestrator.record", lambda *a, **k: None)
    orch = DiagnosticOrchestrator(FakeRouter(CONFIDENT), _reg(), _det_factory([(1, 2, 3, 4, 0.9)]))
    report = orch.analyze_frame(FRAME)
    assert report.findings


def test_event_payloads_never_carry_pixels_or_arrays(monkeypatch):
    monkeypatch.setattr("src.serve.orchestrator.record", lambda *a, **k: None)
    bus = EventBus()
    got = _collect(bus)
    orch = DiagnosticOrchestrator(FakeRouter(CONFIDENT), _reg(), _det_factory([(1, 2, 3, 4, 0.9)]),
                                  bus=bus)
    orch.analyze_frame(FRAME)
    for e in got:
        for v in e["data"].values():
            assert not isinstance(v, np.ndarray)
            assert not isinstance(v, (bytes, bytearray))
    frame_evt = next(e for e in got if e["topic"] == "frame.received")
    # Same 16-hex digest audit.jsonl uses, so events and audit rows cross-correlate by hash.
    assert len(frame_evt["data"]["input_hash"]) == 16


def test_build_orchestrator_attaches_bus_and_publishes_registry_loaded():
    from src.serve.orchestrator import build_orchestrator
    orch = build_orchestrator("configs/orchestrator.yaml")
    assert orch.bus is not None
    loaded = [e for e in orch.bus.ring.snapshot() if e["topic"] == "registry.loaded"]
    assert loaded and all("modality" in e["data"] and "floor_ok" in e["data"] for e in loaded)


# --- /events endpoint ----------------------------------------------------------------------------

def test_events_endpoint_streams_ring_replay_and_terminates():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.serve import app as app_module

    bus = EventBus()
    bus.publish("frame.received", input_hash="a" * 64)
    bus.publish("verdict.emitted", deferred=True, defer_reason="router-unavailable")

    class FakeOrch:
        pass

    fake = FakeOrch()
    fake.bus = bus
    old = app_module._orch
    app_module._orch = fake
    try:
        client = TestClient(app_module.app)
        with client.stream("GET", "/events?replay=10&max_events=2&timeout=2") as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            body = "".join(chunk for chunk in r.iter_text())
    finally:
        app_module._orch = old
    payloads = [json.loads(line[len("data: "):])
                for line in body.splitlines() if line.startswith("data: ")]
    assert [p["topic"] for p in payloads][:2] == ["frame.received", "verdict.emitted"]
