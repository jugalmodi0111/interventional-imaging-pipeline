"""Observe-only pub/sub mirror of the serve pipeline.

The bus never carries control flow -- components still call each other directly and publish a
copy of what happened (`router.decided`, `model.inferred`, `verdict.emitted`, ...). Registrations
are themselves observable: every `subscribe()` publishes a `bus.subscribed` event. A subscriber
that raises is counted and skipped, never re-raised, so a broken observer cannot alter or abort a
clinical verdict by construction.

PHI rule: event payloads carry hashes, ids, counts and reasons -- never pixel data, patient
identifiers, or raw drive paths. The frame digest is `src.eval.audit.input_hash`, the same 16-hex
value written to runs/audit.jsonl, so the two logs cross-correlate row-for-row.
"""
import fnmatch
import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


class RingBuffer:
    """Keeps the last `maxlen` events in memory; backs the /events replay."""

    def __init__(self, maxlen=1000):
        self._events = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def __call__(self, event):
        with self._lock:
            self._events.append(event)

    def snapshot(self):
        with self._lock:
            return list(self._events)


class JsonlSink:
    """Appends every event as one JSON line -- sibling convention to runs/audit.jsonl."""

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")


class EventBus:
    """Synchronous, ordered, in-process pub/sub with fnmatch topic patterns ("model.*", "*").

    Every published event lands in `self.ring` regardless of subscribers, so late observers
    (the /events endpoint) can replay recent history. `errors` counts swallowed subscriber
    exceptions -- a nonzero value means an observer is broken, never that a verdict was affected.
    """

    def __init__(self, ring_maxlen=1000):
        self._subs = []
        self._lock = threading.Lock()
        self._seq = 0
        self.errors = 0
        self.ring = RingBuffer(ring_maxlen)

    def subscribe(self, pattern, handler):
        """Register `handler` for topics matching `pattern`. Returns an unsubscribe callable.
        The registration itself is published as `bus.subscribed` -- registrations are events too.
        Announced BEFORE attaching, so a subscriber observes every registration except its own."""
        self.publish("bus.subscribed", pattern=pattern,
                     handler=getattr(handler, "__name__", type(handler).__name__))
        entry = (pattern, handler)
        with self._lock:
            self._subs.append(entry)

        def unsubscribe():
            with self._lock:
                if entry in self._subs:
                    self._subs.remove(entry)
        return unsubscribe

    def publish(self, topic, **data):
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq,
                     "ts": datetime.now(timezone.utc).isoformat(),
                     "topic": topic, "data": data}
            subs = list(self._subs)
        self.ring(event)
        for pattern, handler in subs:
            if fnmatch.fnmatchcase(topic, pattern):
                try:
                    handler(event)
                except Exception:
                    self.errors += 1          # observer bug: counted, never propagated
        return event
