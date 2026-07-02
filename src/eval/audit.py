"""Audit trail: log (timestamp, model version, input hash, prediction summary) per inference.

Nearly free to add now, expensive to retrofit for incident review / regulatory traceability.
Wire `record(...)` into the inference loop from the very first demo.
"""
import hashlib, json, os, time
import numpy as np


def input_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def record(model_version, input_arr, summary, path="runs/audit.jsonl"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"ts": time.time(), "model": model_version,
           "input_sha": input_hash(np.asarray(input_arr)), "summary": summary}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    print(record("coronary-student-v0", np.zeros((512, 512), np.float32),
                 {"dice_self": None, "deferred": False}, path="runs/audit.jsonl"))
