"""Edge benchmark: params, latency, fps, model size for an ONNX/torch model on THIS device."""
import argparse, os, time, numpy as np

def bench_onnx(path, shape=(1, 1, 512, 512), runs=50, warmup=10):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(*shape).astype(np.float32)
    for _ in range(warmup): sess.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(runs): sess.run(None, {name: x})
    ms = (time.perf_counter() - t) / runs * 1000
    return {"model": os.path.basename(path), "size_mb": round(os.path.getsize(path)/1e6, 2),
            "latency_ms": round(ms, 2), "fps": round(1000/ms, 1)}

def bench_coreml(path, shape=(1, 1, 512, 512), runs=50, warmup=10):
    """Benchmark a CoreML .mlpackage on macOS (uses compute_units baked into the model)."""
    import coremltools as ct
    m = ct.models.MLModel(path)
    name = m.get_spec().description.input[0].name
    x = np.random.randn(*shape).astype(np.float32)
    for _ in range(warmup): m.predict({name: x})
    t = time.perf_counter()
    for _ in range(runs): m.predict({name: x})
    ms = (time.perf_counter() - t) / runs * 1000
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(path) for f in fs) / 1e6   # .mlpackage is a dir
    return {"model": os.path.basename(path), "size_mb": round(size, 2),
            "latency_ms": round(ms, 2), "fps": round(1000/ms, 1)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    a = ap.parse_args()
    if a.model.endswith(".onnx"):
        print(bench_onnx(a.model))
    elif a.model.endswith(".mlpackage"):
        print(bench_coreml(a.model))
    else:
        print("TODO: torch path -> count params + cuda/cpu timing")
