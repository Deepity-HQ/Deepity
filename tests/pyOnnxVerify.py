"""Deepity -> ONNX -> onnxruntime: outputs of the exported forward path match predict(x, 0).

  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run --with onnx --with onnxruntime python tests/pyOnnxVerify.py
Skipped (exit 0) when onnx or onnxruntime is not installed.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:
    print("SKIPPED: onnx/onnxruntime not installed")
    sys.exit(0)

failures = 0


def check(ok, what):
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


rng = np.random.default_rng(0)
x = rng.standard_normal((8, 12)).astype(np.float32)
with tempfile.TemporaryDirectory() as tmp:
    for cls in (dp.FullPCN, dp.DKPPCN):
        for act in (dp.ReLU, dp.TanH, dp.Sigmoid, dp.GeLU):
            net = cls(dp.Linear(12, 10), act(), dp.Linear(10, 6), act(), dp.Linear(6, 4), batch_size=8).configure(learning_rate=1e-3)
            weights = {}
            for i, (o, n) in enumerate([(10, 12), (6, 10), (4, 6)]):
                weights[f"layers.{i}.weight"] = rng.standard_normal((o, n)).astype(np.float32) * 0.5
                weights[f"layers.{i}.bias"] = rng.standard_normal(o).astype(np.float32)
            dp.import_weights(net, weights)
            path = os.path.join(tmp, "m.onnx")
            dp.export_onnx(net, path)
            session = ort.InferenceSession(path)
            got = session.run(None, {"x": x})[0]
            want = net.predict(x, 0)
            check(np.allclose(got, want, atol=1e-5), f"{cls.__name__} {act.__name__}: onnxruntime matches predict(x, 0)")
            check(session.run(None, {"x": x[:3]})[0].shape == (3, 4), f"{cls.__name__} {act.__name__}: batch size is dynamic")
print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
