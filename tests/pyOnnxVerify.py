"""Deepity -> ONNX -> onnxruntime: outputs of the exported forward path match predict(x, 0) (dense) or predict_with_projection(x, 0) (convolutional).

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
    import onnxruntime as ort  # type: ignore[import-not-found]
except ImportError:
    print("SKIPPED: onnx/onnxruntime not installed")
    sys.exit(0)

failures = 0


def check(ok, what):
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


def set_weights(net, weights, tmp):
    """Loads a dict of arrays named layers.i.weight/bias into a configured network."""
    path = os.path.join(tmp, "weights.safetensors")
    dp.write_tensors(path, weights)
    net.load_weights(path, False)


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
            set_weights(net, weights, tmp)
            path = os.path.join(tmp, "m.onnx")
            dp.export_onnx(net, path)
            session = ort.InferenceSession(path)
            got = session.run(None, {"x": x})[0]
            want = net.predict(x, 0)
            check(np.allclose(got, want, atol=1e-5), f"{cls.__name__} {act.__name__}: onnxruntime matches predict(x, 0)")
            check(session.run(None, {"x": x[:3]})[0].shape == (3, 4), f"{cls.__name__} {act.__name__}: batch size is dynamic")

    # GaussSeidelPCN has no separate terminal layer: its output is the previous layer's prediction, so the
    # exported graph ends one layer earlier and matches predict at any step count.
    for act in (dp.ReLU, dp.TanH, dp.Sigmoid, dp.GeLU):
        net = dp.GaussSeidelPCN(dp.Linear(12, 10), act(), dp.Linear(10, 6), act(), dp.Linear(6, 4), batch_size=8)
        net.configure(learning_rate=1e-3, inference_rate=0.1)
        weights = {}
        for i, (o, k) in enumerate([(10, 12), (6, 10), (4, 6)]):
            weights[f"layers.{i}.weight"] = rng.standard_normal((o, k)).astype(np.float32) * 0.5
            weights[f"layers.{i}.bias"] = rng.standard_normal(o).astype(np.float32) * 0.5
        set_weights(net, weights, tmp)
        path = os.path.join(tmp, "g.onnx")
        dp.export_onnx(net, path)
        session = ort.InferenceSession(path)
        got = session.run(None, {"x": x})[0]
        want = np.asarray(net.predict(x, 5)).reshape(got.shape)
        check(np.allclose(got, want, atol=1e-5), f"GaussSeidelPCN {act.__name__}: onnxruntime matches predict")
    for cls, kwargs in ((dp.SimpleConvolutionalPCN, {}), (dp.ConvolutionalPCN, {})):
        for act in (dp.ReLU, dp.TanH):
            net = cls(
                dp.Convolution(4, 3, 3, 1, 1, 1, 1), act(), dp.Convolution(6, 4, 4, 2, 2, 1, 1), act(),
                dp.Convolution(10, 4, 4), dp.Convolution(0, 1, 1), input_shape=(2, 8, 8), batch_size=8,
            )
            net.configure(learning_rate=1e-3)
            xc = rng.standard_normal((8, 2 * 8 * 8)).astype(np.float32)
            path = os.path.join(tmp, "c.onnx")
            dp.export_onnx(net, path)
            session = ort.InferenceSession(path)
            got = session.run(None, {"x": xc})[0]
            # predict(x, 0) leaves a conv network's output at zero; the projection pass is its forward path.
            want = np.asarray(net.predict_with_projection(xc, 0)).reshape(got.shape)
            check(got.shape == want.shape and np.allclose(got, want, atol=1e-4), f"{cls.__name__} {act.__name__}: onnxruntime matches predict(x, 0)")
            check(session.run(None, {"x": xc[:3]})[0].shape[0] == 3, f"{cls.__name__} {act.__name__}: batch size is dynamic")
print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
