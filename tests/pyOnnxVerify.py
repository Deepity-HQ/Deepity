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

    for cls in (dp.SimplePCN, dp.SequentialPCN, dp.FullPCN, dp.DKPPCN, dp.GaussSeidelPCN):
        for act in (dp.ReLU, dp.TanH, dp.Sigmoid, dp.GeLU):
            net = cls(dp.Linear(12, 10), act(), dp.Linear(10, 6), act(), dp.Linear(6, 4), batch_size=8)
            net.configure(learning_rate=1e-3, inference_rate=0.1)
            weights = {}
            for i, (o, k) in enumerate([(10, 12), (6, 10), (4, 6)]):
                weights[f"layers.{i}.weight"] = rng.standard_normal((o, k)).astype(np.float32) * 0.5
                weights[f"layers.{i}.bias"] = rng.standard_normal(o).astype(np.float32) * 0.5
            if cls is dp.SequentialPCN:  # non-trivial learned precisions, so they are really exercised
                for i, width in ((1, 10), (2, 6), (3, 4)):
                    weights[f"layers.{i}.p"] = rng.uniform(0.3, 2.5, width).astype(np.float32)
            dp.import_weights(net, weights)
            for steps in (1, 5, 20):
                path = os.path.join(tmp, "s.onnx")
                dp.export_onnx(net, path, settling_steps=steps)
                session = ort.InferenceSession(path)
                got = session.run(None, {"x": x})[0]
                want = np.asarray(net.predict(x, steps)).reshape(got.shape)
                err = np.abs(got - want).max()
                check(err < 1e-4, f"{cls.__name__} {act.__name__}: {steps} settling steps match predict (max error {err:.1e})")
            check(session.run(None, {"x": x[:3]})[0].shape == (3, want.shape[1]), f"{cls.__name__} {act.__name__}: unrolled graph has a dynamic batch")
            if cls in (dp.SimplePCN, dp.SequentialPCN):
                check(not np.allclose(net.predict(x, 20), net.predict(x, 0), atol=1e-3), "settling changes the output, so the match is not vacuous")
    for cls in (dp.SimpleConvolutionalPCN, dp.ConvolutionalPCN):
        for act in (dp.ReLU, dp.TanH, dp.Sigmoid, dp.GeLU):
            # The stride-2 layer leaves a 3x3 map from 8x8, so the transposed convolution needs output_padding.
            net = cls(
                dp.Convolution(4, 3, 3, 1, 1, 1, 1), act(), dp.Convolution(6, 3, 3, 2, 2, 0, 0), act(),
                dp.Convolution(10, 3, 3), dp.Convolution(0, 1, 1), input_shape=(2, 8, 8), batch_size=8,
            )
            net.configure(learning_rate=1e-3, inference_rate=0.1)
            xc = rng.standard_normal((8, 2 * 8 * 8)).astype(np.float32)
            if cls is dp.ConvolutionalPCN:  # non-trivial learned precisions, so they are really exercised
                net.save(os.path.join(tmp, "p.safetensors"))
                weights = {k: np.array(v) for k, v in dp.read_tensors(os.path.join(tmp, "p.safetensors")).items()}
                for key in [k for k in weights if k.endswith(".p")]:
                    weights[key] = rng.uniform(0.3, 2.5, weights[key].shape).astype(np.float32)
                dp.import_weights(net, weights)
            for steps in (1, 5, 20):
                path = os.path.join(tmp, "cs.onnx")
                dp.export_onnx(net, path, settling_steps=steps)
                session = ort.InferenceSession(path)
                got = session.run(None, {"x": xc})[0]
                want = np.asarray(net.predict(xc, steps)).reshape(got.shape)
                err = np.abs(got - want).max()
                check(err < 1e-4 * max(1.0, np.abs(want).max()), f"{cls.__name__} {act.__name__}: {steps} settling steps match predict (max error {err:.1e})")
            check(session.run(None, {"x": xc[:3]})[0].shape == (3, want.shape[1]), f"{cls.__name__} {act.__name__}: unrolled graph has a dynamic batch")
            check(not np.allclose(net.predict(xc, 20), net.predict(xc, 1), atol=1e-4), f"{cls.__name__} {act.__name__}: settling changes the output, so the match is not vacuous")
    unsupported = dp.ConvolutionalPCN(dp.Convolution(4, 3, 3), dp.Convolution(0, 1, 1), input_shape=(1, 8, 8), batch_size=2)
    unsupported.configure(learning_rate=1e-3)
    try:
        dp.export_onnx(unsupported, os.path.join(tmp, "u.onnx"), settling_steps=100000)
        check(False, "an unreasonably large unrolled graph is refused")
    except ValueError:
        check(True, "an unreasonably large unrolled graph is refused")
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
