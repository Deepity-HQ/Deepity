"""Importing real PyTorch files: .pt and .pth written by torch.save, checked against torch's own forward pass.

  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run --with torch python tests/pyTorchImportVerify.py
Skipped (exit 0) when torch is not installed.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402

try:
    import torch
except ImportError:
    print("SKIPPED: torch not installed")
    sys.exit(0)

failures = 0


def check(ok, what):
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


def dense_net():
    return dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 6), dp.ReLU(), dp.Linear(6, 4), batch_size=8).configure(learning_rate=1e-3)


def dense_model(dtype=torch.float32):
    return torch.nn.Sequential(torch.nn.Linear(12, 10), torch.nn.ReLU(), torch.nn.Linear(10, 6), torch.nn.ReLU(), torch.nn.Linear(6, 4)).to(dtype)


torch.manual_seed(0)
x = np.random.default_rng(0).standard_normal((8, 12)).astype(np.float32)
with tempfile.TemporaryDirectory() as tmp:
    model = dense_model()
    want = model(torch.from_numpy(x)).detach().numpy()

    path = os.path.join(tmp, "seq.pt")
    torch.save(model.state_dict(), path)
    net = dense_net()
    missing, unexpected = dp.import_weights(net, path, rename="sequential")
    check(np.allclose(net.predict(x, 0), want, atol=1e-5), "a Sequential state_dict (.pt) matches torch's forward pass")
    check(missing == ["layers.3.weight", "layers.3.bias"] and not unexpected, "only the terminal layer's own tensors are reported missing")

    path = os.path.join(tmp, "ckpt.pth")
    torch.save({"epoch": 3, "state_dict": model.state_dict(), "optimizer": {"lr": 0.1}}, path)
    net = dense_net()
    dp.import_weights(net, path, rename="sequential")
    check(np.allclose(net.predict(x, 0), want, atol=1e-5), "a checkpoint holding a state_dict (.pth) imports")

    for dtype in (torch.float16, torch.bfloat16):
        half = dense_model(dtype)
        path = os.path.join(tmp, f"{dtype}.pt")
        torch.save(half.state_dict(), path)
        net = dense_net()
        dp.import_weights(net, path, rename="sequential")
        reference = half.float()(torch.from_numpy(x)).detach().numpy()
        check(np.allclose(net.predict(x, 0), reference, atol=1e-5), f"{dtype} weights are widened to float32 exactly")

    conv = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 10, 8))
    path = os.path.join(tmp, "conv.pt")
    torch.save(conv.state_dict(), path)
    cnet = dp.SimpleConvolutionalPCN(dp.Convolution(4, 3, 3, 1, 1, 1, 1), dp.ReLU(), dp.Convolution(10, 8, 8), dp.Convolution(0, 1, 1),
                                     input_shape=(2, 8, 8), batch_size=8)
    cnet.configure(learning_rate=1e-3)
    dp.import_weights(cnet, path, rename="sequential")
    xc = np.random.default_rng(1).standard_normal((8, 2, 8, 8)).astype(np.float32)
    got = np.asarray(cnet.predict_with_projection(xc.reshape(8, -1), 0)).reshape(8, -1)
    check(np.allclose(got, conv(torch.from_numpy(xc)).detach().numpy().reshape(8, -1), atol=1e-4), "a Conv2d state_dict matches torch's forward pass")

    path = os.path.join(tmp, "whole.pt")
    torch.save(model, path)
    try:
        dp.import_weights(dense_net(), path, rename="sequential")
        check(False, "a whole pickled model is refused")
    except Exception as error:  # noqa: BLE001
        check("weights" in str(error).lower(), f"a whole pickled model is refused ({type(error).__name__})")

    class Evil:
        def __reduce__(self):
            return os.system, ("echo pwned > " + os.path.join(tmp, "pwned"),)

    path = os.path.join(tmp, "evil.pt")
    torch.save({"w": Evil()}, path)
    try:
        dp.import_weights(dense_net(), path)
    except Exception:  # noqa: BLE001
        pass
    check(not os.path.exists(os.path.join(tmp, "pwned")), "a malicious pickle does not run code")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
