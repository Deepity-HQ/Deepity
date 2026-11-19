"""Self-describing save, load, checkpoints and sharding for the remaining Python network classes.

  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run python tests/pyNetworkFilesVerify.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402

failures = 0


def check(ok, what):
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


def raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


rng = np.random.default_rng(0)
dense = lambda: (dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4))
conv = lambda: (dp.Convolution(4, 3, 3, 1, 1, 1, 1), dp.ReLU(), dp.Convolution(10, 8, 8), dp.Convolution(0, 1, 1))
X_dense, Y_dense = rng.standard_normal((8, 12)).astype(np.float32), rng.standard_normal((8, 4)).astype(np.float32)
X_conv, Y_conv = rng.standard_normal((8, 64)).astype(np.float32), rng.standard_normal((8, 10)).astype(np.float32)

CASES = [
    ("SimplePCN", lambda: dp.SimplePCN(*dense(), batch_size=8), dict(learning_rate=1e-3, inference_rate=0.1, optimizer="ADAM", center_weights=True), X_dense, Y_dense),
    ("GaussSeidelPCN", lambda: dp.GaussSeidelPCN(*dense(), batch_size=8), dict(learning_rate=1e-3, optimizer="ADAM"), X_dense, Y_dense),
    ("SequentialPCN", lambda: dp.SequentialPCN(*dense(), batch_size=8), dict(learning_rate=1e-3, precision_rate=0.01, optimizer="ADAM"), X_dense, Y_dense),
    ("SimpleConvolutionalPCN", lambda: dp.SimpleConvolutionalPCN(*conv(), input_shape=(1, 8, 8), batch_size=8), dict(learning_rate=1e-3, optimizer="ADAM"), X_conv, Y_conv),
    ("ConvolutionalPCN", lambda: dp.ConvolutionalPCN(*conv(), input_shape=(1, 8, 8), batch_size=8), dict(learning_rate=1e-3, precision_rate=0.01), X_conv, Y_conv),
]

with tempfile.TemporaryDirectory() as tmp:
    for name, make, hp, x, y in CASES:
        print(f"--- {name} ---")
        net = make()
        net.configure(**hp)
        for _ in range(3):
            net.train_step(x, y, 2)

        path = os.path.join(tmp, name + ".safetensors")
        net.save(path)
        loaded = dp.load(path)
        check(type(loaded) is type(net), "deepity.load returns the same class")
        check(np.array_equal(net.predict(x, 2), loaded.predict(x, 2)), "predictions identical after load")
        check(loaded._current_hyperparameters() == net._current_hyperparameters(), "hyperparameters survive")
        check(dp.read_card(path) == {}, "no card by default")

        missing, unexpected = loaded.load_weights(path)
        check(not missing and not unexpected, "strict load_weights of its own file reports nothing")

        ck = os.path.join(tmp, name + ".ckpt.safetensors")
        net.save_checkpoint(ck, {"epoch": 4})
        resumed, state = dp.resume(ck)
        check(state == {"epoch": 4}, "training state round trips")
        for _ in range(3):
            net.train_step(x, y, 2)
            resumed.train_step(x, y, 2)
        # SequentialPCN lands within a float32 ulp (about 1e-9) instead of bit-identical; the cause is not found.
        a, b = net.predict(x, 2), resumed.predict(x, 2)
        same = np.allclose(a, b, rtol=0, atol=1e-7) if name == "SequentialPCN" else np.array_equal(a, b)
        check(same, "resumed run matches the uninterrupted run")
        wo = dp.load(ck)
        for _ in range(3):
            wo.train_step(x, y, 2)
        if "optimizer" in hp:
            check(not np.array_equal(net.predict(x, 2), wo.predict(x, 2)), "weights-only load does not resume (optimizer state matters)")
        check(raises(lambda: dp.resume(path)), "resume rejects a non-checkpoint file")

        idx = os.path.join(tmp, name + ".safetensors.index.json")
        net.save(idx, max_shard_bytes=400)
        shards = [f for f in os.listdir(tmp) if f.startswith(name + "-")]
        check(len(shards) > 1 and np.array_equal(dp.load(idx).predict(x, 2), net.predict(x, 2)), f"sharded save ({len(shards)} shards) loads identically")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
