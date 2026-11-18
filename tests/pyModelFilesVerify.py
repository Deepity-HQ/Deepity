"""Self-describing model files: save, deepity.load, settling defaults, error paths.

Run from the repo root with the built extension importable, e.g.
  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run python tests/pyModelFilesVerify.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402

failures = 0


def check(ok: bool, what: str) -> None:
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


def raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


x = np.random.default_rng(0).standard_normal((8, 12)).astype(np.float32)

with tempfile.TemporaryDirectory() as tmp:
    for name, make in [
        ("DKPPCN", lambda: dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)
         .configure(learning_rate=1e-3, center_weights=True, psi_decay=1e-4)),
        ("FullPCN", lambda: dp.FullPCN(dp.Linear(12, 10), dp.GeLU(), dp.Linear(10, 4), batch_size=8)
         .configure(learning_rate=1e-3, use_momentum=True, momentum_beta=0.8)),
        ("FullPCN float64", lambda: dp.FullPCN(dp.Linear(12, 10), dp.TanH(), dp.Linear(10, 4), batch_size=8, dtype="float64")
         .configure(learning_rate=1e-3)),
    ]:
        print(f"--- {name} ---")
        net = make()
        net.default_inference_steps = 3
        path = os.path.join(tmp, name.replace(" ", "_") + ".safetensors")
        net.save(path)
        want = net.predict(x)
        loaded = dp.load(path)
        check(type(loaded) is type(net), "deepity.load returns the same network class")
        check(loaded.default_inference_steps == 3, "settling default restored")
        check(np.array_equal(loaded.predict(x), want), "predictions identical after load")
        check(np.array_equal(loaded.predict(x, 1), net.predict(x, 1)), "explicit inference_steps still overrides")

        meta = dp.dy.read_model_metadata(path)
        check(meta["deepity.format_version"] == "1" and meta["deepity.kind"] == "inference", "metadata present")
        check("deepity.content_blake3" in meta, "content hash present")
        check(raises(lambda: dp.load(path + ".missing")), "missing file raises")

    print("--- hyperparameters survive ---")
    net = dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8).configure(
        learning_rate=2e-3, inference_rate=0.2, feedback_rate=3e-4, lmbda=5e-3, center_weights=True, psi_decay=1e-4)
    net.set_learning_rate(7e-4)
    path = os.path.join(tmp, "hp.safetensors")
    net.save(path)
    loaded = dp.load(path)
    check(loaded._learning_rate == 7e-4 and loaded._inference_rate == 0.2 and loaded._feedback_rate == 3e-4,
          "rates survive, including one changed after configure()")
    check(loaded._flags == {"center_weights": True, "psi_decay": 1e-4}, "center_weights and psi_decay survive")

    print("--- weights-only files and overrides ---")
    bare = dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8).configure()
    dp.dy.DirectKPPCNetwork.save(bare, os.path.join(tmp, "bare.safetensors"))
    check(raises(lambda: dp.load(os.path.join(tmp, "bare.safetensors"))), "file without an architecture raises a clear error")
    check(dp.load(path, batch_size=4).batch_size == 4, "batch_size override")

    print("--- corruption ---")
    raw = bytearray(open(path, "rb").read())
    raw[-3] ^= 1
    bad = os.path.join(tmp, "bad.safetensors")
    open(bad, "wb").write(raw)
    check(raises(lambda: dp.load(bad)), "flipped byte is rejected by the hash")

print("--- checkpoints resume training exactly ---")
y = np.random.default_rng(1).standard_normal((8, 4)).astype(np.float32)
with tempfile.TemporaryDirectory() as tmp:
    for name, make in [
        ("DKPPCN", lambda: dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)
         .configure(learning_rate=1e-3, optimizer="ADAM", psi_optimizer="ADAM")),
        ("FullPCN", lambda: dp.FullPCN(dp.Linear(12, 10), dp.GeLU(), dp.Linear(10, 4), batch_size=8)
         .configure(learning_rate=1e-3, optimizer="ADAM", psi_optimizer="ADAM")),
    ]:
        net = make()
        for _ in range(4):
            net.train_step(x, y, 2)
        rng = np.random.default_rng(42)
        rng.random(5)
        state = {"epoch": 3, "global_step": 4, "lr": 1e-3, "rng": rng.bit_generator.state}
        path = os.path.join(tmp, name + ".ckpt.safetensors")
        net.save_checkpoint(path, state)
        resumed, got = dp.resume(path)
        check(got == state, f"{name}: training state (including numpy RNG state) round trips")
        rng2 = np.random.default_rng()
        rng2.bit_generator.state = got["rng"]
        check(rng2.random() == rng.random(), f"{name}: restored RNG continues the same stream")
        for _ in range(3):
            net.train_step(x, y, 2)
            resumed.train_step(x, y, 2)
        check(np.array_equal(net.predict(x), resumed.predict(x)), f"{name}: resumed run matches the uninterrupted run")
        weights_only = dp.load(path)
        for _ in range(3):
            weights_only.train_step(x, y, 2)
        check(not np.array_equal(net.predict(x), weights_only.predict(x)),
              f"{name}: loading only the weights does NOT resume (optimizer state matters)")
        plain = os.path.join(tmp, name + ".plain.safetensors")
        net.save(plain)
        check(raises(lambda: dp.resume(plain)), f"{name}: resume rejects a non-checkpoint file")
    f64 = dp.FullPCN(dp.Linear(12, 10), dp.TanH(), dp.Linear(10, 4), batch_size=8, dtype="float64").configure(
        learning_rate=1e-3, optimizer="ADAM", psi_optimizer="ADAM")
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    for _ in range(3):
        f64.train_step(x64, y64, 2)
    path = os.path.join(tmp, "f64.ckpt.safetensors")
    f64.save_checkpoint(path)
    res64, _ = dp.resume(path)
    for _ in range(2):
        f64.train_step(x64, y64, 2)
        res64.train_step(x64, y64, 2)
    check(np.array_equal(f64.predict(x64), res64.predict(x64)), "FullPCN float64: resumed run matches the uninterrupted run")

print("--- load_weights and inference mode ---")
with tempfile.TemporaryDirectory() as tmp:
    for name, make in [
        ("DKPPCN", lambda: dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)),
        ("FullPCN", lambda: dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)),
    ]:
        src, dst = make().configure(learning_rate=1e-3), make().configure(learning_rate=1e-3)
        path = os.path.join(tmp, name + ".safetensors")
        src.save(path)
        missing, unexpected = dst.load_weights(path)
        check(not missing and not unexpected, f"{name}: strict load_weights of its own file reports nothing")
        check(np.array_equal(src.predict(x), dst.predict(x)), f"{name}: load_weights reproduces predictions")
        check(raises(lambda: dst.load_weights(path, rename=lambda n: "x." + n)), f"{name}: strict rename mismatch raises")
        missing, unexpected = dst.load_weights(path, strict=False, rename=lambda n: "x." + n)
        check(len(missing) > 0 and len(unexpected) > 0, f"{name}: non-strict reports missing and unexpected names")
    import json
    from deepity import _serialization as ser
    meta = ser.build_metadata(src, "FullPCN", {}, "float32")
    check(json.loads(meta["deepity.inference"])["mode"] == "iterative", "file states its inference mode")

print("--- read_tensors ---")
with tempfile.TemporaryDirectory() as tmp:
    net = dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8).configure(learning_rate=1e-3)
    path = os.path.join(tmp, "t.safetensors")
    net.save(path)
    for mm in (True, False):
        tensors = dp.read_tensors(path, mmap=mm)
        check(set(tensors) >= {"layers.0.weight", "layers.0.bias"}, f"mmap={mm}: names present")
        w = tensors["layers.0.weight"]
        check(w.shape == (10, 12) and w.dtype == np.float32, f"mmap={mm}: shape and dtype")
        check(not w.flags.writeable or not mm, f"mmap={mm}: mapped views are read-only")
    del tensors
    keep = dp.read_tensors(path)["layers.0.weight"]
    import gc; gc.collect()
    check(float(np.abs(keep).sum()) > 0, "view stays valid after the dict is dropped")
    check(raises(lambda: dp.read_tensors(os.path.join(tmp, "missing.safetensors"))), "missing file raises")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
