"""Self-describing model files: save, deepity.load, settling defaults, error paths.

Run from the repo root with the built extension importable, e.g.
  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run python tests/pyModelFilesVerify.py
"""
import json
import os
import struct
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402
from deepity.cli import describe as inspect_file  # noqa: E402

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


def edit_header(src: str, dst: str, edit) -> None:
    """Copies a model file with `edit(header_dict)` applied to its JSON header. Tensor offsets are
    relative to the data region and the hash covers only the data, so both stay valid."""
    raw = open(src, "rb").read()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    edit(header)
    text = json.dumps(header).encode()
    open(dst, "wb").write(struct.pack("<Q", len(text)) + text + raw[8 + size :])


def with_model(src: str, dst: str, edit_model) -> None:
    def edit(header):
        model = json.loads(header["__metadata__"]["deepity.model"])
        edit_model(model)
        header["__metadata__"]["deepity.model"] = json.dumps(model)

    edit_header(src, dst, edit)


def error_text(fn) -> str:
    try:
        fn()
    except Exception as error:
        return str(error)
    return ""


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

print("--- sharded save ---")
with tempfile.TemporaryDirectory() as tmp:
    for name, make in [
        ("DKPPCN", lambda: dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)),
        ("FullPCN", lambda: dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)),
    ]:
        net = make().configure(learning_rate=1e-3, optimizer="ADAM", psi_optimizer="ADAM")
        index = os.path.join(tmp, name + ".safetensors.index.json")
        net.save(index, max_shard_bytes=300)
        shards = [f for f in os.listdir(tmp) if f.startswith(name + "-")]
        check(len(shards) > 1, f"{name}: weights split over {len(shards)} shards")
        loaded = dp.load(index)
        check(np.array_equal(net.predict(x), loaded.predict(x)), f"{name}: deepity.load(index) reproduces predictions")
        y = np.zeros((8, 4), dtype=np.float32)
        for _ in range(2):
            net.train_step(x, y, 2)
        ck = os.path.join(tmp, name + ".ck.safetensors.index.json")
        net.save_checkpoint(ck, {"epoch": 3}, max_shard_bytes=300)
        resumed, state = dp.resume(ck)
        net.train_step(x, y, 2)
        resumed.train_step(x, y, 2)
        check(state == {"epoch": 3} and np.array_equal(net.predict(x), resumed.predict(x)),
              f"{name}: sharded checkpoint resumes exactly")
        check(raises(lambda: net.save(os.path.join(tmp, "plain.safetensors"), max_shard_bytes=300)),
              f"{name}: max_shard_bytes with a plain path raises")
        check(raises(lambda: net.save(index, max_shard_bytes=0)), f"{name}: zero shard size raises")

print("--- write_tensors ---")
with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(1)
    data = {
        "a": rng.standard_normal((5, 7)).astype(np.float32),
        "b": rng.standard_normal(11),
        "c": np.arange(6, dtype=np.int32).reshape(2, 3),
        "d": np.array([1, 2, 3], dtype=np.uint8),
        "e": np.asfortranarray(rng.standard_normal((4, 3)).astype(np.float32)),
        "f": np.array([True, False]),
        "g": np.float32(2.5) * np.ones((), dtype=np.float32),
    }
    path = os.path.join(tmp, "w.safetensors")
    dp.write_tensors(path, data, {"note": "hi"})
    back = dp.read_tensors(path)
    check(all(np.array_equal(back[k], np.asarray(v)) and back[k].shape == np.asarray(v).shape for k, v in data.items() if k != "f"),
          "round trip through read_tensors (incl. non-contiguous and 0-d)")
    check(back["f"].tolist() == [1, 0], "bool arrays are stored as one byte")
    check(dp._backend.dy.read_model_metadata(path)["note"] == "hi", "metadata is stored")
    check("deepity.content_blake3" in dp._backend.dy.read_model_metadata(path), "content hash is written")
    idx = os.path.join(tmp, "w.safetensors.index.json")
    dp.write_tensors(idx, data, max_shard_bytes=100)
    check(len([f for f in os.listdir(tmp) if f.startswith("w-")]) > 1, "sharded write splits the arrays")
    check(np.array_equal(dp.read_tensors(idx)["a"], data["a"]), "sharded arrays read back through the index")
    check(raises(lambda: dp.write_tensors(path, {"x": np.array(["a"])})), "string arrays are rejected")
    check(raises(lambda: dp.write_tensors(path, {"x": np.zeros(2, dtype=np.complex64)})), "complex arrays are rejected")
    check(raises(lambda: dp.write_tensors(path, {})), "empty dict is rejected")
    check(raises(lambda: dp.write_tensors(path, data, {"deepity.kind": "x"})), "reserved metadata keys are rejected")
    check(raises(lambda: dp.write_tensors(os.path.join(tmp, "no", "dir", "x.safetensors"), data)), "unwritable path raises")

print("--- load_weights with renamed tensors ---")
with tempfile.TemporaryDirectory() as tmp:
    for name, make in [
        ("DKPPCN", lambda: dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)),
        ("FullPCN", lambda: dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)),
    ]:
        src, dst = make().configure(learning_rate=1e-3), make().configure(learning_rate=1e-3)
        path = os.path.join(tmp, name + ".safetensors")
        src.save(path)
        own = {k: v for k, v in dp.read_tensors(path).items() if not k.endswith("psi")}
        foreign = {"fc1.weight": own["layers.0.weight"], "fc1.bias": own["layers.0.bias"],
                   "fc2.weight": own["layers.1.weight"], "fc2.bias": own["layers.1.bias"]}
        other = os.path.join(tmp, name + ".foreign.safetensors")
        dp.write_tensors(other, foreign)
        rename_map = {"fc1.weight": "layers.0.weight", "fc1.bias": "layers.0.bias", "fc2.weight": "layers.1.weight", "fc2.bias": "layers.1.bias"}
        missing, unexpected = dst.load_weights(other, False, lambda n: rename_map.get(n, n))
        check(not unexpected and all(m.startswith("layers.2") or m.endswith(".psi") for m in missing), f"{name}: a rename function maps foreign names to layers; only the terminal layer and the feedback weights are missing")
        check(np.array_equal(src.predict(x), dst.predict(x)), f"{name}: renamed weights reproduce predictions")
        check(raises(lambda: make().configure(learning_rate=1e-3).load_weights(other, True, None)), f"{name}: strict loading of unmapped names raises")
        wrong = dict(foreign, **{"fc1.weight": np.zeros((9, 12), dtype=np.float32)})
        bad = os.path.join(tmp, name + ".wrong.safetensors")
        dp.write_tensors(bad, wrong)
        check(raises(lambda: make().configure(learning_rate=1e-3).load_weights(bad, False, lambda n: rename_map.get(n, n))), f"{name}: a wrong shape raises even when not strict")

with tempfile.TemporaryDirectory() as tmp:
    print("--- format versions and unknown fields ---")
    base = dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8).configure(learning_rate=1e-3)
    src = os.path.join(tmp, "base.safetensors")
    mod = os.path.join(tmp, "mod.safetensors")
    base.save(src)
    want = base.predict(x)

    def set_meta(key, value):
        return lambda h: h["__metadata__"].__setitem__(key, value) if value is not None else h["__metadata__"].pop(key)

    edit_header(src, mod, set_meta("deepity.format_version", "2"))
    text = error_text(lambda: dp.load(mod))
    check("version 2" in text and "up to version 1" in text, "a newer format version is refused, naming both versions")
    check(raises(lambda: dp.dy.read_model_metadata(mod)), "metadata reads refuse it too")
    check(inspect_file(mod)["format_version"] == "2", "inspect still shows a newer file")
    edit_header(src, mod, set_meta("deepity.format_version", "abc"))
    check("invalid" in error_text(lambda: dp.load(mod)), "a malformed format version is refused")

    edit_header(src, mod, set_meta("deepity.format_version", None))
    check(np.array_equal(dp.load(mod).predict(x), want), "a file with no format version loads")

    def add_unknowns(header):
        header["__metadata__"]["deepity.from_the_future"] = "{}"
        model = json.loads(header["__metadata__"]["deepity.model"])
        model["new_top_level_field"] = [1, 2]
        model["layers"][0]["new_layer_field"] = "x"
        model["hyperparameters"]["learning_rate_warmup"] = 100
        header["__metadata__"]["deepity.model"] = json.dumps(model)
        inference = json.loads(header["__metadata__"].get("deepity.inference", "{}"))
        inference["new_inference_field"] = True
        header["__metadata__"]["deepity.inference"] = json.dumps(inference)

    edit_header(src, mod, add_unknowns)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = dp.load(mod)
    check(np.array_equal(loaded.predict(x), want), "unknown keys and fields are ignored, weights load")
    check(any("learning_rate_warmup" in str(w.message) for w in caught), "an unknown hyperparameter is dropped with a warning")

    with_model(src, mod, lambda m: m["layers"].append({"kind": "attention"}))
    check("attention" in error_text(lambda: dp.load(mod)), "an unknown layer kind is a clear error")
    with_model(src, mod, lambda m: m["layers"][0].pop("out"))
    check("lacks" in error_text(lambda: dp.load(mod)), "a layer missing a field is a clear error")
    with_model(src, mod, lambda m: m.pop("layers"))
    check("layers" in error_text(lambda: dp.load(mod)), "a description without layers is a clear error")
    edit_header(src, mod, lambda h: h["__metadata__"].__setitem__("deepity.model", "{not json"))
    check("deepity.model" in error_text(lambda: dp.load(mod)), "malformed JSON names the key")

def configured(net, **kwargs):
    net.configure(**kwargs)
    return net


with tempfile.TemporaryDirectory() as tmp:
    print("--- inference artifacts (no Psi) ---")
    y = np.random.default_rng(1).standard_normal((8, 4)).astype(np.float32)
    for name, make, has_psi in [
        ("DKPPCN", lambda: dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)
         .configure(learning_rate=1e-3, optimizer="ADAM"), True),
        ("FullPCN", lambda: dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8)
         .configure(learning_rate=1e-3, optimizer="ADAM"), True),
        ("SimplePCN", lambda: configured(dp.SimplePCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8),
                                         learning_rate=1e-3, inference_rate=0.1, optimizer="ADAM"), False),
    ]:
        net = make()
        for _ in range(2):
            net.train_step(x, y, 2)
        full, slim, ck, slimmed = (os.path.join(tmp, f"{name}.{k}.safetensors") for k in ("full", "slim", "ckpt", "conv"))
        net.save(full)
        net.save(slim, for_inference=True)
        net.save_checkpoint(ck, {"epoch": 3})
        names = lambda path: list(dp.read_tensors(path))
        has = lambda path: any(n.endswith(".psi") or n.startswith("optimizer.") for n in names(path))
        check(has(full) == has_psi and not has(slim), f"{name}: save(for_inference=True) leaves out the training-only tensors")
        check(os.path.getsize(slim) < os.path.getsize(full) if has_psi else abs(os.path.getsize(slim) - os.path.getsize(full)) < 256,
              f"{name}: the file is smaller" if has_psi else f"{name}: no Psi to drop, same size")
        meta = dp.dy.read_model_metadata(slim)
        check(meta["deepity.kind"] == "inference" and meta["deepity.training_tensors"] == "omitted", f"{name}: marked as an inference artifact")
        loaded = dp.load(slim)
        check(np.array_equal(loaded.predict(x, 2), net.predict(x, 2)), f"{name}: predictions identical after load")
        energy = loaded.train_step(x, y, 2)
        check(np.isfinite(energy) and np.isfinite(loaded.predict(x, 2)).all(), f"{name}: the loaded model can train again")
        check(raises(lambda: dp.resume(slim)), f"{name}: an inference artifact is not a checkpoint")

        from deepity.cli import convert
        convert(ck, slimmed, for_inference=True)
        check(not has(slimmed) and dp.dy.read_model_metadata(slimmed)["deepity.kind"] == "inference", f"{name}: convert --inference turns a checkpoint into a model file")
        check("deepity.training" not in dp.dy.read_model_metadata(slimmed), f"{name}: the saved training state is dropped")
        check(np.array_equal(dp.load(slimmed).predict(x, 2), net.predict(x, 2)), f"{name}: the converted file predicts the same")
        check(os.path.getsize(slimmed) < os.path.getsize(ck), f"{name}: smaller than the checkpoint it came from")

    index = os.path.join(tmp, "m.safetensors.index.json")
    net = dp.DKPPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8).configure(learning_rate=1e-3)
    net.save(index, max_shard_bytes=200, for_inference=True)
    check(np.array_equal(dp.load(index).predict(x), net.predict(x)), "a sharded inference artifact loads")

    # A file that simply lost a tensor, with no marker, must not load with random weights.
    bad = os.path.join(tmp, "cut.safetensors")
    net.save(os.path.join(tmp, "whole.safetensors"))
    edit_header(os.path.join(tmp, "whole.safetensors"), bad, lambda h: [h.pop(k) for k in list(h) if k.endswith(".weight") and k.startswith("layers.0")])
    check(raises(lambda: dp.load(bad)), "a file missing a weight tensor is refused")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
