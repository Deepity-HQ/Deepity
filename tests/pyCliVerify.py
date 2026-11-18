"""Model cards, `deepity inspect`, `deepity convert` and the conversions behind them.

  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run python tests/pyCliVerify.py
"""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402
from deepity import cli  # noqa: E402

failures = 0


def check(ok, what):
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(args))
    return code, out.getvalue(), err.getvalue()


x = np.random.default_rng(0).standard_normal((8, 12)).astype(np.float32)
with tempfile.TemporaryDirectory() as tmp:
    net = dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8).configure(learning_rate=1e-3)
    net.card = {"license": "MIT", "description": "tiny test net", "metrics": {"acc": 0.5}}
    path = os.path.join(tmp, "m.safetensors")
    net.save(path)

    print("--- model card ---")
    check(dp.read_card(path)["license"] == "MIT", "card is stored and read back")
    again = dp.load(path)
    check(again.card == net.card, "deepity.load restores the card")
    again.save(os.path.join(tmp, "again.safetensors"))
    check(dp.read_card(os.path.join(tmp, "again.safetensors")) == net.card, "a re-save keeps the card")
    net.card = None
    net.save(os.path.join(tmp, "nocard.safetensors"))
    check(dp.read_card(os.path.join(tmp, "nocard.safetensors")) == {}, "no card means an empty card")
    for bad, what in (({"x": object()}, "non-JSON card"), ({"x": "a" * 70000}, "oversized card"), ("text", "non-dict card")):
        net.card = bad
        try:
            net.save(os.path.join(tmp, "bad.safetensors"))
            check(False, f"{what} is rejected")
        except (TypeError, ValueError):
            check(True, f"{what} is rejected")
    net.card = {"license": "MIT"}

    print("--- inspect ---")
    code, out, _ = run("inspect", path)
    check(code == 0 and "FullPCN" in out and "layers.0.weight" in out and "verified" in out, "inspect prints network, tensors and hash status")
    check("license" in out and "MIT" in out, "inspect prints the card")
    code, out, _ = run("inspect", path, "--json")
    info = json.loads(out)
    check(code == 0 and info["format"] == "safetensors" and info["tensor_count"] == len(info["tensors"]), "inspect --json is valid JSON")
    code, _, err = run("inspect", os.path.join(tmp, "missing.safetensors"))
    check(code == 1 and "error" in err, "inspect reports a missing file with exit code 1")
    raw = bytearray(open(path, "rb").read())
    raw[-1] ^= 1
    open(os.path.join(tmp, "corrupt.safetensors"), "wb").write(raw)
    check(run("inspect", os.path.join(tmp, "corrupt.safetensors"))[0] == 1, "inspect rejects a corrupted file")
    check(run("inspect", os.path.join(tmp, "corrupt.safetensors"), "--no-verify")[0] == 0, "--no-verify skips the hash")
    proc = subprocess.run([sys.executable, "-m", "deepity", "inspect", path], capture_output=True, text=True,
                          env={**os.environ, "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"})
    check(proc.returncode == 0 and "FullPCN" in proc.stdout, "python -m deepity works")

    print("--- convert ---")
    index = os.path.join(tmp, "s.safetensors.index.json")
    check(run("convert", path, index, "--max-shard-size", "300")[0] == 0, "convert to a shard set")
    check(len([f for f in os.listdir(tmp) if f.startswith("s-")]) > 1, "the shard set has several files")
    check(np.array_equal(dp.load(index).predict(x), dp.load(path).predict(x)), "the sharded copy predicts identically")
    merged = os.path.join(tmp, "merged.safetensors")
    check(run("convert", index, merged)[0] == 0 and np.array_equal(dp.load(merged).predict(x), dp.load(path).predict(x)), "a shard set merges back")
    check(dp.read_card(merged) == dp.read_card(path) and dp.read_card(path)["license"] == "MIT", "conversion keeps the metadata")
    check(run("convert", path, os.path.join(tmp, "p.safetensors"), "--max-shard-size", "1MB")[0] == 1, "a shard size without an index destination is an error")
    q = os.path.join(tmp, "q.safetensors")
    dp.write_tensors(os.path.join(tmp, "f.safetensors"), {"a": np.arange(4, dtype=np.float32)})

    def gguf_blob():
        def s(t): return struct.pack("<Q", len(t)) + t.encode()
        data = np.arange(8, dtype=np.float32).tobytes()
        head = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + s("general.name") + struct.pack("<I", 8) + s("tiny")
        head += s("w") + struct.pack("<I", 2) + struct.pack("<QQ", 4, 2) + struct.pack("<IQ", 0, 0)
        head += b"\0" * ((-len(head)) % 32)
        return head + data
    g = os.path.join(tmp, "t.gguf")
    open(g, "wb").write(gguf_blob())
    code, out, _ = run("inspect", g)
    check(code == 0 and "gguf" in out and "tiny" in out, "inspect reads a GGUF file")
    out_st = os.path.join(tmp, "from_gguf.safetensors")
    check(run("convert", g, out_st)[0] == 0, "convert GGUF to safetensors")
    t = dp.read_tensors(out_st)["w"]
    check(t.shape == (2, 4) and np.array_equal(t.ravel(), np.arange(8)), "GGUF tensor values and shape survive")
    check(json.loads(dy_meta := dp._backend.dy.read_model_metadata(out_st)["gguf.metadata"])["general.name"] == "tiny", "GGUF metadata is kept")

    print("--- sizes ---")
    check(cli.parse_size("2GiB") == 2 << 30 and cli.parse_size("500MB") == 500 * 10**6 and cli.parse_size("1024") == 1024, "size strings parse")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
