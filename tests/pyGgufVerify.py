"""GGUF reader: hand-built files, hostile files, and (when the `gguf` package is installed) real ones.

  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run --with gguf python tests/pyGgufVerify.py
"""
import json
import os
import struct
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


def s(text):
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def build(tensors, kv=b"", n_kv=0, version=3, align=32):
    """tensors: list of (name, ne list, ggml type, raw bytes)."""
    infos, data, off = b"", b"", 0
    for name, ne, typ, raw in tensors:
        infos += s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne)
        infos += struct.pack("<IQ", typ, off)
        data += raw + b"\0" * ((-len(raw)) % align)
        off += len(raw) + ((-len(raw)) % align)
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), n_kv) + kv + infos
    head += b"\0" * ((-len(head)) % align)
    return head + data


def kv_u32(key, v):
    return s(key) + struct.pack("<II", 4, v)


def kv_str(key, v):
    return s(key) + struct.pack("<I", 8) + s(v)


def write(tmp, name, blob):
    path = os.path.join(tmp, name)
    with open(path, "wb") as f:
        f.write(blob)
    return path


rng = np.random.default_rng(0)
with tempfile.TemporaryDirectory() as tmp:
    f32 = rng.standard_normal((3, 4)).astype(np.float32)
    f16 = rng.standard_normal((2, 8)).astype(np.float16)
    d = np.float16(0.25)
    q8 = rng.integers(-128, 128, 32).astype(np.int8)
    q4_bytes = rng.integers(0, 256, 16).astype(np.uint8)
    ints = np.arange(5, dtype=np.int32)
    blob = build(
        [
            ("w", [4, 3], 0, f32.tobytes()),                       # GGUF lists the fastest axis first
            ("h", [8, 2], 1, f16.tobytes()),
            ("q8", [32], 8, d.tobytes() + q8.tobytes()),
            ("q4", [32], 2, d.tobytes() + q4_bytes.tobytes()),
            ("i", [5], 26, ints.tobytes()),
        ],
        kv_u32("general.alignment", 32) + kv_str("general.name", "tiny"),
        2,
    )
    path = write(tmp, "tiny.gguf", blob)
    tensors, meta = dp.read_gguf(path)
    check(np.array_equal(tensors["w"], f32) and tensors["w"].shape == (3, 4), "F32 tensor, shape reversed to row-major")
    check(np.array_equal(tensors["h"], f16.astype(np.float32)), "F16 tensor widened to float32")
    check(np.allclose(tensors["q8"], q8 * np.float32(d)), "Q8_0 dequantizes")
    want4 = np.concatenate([((q4_bytes & 15).astype(np.float32) - 8), ((q4_bytes >> 4).astype(np.float32) - 8)]) * np.float32(d)
    check(np.allclose(tensors["q4"], want4), "Q4_0 dequantizes (low nibbles first, then high)")
    check(tensors["i"].dtype == np.int64 and tensors["i"].tolist() == [0, 1, 2, 3, 4], "I32 tensor comes back as int64")
    check(meta == {"general.alignment": 32, "general.name": "tiny"}, "metadata is decoded")

    # import_weights takes a .gguf path
    net = dp.FullPCN(dp.Linear(4, 3), dp.ReLU(), dp.Linear(3, 2), batch_size=2).configure(learning_rate=1e-3)
    gg = write(tmp, "net.gguf", build([("layers.0.weight", [4, 3], 0, f32.tobytes()), ("layers.0.bias", [3], 0, np.zeros(3, np.float32).tobytes())]))
    missing, unexpected = dp.import_weights(net, gg)
    check("layers.0.weight" not in missing and not unexpected, "import_weights reads a .gguf file")

    # hostile files
    check(raises(lambda: dp.read_gguf(write(tmp, "magic.gguf", b"GGUX" + blob[4:]))), "bad magic is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "ver.gguf", build([], version=9)))), "unknown version is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "trunc.gguf", blob[: len(blob) - 24]))), "a tensor running past the end is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "hdr.gguf", blob[:40]))), "a file ending inside the header is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "count.gguf", b"GGUF" + struct.pack("<IQQ", 3, 2 ** 40, 0)))), "an absurd tensor count is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "kv.gguf", b"GGUF" + struct.pack("<IQQ", 3, 0, 2 ** 40)))), "an absurd key-value count is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "str.gguf", b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2 ** 62)))), "an absurd string length is rejected")
    arr = s("a") + struct.pack("<IIQ", 9, 4, 2 ** 40)
    check(raises(lambda: dp.read_gguf(write(tmp, "arr.gguf", build([], arr, 1)))), "an array longer than the file is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "dup.gguf", build([("a", [1], 0, b"\0" * 4), ("a", [1], 0, b"\0" * 4)])))), "duplicate tensor names are rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "blk.gguf", build([("q", [33], 8, b"\0" * 40)])))), "a row that is not a whole number of blocks is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "typ.gguf", build([("q", [4], 99, b"\0" * 16)])))), "an unknown ggml type is rejected")
    check(raises(lambda: dp.read_gguf(write(tmp, "q9.gguf", build([("q", [32], 9, b"\0" * 40)])))),
          "a type this reader cannot decode (Q8_1) raises")
    # K-quants: an all-zero block decodes to zeros
    kz = dp.read_gguf(write(tmp, "kz.gguf", build([("q", [256], 12, b"\0" * 144)])))[0]["q"]
    check(kz.shape == (256,) and not kz.any(), "an all-zero Q4_K block decodes to zeros")
    check(raises(lambda: dp.read_gguf(os.path.join(tmp, "missing.gguf"))), "a missing file raises")
    check(raises(lambda: dp.read_gguf(write(tmp, "empty.gguf", b""))), "an empty file raises")
    big = write(tmp, "align.gguf", build([("w", [1], 0, b"\0" * 4)], kv_u32("general.alignment", 3), 1))
    check(raises(lambda: dp.read_gguf(big)), "a non power-of-two alignment is rejected")
    # random bytes after a valid magic must never crash
    ok = True
    for _ in range(200):
        junk = b"GGUF" + struct.pack("<I", 3) + rng.bytes(int(rng.integers(0, 200)))
        try:
            dp.read_gguf(write(tmp, "junk.gguf", junk))
        except Exception:
            pass
    check(ok, "200 random headers raise cleanly instead of crashing")

    # oracle: the reference gguf package, when available
    try:
        import gguf
        from gguf import quants
    except ImportError:
        print("  (gguf package not installed: skipping the reference-implementation checks)")
    else:
        for qt in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q4_1, gguf.GGMLQuantizationType.Q5_0,
                   gguf.GGMLQuantizationType.Q5_1, gguf.GGMLQuantizationType.Q8_0):
            x = rng.standard_normal((6, 64)).astype(np.float32)
            ref_path = os.path.join(tmp, f"ref_{qt.name}.gguf")
            w = gguf.GGUFWriter(ref_path, "llama")
            w.add_tensor("t", quants.quantize(x, qt), raw_dtype=qt)
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            mine = dp.read_gguf(ref_path)[0]["t"]
            theirs = quants.dequantize(quants.quantize(x, qt), qt)
            check(mine.shape == x.shape and np.allclose(mine, theirs, atol=1e-6), f"{qt.name}: matches the reference gguf dequantizer")
            check(np.abs(mine - x).max() < 0.5, f"{qt.name}: close to the original values")

def k_quant_oracle(tmp):
    """Random block bytes (valid for any bit pattern once the half scales are finite) against gguf's dequantizer."""
    import gguf
    from gguf import quants

    layout = {  # type: (bytes per block, offsets of half-float fields, offset of a float32 field or None)
        "Q2_K": (84, [80, 82], None), "Q3_K": (110, [108], None), "Q4_K": (144, [0, 2], None),
        "Q5_K": (176, [0, 2], None), "Q6_K": (210, [208], None), "Q8_K": (292, [], 0),
        "IQ2_XXS": (66, [0], None), "IQ2_XS": (74, [0], None), "IQ2_S": (82, [0], None),
        "IQ3_XXS": (98, [0], None), "IQ3_S": (110, [0], None), "IQ1_S": (50, [0], None),
        "IQ1_M": (56, [], None), "IQ4_NL": (18, [0], None), "IQ4_XS": (136, [0], None),
    }
    for name, (size, halves, f32off) in layout.items():
        qt = getattr(gguf.GGMLQuantizationType, name)
        rows, blocks = 5, 3
        raw = rng.integers(0, 256, (rows, blocks * size), dtype=np.uint8)
        for r in range(rows):
            for b in range(blocks):
                for off in halves:
                    raw[r, b * size + off: b * size + off + 2] = np.frombuffer(np.float16(rng.uniform(0.01, 0.5)).tobytes(), np.uint8)
                if name == "IQ1_M":  # the half scale is spread over the top nibbles of the four 16-bit scales
                    bits = int(np.frombuffer(np.float16(rng.uniform(0.01, 0.5)).tobytes(), np.uint16)[0])
                    for i in range(4):
                        pos = b * size + 48 + 2 * i + 1
                        raw[r, pos] = (raw[r, pos] & 0x0F) | (((bits >> (4 * i)) & 0xF) << 4)
                if f32off is not None:
                    raw[r, b * size: b * size + 4] = np.frombuffer(np.float32(rng.uniform(0.01, 0.5)).tobytes(), np.uint8)
        path = os.path.join(tmp, f"k_{name}.gguf")
        w = gguf.GGUFWriter(path, "llama")
        w.add_tensor("t", raw, raw_dtype=qt)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        mine = dp.read_gguf(path)[0]["t"]
        if name == "Q8_K":  # the reference package has no Q8_K dequantizer: d * qs, per block
            blk = raw.reshape(rows, blocks, size)
            d = np.ascontiguousarray(blk[:, :, :4]).view(np.float32)[:, :, 0]
            theirs = (blk[:, :, 4:260].view(np.int8).astype(np.float32) * d[:, :, None]).reshape(rows, blocks * 256)
        else:
            theirs = quants.dequantize(raw, qt)
        check(mine.shape == theirs.shape and np.allclose(mine, theirs, rtol=1e-5, atol=1e-6),
              f"{name}: random blocks match the reference gguf dequantizer")


with tempfile.TemporaryDirectory() as tmp2:
    try:
        import gguf  # noqa: F401
    except ImportError:
        pass
    else:
        k_quant_oracle(tmp2)


def writer_checks(tmp):
    from deepity.cli import convert

    x = rng.standard_normal((4, 64)).astype(np.float32)
    tensors = {"a.weight": x, "a.bias": rng.standard_normal(4).astype(np.float32), "i": np.arange(6, dtype=np.int32).reshape(2, 3),
               "d": rng.standard_normal((2, 2)).astype(np.float64), "s": np.float32(3.5), "h": x.astype(np.float16)}
    meta = {"general.name": "w", "n": 7, "big": 1 << 40, "f": 0.5, "flag": True, "tags": ["a", "b"], "nums": [1, 2.5], "obj": {"k": [1, 2]}}
    path = os.path.join(tmp, "w.gguf")
    dp.write_gguf(path, tensors, meta)
    got, gm = dp.read_gguf(path)
    check(np.array_equal(got["a.weight"], x) and got["a.weight"].shape == (4, 64), "F32 round trip keeps values and shape")
    check(np.array_equal(got["h"], x.astype(np.float16).astype(np.float32)), "F16 tensors stay F16")
    check(np.allclose(got["d"], tensors["d"]) and got["i"].tolist() == [[0, 1, 2], [3, 4, 5]], "F64 and int tensors round trip")
    check(got["s"].shape == (1,) and got["s"][0] == 3.5, "a scalar is stored as one element")
    check(gm["general.name"] == "w" and gm["n"] == 7 and gm["big"] == 1 << 40 and gm["flag"] is True and gm["tags"] == ["a", "b"]
          and gm["general.architecture"] == "deepity" and json.loads(gm["obj"]) == {"k": [1, 2]}, "metadata round trips")
    for dt, tol in (("f16", 2e-3), ("bf16", 2e-2), ("q8_0", 0.05)):
        p2 = os.path.join(tmp, f"w_{dt}.gguf")
        dp.write_gguf(p2, {"w": x, "b": tensors["a.bias"]}, None, dt)
        back = dp.read_gguf(p2)[0]
        check(back["w"].shape == x.shape and np.abs(back["w"] - x).max() < tol, f"{dt}: values within tolerance")
        check(np.array_equal(back["b"], tensors["a.bias"]) if dt == "q8_0" else True, f"{dt}: small tensors handled")
    check(raises(lambda: dp.write_gguf(os.path.join(tmp, "x.gguf"), {"n" * 65: x})), "a name over 64 bytes is rejected")
    check(raises(lambda: dp.write_gguf(os.path.join(tmp, "x.gguf"), {"t": np.zeros((1,) * 5, np.float32)})), "more than 4 dimensions is rejected")
    check(raises(lambda: dp.write_gguf(os.path.join(tmp, "x.gguf"), {"t": x}, {"general.alignment": 64})), "the alignment key is reserved")
    check(raises(lambda: dp.write_gguf(os.path.join(tmp, "x.gguf"), {"t": x}, None, "q4_0")), "an unknown dtype is rejected")
    check(not [f for f in os.listdir(tmp) if f.endswith(".tmp")], "no temp files are left behind")
    try:
        import gguf
        reader = gguf.GGUFReader(path)
        names = {t.name: t for t in reader.tensors}
        check(set(names) == set(tensors) and names["a.weight"].shape.tolist() == [64, 4], "the reference gguf reader opens the file")
        check(np.array_equal(np.array(names["a.weight"].data).reshape(4, 64), x), "the reference reader sees the same values")
        qpath = os.path.join(tmp, "w_q8_0.gguf")
        qt = {t.name: t for t in gguf.GGUFReader(qpath).tensors}["w"]
        ref = gguf.quants.dequantize(np.array(qt.data), gguf.GGMLQuantizationType.Q8_0)
        check(np.array_equal(ref.reshape(x.shape), dp.read_gguf(qpath)[0]["w"]), "Q8_0 written here decodes identically in the reference package")
    except ImportError:
        print("  (gguf package not installed: skipping the reference reader checks)")

    net = dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=2).configure(learning_rate=1e-3)
    ng = os.path.join(tmp, "net.gguf")
    dp.export_gguf(net, ng)
    nt, nm = dp.read_gguf(ng)
    check("layers.0.weight" in nt and nt["layers.0.weight"].shape == (10, 12), "export_gguf writes the network's weights")
    check(json.loads(nm["deepity.model"])["network"] == "FullPCN", "export_gguf records the architecture")
    clone = dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=2).configure(learning_rate=1e-3)
    missing, unexpected = dp.import_weights(clone, ng)
    xin = rng.standard_normal((2, 12)).astype(np.float32)
    check(np.array_equal(net.predict(xin, 0), clone.predict(xin, 0)), "a network imported from its own GGUF predicts identically")

    st = os.path.join(tmp, "c.safetensors")
    net.save(st)
    cg = os.path.join(tmp, "c.gguf")
    convert(st, cg, None, "f16")
    check(dp.read_gguf(cg)[0]["layers.0.weight"].shape == (10, 12), "convert safetensors to GGUF")
    back = os.path.join(tmp, "back.safetensors")
    convert(cg, back)
    check(raises(lambda: convert(st, cg, 1000)), "a shard size with a GGUF destination is an error")
    check(raises(lambda: convert(cg, back, None, "f16")), "--dtype with a safetensors destination is an error")


with tempfile.TemporaryDirectory() as tmp3:
    writer_checks(tmp3)

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
