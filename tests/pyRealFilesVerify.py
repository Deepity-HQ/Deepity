"""Real files from a real server: two small llama.cpp models from Hugging Face.

  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run --with gguf python tests/pyRealFilesVerify.py
Not part of CI (it depends on a third-party server). Skipped (exit 0) when the server cannot be reached
or the gguf package is missing. Checks the redirects, the ETag cache and the GGUF reader against files
that llama.cpp itself wrote (F32, Q4_0 and Q8_0), and this library's Q4_0 writer against llama.cpp's.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402

try:
    import gguf
except ImportError:
    print("SKIPPED: gguf not installed")
    sys.exit(0)

BASE = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/"
failures = 0


def check(ok, what):
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


with tempfile.TemporaryDirectory() as tmp:
    os.environ["DEEPITY_CACHE_DIR"] = os.path.join(tmp, "cache")
    try:
        local = dp.fetch(BASE + "stories260K.gguf")
    except OSError as error:
        print(f"SKIPPED: cannot reach the server ({str(error)[:100]})")
        sys.exit(0)
    check(os.path.getsize(local) == 1185376, "a model fetched through Hugging Face's redirects has the right size")
    stamp = os.stat(local).st_mtime_ns
    check(dp.fetch(BASE + "stories260K.gguf") == local and os.stat(local).st_mtime_ns == stamp, "the second fetch revalidates and does not download again")

    for name, kinds in (("stories260K.gguf", {"F32"}), ("stories15M-q4_0.gguf", {"Q4_0", "F32", "Q8_0"})):
        path = dp.fetch(BASE + name)
        tensors, meta = dp.read_gguf(path)
        reader = gguf.GGUFReader(path)
        seen, worst = set(), 0.0
        for t in reader.tensors:
            seen.add(t.tensor_type.name)
            ref = gguf.quants.dequantize(np.array(t.data), t.tensor_type).reshape(tensors[t.name].shape)
            worst = max(worst, float(np.abs(ref - tensors[t.name]).max()))
        check(seen == kinds and worst == 0.0 and meta["general.architecture"] == "llama", f"{name}: every tensor ({', '.join(sorted(seen))}) decodes identically to the reference package")

    path = dp.fetch(BASE + "stories15M-q4_0.gguf")
    tensors = dp.read_gguf(path)[0]
    first = next(t.name for t in gguf.GGUFReader(path).tensors if t.tensor_type.name == "Q4_0")
    again = os.path.join(tmp, "again.gguf")
    dp.write_gguf(again, {"w": tensors[first]}, None, "q4_0")
    check(np.array_equal(dp.read_gguf(again)[0]["w"], tensors[first]), "requantizing llama.cpp's Q4_0 weights with this writer reproduces them exactly")

    # hf:// names the same file, and a download interrupted partway resumes against the real CDN.
    import deepity._sources as sources

    os.environ["DEEPITY_CACHE_DIR"] = os.path.join(tmp, "cache2")
    original = sources._Response.read
    seen = {"bytes": 0, "cut": True}

    def flaky(self, n):
        if seen["cut"] and seen["bytes"] >= 4_000_000:
            raise ConnectionResetError("simulated cut")
        chunk = original(self, n)
        seen["bytes"] += len(chunk)
        return chunk

    sources._Response.read = flaky
    url = "hf://ggml-org/models/tinyllamas/stories15M-q4_0.gguf"
    try:
        dp.fetch(url)
        check(False, "the interrupted download raised")
    except OSError as error:
        check("cut short" in str(error), "a download cut at 4 MB of 19 MB is reported and its bytes are kept")
    seen["cut"] = False
    sources._Response.read = original
    resumed = dp.fetch(url)
    reference = dp.fetch(BASE + "stories15M-q4_0.gguf")
    check(os.path.getsize(resumed) == 19077344 and open(resumed, "rb").read() == open(reference, "rb").read(), "hf:// resumes with a real Range request and the file is identical to a straight download")
    check(np.array_equal(dp.read_gguf(resumed)[0][first], tensors[first]), "and it decodes")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
