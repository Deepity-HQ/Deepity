"""Load speed of Deepity model files against raw safetensors and np.load.

  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run --with safetensors python experiments/benchmark_model_io.py [MiB]

Writes one file of random float32 tensors (default 512 MiB) in each format, then times a full
load that touches every byte (sum of each tensor), best of 5. Timings are warm-cache: drop the page
cache as root (`echo 3 > /proc/sys/vm/drop_caches`) between runs for cold numbers.
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402
from deepity._backend import dy  # noqa: E402


def best(fn, runs=5):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def touch(tensors):
    return sum(float(t.sum(dtype=np.float64)) for t in tensors.values())


def main():
    mib = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rows = 16
    cols = mib * (1 << 20) // 4 // rows // 8
    rng = np.random.default_rng(0)
    data = {f"layers.{i}.weight": rng.standard_normal((rows * 1, cols)).astype(np.float32) for i in range(8)}
    total = sum(a.nbytes for a in data.values())
    print(f"{len(data)} tensors, {total / (1 << 20):.0f} MiB")

    with tempfile.TemporaryDirectory() as tmp:
        st_path, npz_path = os.path.join(tmp, "m.safetensors"), os.path.join(tmp, "m.npz")
        net_less = {}
        try:
            from safetensors.numpy import load_file, save_file

            save_file(data, st_path)
        except ImportError:
            load_file = None
            print("safetensors not installed; Deepity-only run (write needs the package)")
        np.savez(npz_path, **data)

        cases = []
        if load_file is not None:
            cases += [
                ("deepity read_tensors, mmap, verify off", lambda: touch(dy.read_tensors(st_path, False, True))),
                ("deepity read_tensors, copy, verify off", lambda: touch(dy.read_tensors(st_path, False, False))),
                ("safetensors.numpy.load_file", lambda: touch(load_file(st_path))),
            ]
        cases.append(("np.load (.npz)", lambda: touch(np.load(npz_path))))
        # A Deepity-written file carries a BLAKE3, so verification can be timed too.
        dp_path = os.path.join(tmp, "d.safetensors")
        net = dp.FullPCN(dp.Linear(8192, 8192), batch_size=1)
        net.configure(learning_rate=1e-3)
        net.save(dp_path)
        size = os.path.getsize(dp_path)
        cases += [
            ("deepity file, mmap, verify off", lambda: touch(dy.read_tensors(dp_path, False, True))),
            ("deepity file, mmap, verify on (BLAKE3)", lambda: touch(dy.read_tensors(dp_path, True, True))),
        ]
        for name, fn in cases:
            t = best(fn)
            nbytes = size if name.startswith("deepity file") else total
            print(f"{name:42s} {t * 1000:9.1f} ms {nbytes / t / 1e9:7.2f} GB/s")


if __name__ == "__main__":
    main()
