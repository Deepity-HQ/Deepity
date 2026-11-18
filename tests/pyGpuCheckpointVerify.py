"""GPU save, load and checkpoint resume (needs a CUDA build and a GPU).

  python tests/pyGpuCheckpointVerify.py
Skipped (exit 0) when no CUDA device is usable. GPU math is not bit-reproducible (atomics, cuBLAS),
so resumed runs are compared with a tolerance, and the CPU-vs-GPU comparisons with a looser one.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402

failures = 0


# CPU and GPU matmuls differ by about 1e-3 relative on an A100 (TF32), so cross-device checks are relative.
REL_TOL = 5e-3


def check(ok, what):
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


def make(cls, device):
    net = cls(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8, device=device)
    net.configure(learning_rate=1e-3, inference_rate=0.1, optimizer="ADAM")
    return net


try:
    probe = make(dp.FullPCN, "cuda")
    probe.predict(np.zeros((8, 12), np.float32), 1)
except Exception as error:  # noqa: BLE001
    print(f"SKIPPED: no usable CUDA device ({type(error).__name__}: {str(error)[:100]})")
    sys.exit(0)

rng = np.random.default_rng(0)
x = rng.standard_normal((8, 12)).astype(np.float32)
y = rng.standard_normal((8, 4)).astype(np.float32)

with tempfile.TemporaryDirectory() as tmp:
    for cls in (dp.FullPCN, dp.DKPPCN, dp.SimplePCN):
        print(f"--- {cls.__name__} on cuda ---")
        gpu = make(cls, "cuda")
        for _ in range(3):
            gpu.train_step(x, y, 2)

        path = os.path.join(tmp, "g.safetensors")
        gpu.save(path)
        weights = dp.read_tensors(path)
        check(all(np.isfinite(v).all() for v in weights.values()), "a GPU save writes finite weights")
        check(any(np.abs(v).max() > 0 for v in weights.values()), "the saved weights are not all zero (real device data)")

        cpu = dp.load(path, device="cpu")
        a, b = np.asarray(gpu.predict(x, 2)), np.asarray(cpu.predict(x, 2))
        d_cpu = np.abs(a - b).max()
        check(d_cpu < REL_TOL * np.abs(a).max(), f"a GPU-saved model loads on the CPU and predicts the same (max difference {d_cpu:.1e}, output scale {np.abs(a).max():.1e})")

        back = dp.load(path, device="cuda")
        check(np.allclose(np.asarray(back.predict(x, 2)), a, atol=1e-5), "it loads back onto the GPU and predicts the same")

        ck = os.path.join(tmp, "g.ckpt.safetensors")
        gpu.save_checkpoint(ck, {"step": 3})
        resumed, state = dp.resume(ck, device="cuda")
        check(state == {"step": 3}, "training state survives")
        for _ in range(3):
            gpu.train_step(x, y, 2)
            resumed.train_step(x, y, 2)
        diff = np.abs(np.asarray(gpu.predict(x, 2)) - np.asarray(resumed.predict(x, 2))).max()
        check(diff < 1e-4, f"a resumed GPU run tracks the uninterrupted one (max difference {diff:.1e})")

        wo = dp.load(ck, device="cuda")
        for _ in range(3):
            wo.train_step(x, y, 2)
        wdiff = np.abs(np.asarray(gpu.predict(x, 2)) - np.asarray(wo.predict(x, 2))).max()
        check(wdiff > 10 * max(diff, 1e-7), f"a weights-only load does not resume (difference {wdiff:.1e}), so optimizer state was restored")

        cpu_resumed, _ = dp.resume(ck, device="cpu")
        d_res = np.abs(np.asarray(cpu_resumed.predict(x, 2)) - np.asarray(dp.resume(ck, device="cuda")[0].predict(x, 2))).max()
        check(d_res < REL_TOL * np.abs(a).max(), f"a GPU checkpoint resumes on the CPU with the same predictions (max difference {d_res:.1e})")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
