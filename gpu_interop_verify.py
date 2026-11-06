"""OSC verification script for GPU DLPack interop (FullPCN device="cuda").
Run from the repo root with the built extension importable. See the
companion command block for exact build/run steps.
"""

import time

import numpy as np
import torch

from deepity import FullPCN
from deepity.layer import Linear, ReLU

torch.manual_seed(0)
np.random.seed(0)

BATCH = 256
IN_DIM = 784
HIDDEN = 256
OUT_DIM = 10
STEPS = 5
WARMUP = 3
TIMED = 20

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "No GPU visible -- check nvidia-smi / $CUDA_VISIBLE_DEVICES."


def new_net():
    net = FullPCN(
        Linear(IN_DIM, HIDDEN),
        ReLU(),
        Linear(HIDDEN, OUT_DIM),
        batch_size=BATCH,
        device="cuda",
    )
    net.configure(learning_rate=0.005, inference_rate=0.1, feedback_rate=1e-4, seed=42)
    return net


net = new_net()

x_np = np.random.randn(BATCH, IN_DIM).astype(np.float32)
y_np = np.random.randint(0, OUT_DIM, size=BATCH).astype(np.int64)
y_onehot = np.eye(OUT_DIM, dtype=np.float32)[y_np]

x_cpu_torch = torch.from_numpy(x_np.copy())
y_cpu_torch = torch.from_numpy(y_onehot.copy())
x_cuda_torch = x_cpu_torch.cuda()
y_cuda_torch = y_cpu_torch.cuda()

print("\n=== Correctness: CUDA-tensor path vs. host round-trip ===")
net.randomize_weights(seed=123)
e_host = net.train_step(x_np.copy(), y_onehot.copy(), STEPS)
p_host = net.predict(x_np.copy(), STEPS).copy()

net.randomize_weights(seed=123)
e_cuda = net.train_step(x_cuda_torch, y_cuda_torch, STEPS)
p_cuda = net.predict(x_cuda_torch, STEPS)

print(f"train_step energy  host={e_host:.6f}  cuda={e_cuda:.6f}  "
      f"abs diff={abs(e_host - e_cuda):.6e}")
max_abs_diff = np.max(np.abs(p_host - p_cuda))
print(f"predict() max abs diff: {max_abs_diff:.6e}")
assert max_abs_diff < 1e-2, "Correctness check FAILED -- see memory note's fallback design."
print("Correctness check PASSED.")

print("\n=== Performance: CUDA-tensor path vs. host round-trip ===")


def run_host(n):
    for _ in range(n):
        net.predict(x_np, STEPS)


def run_cuda(n):
    for _ in range(n):
        net.predict(x_cuda_torch, STEPS)


run_host(WARMUP)
torch.cuda.synchronize()
t0 = time.perf_counter()
run_host(TIMED)
torch.cuda.synchronize()
host_s = (time.perf_counter() - t0) / TIMED

run_cuda(WARMUP)
torch.cuda.synchronize()
t0 = time.perf_counter()
run_cuda(TIMED)
torch.cuda.synchronize()
cuda_s = (time.perf_counter() - t0) / TIMED

print(f"host round-trip predict(): {host_s * 1e3:.3f} ms/call")
print(f"CUDA-tensor predict():     {cuda_s * 1e3:.3f} ms/call")
print(f"speedup: {host_s / cuda_s:.2f}x")
