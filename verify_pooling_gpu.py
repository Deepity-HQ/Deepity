"""
GPU smoke test for max pooling: mirrors the CPU check already run before
this (a 2-layer network, pooling conv -> terminal, trained repeatedly on a
fixed input, energy should decrease cleanly). This is the first time the
new MaxPool2DForward/MaxPool2DBackward CUDA kernels actually run -- they
were written by close analogy to the existing, working Im2Col/Col2Im
kernels but never compiled or executed (no CUDA toolkit in the sandbox
that wrote them). Run this before trusting pooling for anything real.

Usage: python verify_pooling_gpu.py
"""
import numpy as np
from pydeepity import dy  # pyright: ignore[reportAttributeAccessIssue]

net = dy.FullConvPCNetwork(batch_size=1, device="gpu")

# 1x4x4 -> conv(2,k3,s1,p0) -> 2x2x2 -> pool(2,2) -> 2x1x1 -> terminal.
net.add_layer(1, 2, 4, 4, 3, 3, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
             terminal_size=2, lr=0.05, ir=0.1, fl=0.0, lmbda=0.0,
             activation="tanh", activation_deriv="dtanh",
             pool_h=2, pool_w=2, pool_stride_h=2, pool_stride_w=2)
net.add_layer(2, 0, 1, 1, 1, 1, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
             terminal_size=2, lr=0.05, ir=0.1, fl=0.0, lmbda=0.0,
             activation="tanh", activation_deriv="dtanh")

net.set_use_mu_pc_scaling(False)
net.set_use_residual_connections(False)
net.compile()
net.randomize_weights(7)
net.set_use_epc(False)
net.set_optimizer("SGD")

layer0 = net.layers[0]
print(f"pooling layer out_height/out_width: {layer0.out_height}/{layer0.out_width}")
assert layer0.out_height == 1 and layer0.out_width == 1, "pooled resolution wrong"

x = np.array([0.1 * (i % 5) - 0.2 for i in range(16)], dtype=np.float32)
y = np.array([0.8, -0.8], dtype=np.float32)

first = 0.0
last = 0.0
for step in range(200):
    e = net.train_step(x, y, 5)
    if not np.isfinite(e):
        raise RuntimeError(f"FAIL: non-finite energy at step {step}")
    if step == 0:
        first = e
    last = e
    if step % 50 == 0:
        print(f"step {step:3d}: energy = {e:.6f}")

print(f"first energy = {first:.6f}, last energy = {last:.6f}")
if last >= first:
    raise RuntimeError("FAIL: energy did not decrease -- GPU pooling kernels are likely wrong")
print("PASS: energy decreased monotonically overall with pooling active on GPU")
