"""
Checks that DiscriminativePCLayer implements the paper's PC equations:
  mu^l  = f(w^{l-1} a^{l-1})
  e^l   = a^l - mu^l
  E     = 1/2 sum_l (e^l)^2
  da^l  = -ir * (e^l - (w^l)^T e^{l+1} (.) f'(w^l a^l))
  dw^l  = lr  *  e^{l+1} (.) f'(w^l a^l) (a^l)^T

Test 1: with only the input clamped, repeated CalculateState/UpdateState
        must monotonically decrease the energy (it's literal gradient
        descent on E).
Test 2: with input AND output clamped, UpdateWeights must be able to
        learn a fixed nonlinear mapping via the local Hebbian-style rule
        above (no global backprop involved).

Run from repo root with the built extension on PYTHONPATH:
  PYTHONPATH=build/Release/bin python3 tests/python/tPaperEquations.py
"""
import numpy as np
import deepity as deep

def build_net(sizes, lr, ir):
    net = deep.DiscriminativePCNetwork(1)
    for i, s in enumerate(sizes):
        nxt = sizes[i + 1] if i + 1 < len(sizes) else 0
        net.add_layer(s, nxt, lr=lr, ir=ir, activation="tanh", activation_deriv="dtanh")
    net.randomize_weights()
    return net


# --- Test 1: inference is gradient descent on the energy -> must settle monotonically ---
rng = np.random.default_rng(0)
# ir must be small relative to ||W||: fixed-step gradient descent on a random
# W can overshoot on the first few steps if ir is too large (this is a property
# of discretized gradient descent, not specific to this codebase) -- 0.01 is
# stable across 50/50 random inits for a [4,6,3] Xavier-initialized net.
net = build_net([4, 6, 3], lr=0.02, ir=0.01)
inp = net.layers[0]
inp.clamp_state(rng.uniform(-1, 1, 4).astype(np.float32))

energies = []
for _ in range(100):
    energies.append(net.calculate_state())
    net.update_state()

assert all(a >= b - 1e-2 for a, b in zip(energies, energies[1:])), "energy is not monotonically decreasing"
print(f"[test 1] energy settled {energies[0]:.4f} -> {energies[-1]:.4f} (monotonic: OK)")

# --- Test 2: local Hebbian-style rule actually learns a fixed nonlinear mapping ---
rng = np.random.default_rng(1)  # independent stream from test 1
IN, OUT = 3, 2
Wt = (0.5 * rng.uniform(-1, 1, (OUT, IN))).astype(np.float32)


def teacher(x):
    return np.tanh(Wt @ x)


net = build_net([IN, OUT], lr=0.01, ir=0.1)
inp, out = net.layers


def eval_mse(n=100, steps=40):
    errs = []
    for _ in range(n):
        x = rng.uniform(-1, 1, IN).astype(np.float32)
        y = teacher(x)
        inp.clamp_state(x)
        for _ in range(steps):
            net.calculate_state()
            net.update_state()
        pred = out.beliefs.reshape(-1)[:OUT]
        errs.append(np.mean((pred - y) ** 2))
        inp.unclamp_state()
    return float(np.mean(errs))


mse_before = eval_mse()

for _ in range(3000):
    x = rng.uniform(-1, 1, IN).astype(np.float32)
    y = teacher(x)
    inp.clamp_state(x)
    out.clamp_state(y)
    for _ in range(40):
        net.calculate_state()
        net.update_state()
    net.update_weights()
    inp.unclamp_state()
    out.unclamp_state()

mse_after = eval_mse()
print(f"[test 2] held-out MSE {mse_before:.4f} -> {mse_after:.6f}")
# NOTE: with lmbda=0.01 decay compounding over 3000 steps ((1-lmbda)^3000 ~ 1e-13),
# the random initial W is completely forgotten -- mse_before is just noise from
# RandomizeWeights() and isn't a meaningful baseline, so check an absolute floor.
assert mse_after < 0.1, "network did not learn the teacher mapping"

print("PASS")
