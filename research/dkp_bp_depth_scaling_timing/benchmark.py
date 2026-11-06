"""Does DKP-PC's per-step wall-clock time grow with network depth L slower
than PyTorch backprop's, as the DKP-PC paper claims ("for sufficiently
deep networks... DKP-PC will be significantly lower than that of BP")?

Today, Deepity's DirectKPPCNetwork::TrainStep() dispatches every phase
(DirectFeedbackUpdate, the inference step, UpdateWeights) as a plain
sequential C++ for-loop over layers -- O(L) dispatch, not the O(1)
cross-layer parallel dispatch the paper's claim actually depends on. This
measures where that leaves things today, before any such parallelization
is built: does DKP-PC (at inference_steps=1, matching the paper's own
Table 1 finding that one step suffices) already scale better than BP just
from doing less total sequential work per step, or does it scale the same
or worse?

Synthetic random data, CPU only, timing only -- no accuracy claim here.
Run with OPENBLAS_NUM_THREADS=1 once Step() is parallelized across layers
via OpenMP, to avoid nested-parallelism oversubscription between the new
per-layer OMP region and OpenBLAS's own internal thread pool.
"""
import os

os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "300000000")

import time

import numpy as np
import torch
import torch.nn as nn

from deepity import DKPPCN
from deepity.layer import Linear, Sigmoid

IN_DIM = 64
HIDDEN = 64
OUT_DIM = 10
BATCH = 128
WARMUP = 3
TIMED = 10
DEPTHS = [2, 4, 8, 16, 32, 64, 96]


def build_dkppcn(depth: int) -> DKPPCN:
    layers = [Linear(IN_DIM, HIDDEN)]
    for _ in range(depth - 1):
        layers += [Sigmoid(), Linear(HIDDEN, HIDDEN)]
    layers += [Sigmoid(), Linear(HIDDEN, OUT_DIM)]
    net = DKPPCN(*layers, batch_size=BATCH, device="cpu")
    net.configure(
        learning_rate=1e-3,
        inference_rate=0.1,
        feedback_rate=1e-3,
        lmbda=1e-5,
        optimizer="ADAM",
        psi_optimizer="ADAM",
    )
    return net


class BPMlp(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        layers = [nn.Linear(IN_DIM, HIDDEN)]
        for _ in range(depth - 1):
            layers += [nn.Sigmoid(), nn.Linear(HIDDEN, HIDDEN)]
        layers += [nn.Sigmoid(), nn.Linear(HIDDEN, OUT_DIM)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def time_dkppcn(depth: int, x: np.ndarray, y: np.ndarray) -> float:
    net = build_dkppcn(depth)
    for _ in range(WARMUP):
        net.train_step(x, y, 1)
    t0 = time.perf_counter()
    for _ in range(TIMED):
        net.train_step(x, y, 1)
    return (time.perf_counter() - t0) / TIMED


def time_bp(depth: int, x: np.ndarray, y: np.ndarray) -> float:
    torch.manual_seed(7)
    model = BPMlp(depth)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)

    def step():
        opt.zero_grad()
        loss = crit(model(xt), yt)
        loss.backward()
        opt.step()

    for _ in range(WARMUP):
        step()
    t0 = time.perf_counter()
    for _ in range(TIMED):
        step()
    return (time.perf_counter() - t0) / TIMED


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH, IN_DIM)).astype(np.float32)
    y_idx = rng.integers(0, OUT_DIM, size=BATCH)
    y = np.eye(OUT_DIM, dtype=np.float32)[y_idx]

    print(f"IN_DIM={IN_DIM} HIDDEN={HIDDEN} OUT_DIM={OUT_DIM} BATCH={BATCH}\n")
    print(f"{'depth':>6}{'dkppcn_ms':>12}{'bp_ms':>10}{'dkp/bp':>10}")

    results = []
    for depth in DEPTHS:
        dkp_s = time_dkppcn(depth, x, y)
        bp_s = time_bp(depth, x, y)
        results.append((depth, dkp_s, bp_s))
        print(f"{depth:>6}{dkp_s * 1e3:>12.3f}{bp_s * 1e3:>10.3f}{dkp_s / bp_s:>10.3f}")

    d0, dkp0, bp0 = results[0]
    dN, dkpN, bpN = results[-1]
    print(f"\nGrowth from depth={d0} to depth={dN} ({dN / d0:.1f}x more layers):")
    print(f"  DKP-PC: {dkpN / dkp0:.2f}x slower")
    print(f"  BP:     {bpN / bp0:.2f}x slower")


if __name__ == "__main__":
    main()
