"""Can DKP's O(1) direct feedback survive at depth if `fl` is given a
muPC-style depth-aware scale, instead of being disabled outright?

Two sweeps on MNIST, numpy-only (fast iteration before touching the real
C++ engine):
1. Depth sweep at fixed settle steps -- does `scale_by_a` stay stable
   where `fixed` (the original bug) destabilizes?
2. Settle-step sweep at fixed depth -- does DKP let a deep model reach
   good accuracy with FEWER settling steps than muPC alone (`off`)?
"""
import gzip
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pc_mlp import DeepPCMLP

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_mnist(n_train: int, n_test: int):
    def read(name, offset):
        with gzip.open(os.path.join(DATA_DIR, name), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)

    X_train = read("train-images-idx3-ubyte.gz", 16).reshape(-1, 784)[:n_train].astype(np.float32) / 255.0
    y_train = read("train-labels-idx1-ubyte.gz", 8)[:n_train]
    X_test = read("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784)[:n_test].astype(np.float32) / 255.0
    y_test = read("t10k-labels-idx1-ubyte.gz", 8)[:n_test]

    eps = 0.001
    Y_train = np.full((n_train, 10), eps, dtype=np.float32)
    Y_train[np.arange(n_train), y_train] = 1.0 - eps
    return X_train, Y_train, X_test, y_test


def run(psi_mode: str, depth: int, hidden: int, steps: int, lr: float, ir: float, fl0: float,
        X_train, Y_train, X_test, y_test, batch_size: int, epochs: int, seed: int):
    net = DeepPCMLP(784, [hidden] * depth, 10, lr=lr, ir=ir, fl0=fl0,
                     psi_mode=psi_mode, seed=seed)

    n = X_train.shape[0]
    t0 = time.perf_counter()
    blew_up = False
    for _ in range(epochs):
        perm = np.random.default_rng(seed).permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss, _, _ = net.train_step(X_train[idx], Y_train[idx], steps)
            if not net.is_finite() or not np.isfinite(loss):
                blew_up = True
                break
        if blew_up:
            break
    elapsed = time.perf_counter() - t0

    if blew_up:
        return {"mode": psi_mode, "depth": depth, "steps": steps, "acc": float("nan"),
                "blew_up": True, "seconds": elapsed}

    preds = net.predict(X_test).argmax(axis=1)
    acc = float((preds == y_test).mean())
    return {"mode": psi_mode, "depth": depth, "steps": steps, "acc": acc,
            "blew_up": False, "seconds": elapsed}


def main():
    X_train, Y_train, X_test, y_test = load_mnist(n_train=10000, n_test=2000)
    common = dict(hidden=128, lr=0.02, ir=0.1, fl0=1e-2, batch_size=200, epochs=3, seed=7)

    print("=== Sweep 1: depth x psi_mode, fixed settle steps=10 ===")
    print(f"{'mode':<14}{'depth':>6}{'steps':>7}{'acc':>8}{'blew_up':>9}{'sec':>8}")
    for depth in (8, 16, 24):
        for mode in ("off", "fixed", "scale_by_a", "scale_by_a2"):
            r = run(mode, depth, steps=10, X_train=X_train, Y_train=Y_train,
                    X_test=X_test, y_test=y_test, **common)
            print(f"{r['mode']:<14}{r['depth']:>6}{r['steps']:>7}"
                  f"{r['acc']*100 if not r['blew_up'] else float('nan'):>7.2f}%"
                  f"{str(r['blew_up']):>9}{r['seconds']:>8.1f}")

    print("\n=== Sweep 2: settle steps x psi_mode, fixed depth=24 ===")
    print(f"{'mode':<14}{'depth':>6}{'steps':>7}{'acc':>8}{'blew_up':>9}{'sec':>8}")
    for steps in (3, 5, 10, 25):
        for mode in ("off", "scale_by_a"):
            r = run(mode, 24, steps=steps, X_train=X_train, Y_train=Y_train,
                    X_test=X_test, y_test=y_test, **common)
            print(f"{r['mode']:<14}{r['depth']:>6}{r['steps']:>7}"
                  f"{r['acc']*100 if not r['blew_up'] else float('nan'):>7.2f}%"
                  f"{str(r['blew_up']):>9}{r['seconds']:>8.1f}")


if __name__ == "__main__":
    main()
