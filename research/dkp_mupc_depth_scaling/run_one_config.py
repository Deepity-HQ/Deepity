"""Single (mode, steps, fl0) config on full MNIST, for parallel dispatch."""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pc_mlp import DeepPCMLP
from run_experiment import load_mnist

mode, steps, fl0, epochs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
fl_decay_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 0
psi_optimizer = sys.argv[7] if len(sys.argv) > 7 else "fixed"
lmbda = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0

X_train, Y_train, X_test, y_test = load_mnist(n_train=60000, n_test=10000)
n = X_train.shape[0]
batch_size = 200

net = DeepPCMLP(784, [128] * 24, 10, lr=0.001, ir=0.1, fl0=fl0, psi_mode=mode, seed=seed,
                fl_decay_steps=fl_decay_steps, psi_optimizer=psi_optimizer, lmbda=lmbda)
step, blew = 0, False
t0 = time.perf_counter()
for epoch in range(epochs):
    perm = np.random.default_rng(seed * 1000 + epoch).permutation(n)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        loss, _, _ = net.train_step(X_train[idx], Y_train[idx], steps)
        step += 1
        if not net.is_finite() or not np.isfinite(loss):
            blew = True
            break
    if blew:
        break
    preds = net.predict(X_test[:2000]).argmax(axis=1)
    acc = (preds == y_test[:2000]).mean()
    print(f"[{mode} steps={steps} fl0={fl0}] epoch {epoch} step {step} acc {acc*100:.2f}%", flush=True)

elapsed = time.perf_counter() - t0
tag = f"{mode}_s{steps}_fl{fl0}_decay{fl_decay_steps}_psi{psi_optimizer}_lmbda{lmbda}_seed{seed}"
if blew:
    print(f"RESULT {tag} BLEW_UP step={step} elapsed={elapsed:.1f}s", flush=True)
else:
    preds = net.predict(X_test).argmax(axis=1)
    acc = (preds == y_test).mean()
    print(f"RESULT {tag} acc={acc*100:.2f}% step={step} elapsed={elapsed:.1f}s", flush=True)
