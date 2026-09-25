import numpy as np
import os
import urllib.request
import gzip
from pydeepity import dy
from time import perf_counter


def load_mnist():
    print("Fetching MNIST...")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
    }
    os.makedirs("./data", exist_ok=True)
    paths = {}
    for key, fname in files.items():
        fp = os.path.join("./data", fname)
        paths[key] = fp
        if not os.path.exists(fp):
            urllib.request.urlretrieve(base_url + fname, fp)

    with gzip.open(paths["x_train"], "rb") as f:
        X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open(paths["y_train"], "rb") as f:
        y_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    eps = 0.001
    Y = np.full((y_labels.shape[0], 10), eps, dtype=np.float32)
    Y[np.arange(y_labels.shape[0]), y_labels] = 1.0 - eps
    return X, Y, y_labels


def run(use_ipc: bool, X, Y, y_labels, batch_size=250, epochs=5, inference_steps=4):
    net = dy.FullPCNetwork(batch_size=batch_size, device="gpu")
    net.add_layer(784, 256, 10, lr=0.00373, ir=0.15, fl=1e-3, lmbda=1e-4, activation="tanh", activation_deriv="dtanh")
    net.add_layer(256, 10, 10, lr=0.00373, ir=0.15, fl=1e-3, lmbda=1e-4, activation="linear", activation_deriv="dlinear")
    net.add_layer(10, 0, 10, lr=0.00373, ir=0.15, fl=1e-3, lmbda=1e-4, activation="linear", activation_deriv="dlinear")

    net.set_use_ipc(use_ipc)
    net.set_optimizer("ADAMW")
    net.set_psi_optimizer("ADAMW")
    net.compile()
    net.randomize_weights()

    n_batches = len(X) // batch_size
    rng = np.random.default_rng(0)
    start = perf_counter()

    energies = []
    for epoch in range(epochs):
        perm = rng.permutation(len(X))
        X_shuf, Y_shuf = X[perm], Y[perm]

        epoch_energy = 0.0
        for b in range(n_batches):
            xb = X_shuf[b * batch_size:(b + 1) * batch_size]
            yb = Y_shuf[b * batch_size:(b + 1) * batch_size]
            epoch_energy += net.train_step(xb, yb, inference_steps)

        avg_energy = epoch_energy / n_batches
        energies.append(avg_energy)
        print(f"  [{'iPC' if use_ipc else 'standard'}] epoch {epoch+1}/{epochs}: avg energy={avg_energy:.4f}")

        if not np.isfinite(avg_energy):
            print(f"  [{'iPC' if use_ipc else 'standard'}] NON-FINITE ENERGY -- STOPPING")
            return energies, -1.0

    elapsed = perf_counter() - start

    correct = 0
    total = 0
    for i in range(0, min(2000, len(X)), batch_size):
        xb = X[i:i + batch_size]
        if xb.shape[0] != batch_size:
            continue
        pred = net.predict(xb, inference_steps).reshape(batch_size, 10)
        pred_classes = np.argmax(pred, axis=1)
        correct += np.sum(pred_classes == y_labels[i:i + batch_size])
        total += batch_size

    acc = 100.0 * correct / total
    print(f"  [{'iPC' if use_ipc else 'standard'}] train time: {elapsed:.1f}s, quick acc check: {acc:.2f}%")
    return energies, acc


if __name__ == "__main__":
    X, Y, y_labels = load_mnist()

    print("\n=== Standard (useIPC=False) ===")
    energies_std, acc_std = run(False, X, Y, y_labels)

    print("\n=== iPC (useIPC=True) ===")
    energies_ipc, acc_ipc = run(True, X, Y, y_labels)

    print("\n=== Comparison ===")
    print(f"Standard final energy: {energies_std[-1]:.4f}, accuracy: {acc_std:.2f}%")
    print(f"iPC      final energy: {energies_ipc[-1]:.4f}, accuracy: {acc_ipc:.2f}%")

    identical = all(abs(a - b) < 1e-6 for a, b in zip(energies_std, energies_ipc))
    if identical:
        print("\nWARNING: energy trajectories are IDENTICAL -- useIPC may not be taking effect at all.")
    elif acc_ipc < 0 or acc_std < 0:
        print("\nFAIL: non-finite energy detected in at least one run.")
    else:
        print("\nOK: trajectories differ and both stayed finite -- iPC is doing SOMETHING distinct.")
        print("(Not a claim it's doing the CORRECT thing -- just that it's not a silent no-op or a crash.)")