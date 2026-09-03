"""
Short, dedicated workload for PGO profile collection -- run by
deepity_build/cli.py's --pgo pass, not meant to be invoked directly for
training. Deliberately NOT the full mnist.py run: PGO only needs to see
which code paths are hot (branch outcomes, call frequency), and the
settling loop's structure repeats identically on batch 1 and batch 234,
so a few dozen batches already captures the same information a full
15-epoch run would. This cost is paid on every --pgo build, so keeping
it short matters.

Matches the real network configuration from mnist.py exactly (same
architecture, same activations) so the instrumented binary actually
exercises the same code paths real training does -- a mismatched
architecture would profile the wrong thing.
"""
import numpy as np
import os
from pydeepity import SimplePCN

N_BATCHES = 20  # enough to exercise every activation type, both
                 # mu-cache branches (hit and miss), and the full
                 # settling loop repeatedly -- not aiming for
                 # convergence, just representative code-path coverage


def load_pgo_subset():
    import gzip
    import urllib.request

    print("PGO workload: fetching MNIST (cached after first run)...")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files: dict[str, str] = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
    }
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for key, fname in files.items():
        filepath = os.path.join(data_dir, fname)
        paths[key] = filepath
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + fname, filepath)

    with gzip.open(paths["x_train"], 'rb') as f:
        X_raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(paths["y_train"], 'rb') as f:
        y_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # Only load enough for N_BATCHES -- no need for the full 60k images
    BATCH_SIZE = 250
    needed = BATCH_SIZE * N_BATCHES
    X = X_raw[:needed].astype(np.float32) / 255.0
    y = y_labels[:needed]

    eps = 0.001
    Y = np.full((y.shape[0], 10), eps, dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0 - eps

    return X, Y, BATCH_SIZE


def main():
    X, Y, BATCH_SIZE = load_pgo_subset()

    # Matches mnist.py's real, current architecture exactly -- same
    # activations, so the same code paths (dSigmoidInto, dLinearInto,
    # etc.) actually get exercised during profiling.
    net = SimplePCN(batch_size=BATCH_SIZE)
    net.add_layer(784, 512, lr=0.00373, ir=0.08, act="linear")
    net.add_layer(512, 512, lr=0.00373, ir=0.08, act="sigmoid")
    net.add_layer(512, 10, lr=0.00373, ir=0.08, act="sigmoid")
    net.add_layer(10, 0, lr=0.00373, ir=0.08, act="linear")
    net.set_optimizer("ADAM")
    net.compile()
    net.randomize_weights()
    net.set_mu_cache_threshold(0.0)

    rng_init = np.random.default_rng(7)
    for layer in net.layers[:-1]:
        layer.weights[:] = rng_init.uniform(-0.3, 0.3, layer.weights.shape).astype(np.float32)

    print(f"PGO workload: running {N_BATCHES} batches, 30 steps each...")
    for b in range(N_BATCHES):
        X_batch = X[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        Y_batch = Y[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        net.train_step_with_projection(X_batch, Y_batch, 30)

    print("PGO workload complete.")


if __name__ == "__main__":
    main()
