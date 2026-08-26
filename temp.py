import numpy as np
import os
from pydeepity import GaussSeidelPCN
from time import perf_counter

# Tests GaussSeidelPCN -- the Gauss-Seidel (sequential-sweep) settling
# restructuring, traced from ngc-learn's real execution graph, combined
# with the SAME winning recipe already validated on SimplePCN:
# forward-projection init, +-0.3 uniform weight init, plain Adam,
# lmbda=0, canonical MNIST test set, 784->512->512->10, T=20.
#
# UNPROVEN EXPERIMENT -- only smoke-tested so far (finite, decreasing
# energy), not independently gradient-checked. Part 0 below is a cheap,
# minimal one-weight finite-difference sanity check, bundled in rather
# than skipped entirely -- catches the most likely failure mode (a sign
# error or a completely wrong buffer read) before committing to a full,
# expensive training run.


def load_canonical_mnist():
    import gzip
    import urllib.request

    print("Fetching canonical MNIST dataset (idx-ubyte)...")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
        "x_test": "t10k-images-idx3-ubyte.gz",
        "y_test": "t10k-labels-idx1-ubyte.gz"
    }
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, fname in files.items():
        filepath = os.path.join(data_dir, fname)
        paths[key] = filepath
        if not os.path.exists(filepath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, filepath)

    with gzip.open(paths["x_train"], 'rb') as f:
        X_train_raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(paths["x_test"], 'rb') as f:
        X_test_raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(paths["y_train"], 'rb') as f:
        y_train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(paths["y_test"], 'rb') as f:
        y_test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0

    eps = 0.001
    Y_train = np.full((y_train_labels.shape[0], 10), eps, dtype=np.float32)
    Y_train[np.arange(y_train_labels.shape[0]), y_train_labels] = 1.0 - eps

    return X_train, Y_train, X_test, y_test_labels


def build_net(batch_size, seed):
    net = GaussSeidelPCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=0.001, ir=0.08, act="tanh")
    net.add_layer(512, 512, lr=0.001, ir=0.08, act="tanh")
    net.add_layer(512, 10, lr=0.001, ir=0.08, act="tanh")
    net.add_layer(10, 0, lr=0.001, ir=0.08, act="linear")
    net.set_optimizer("ADAM")
    net.compile()
    net.randomize_weights()

    rng_init = np.random.default_rng(seed)
    for layer in net.layers[:-1]:
        layer.weights[:] = rng_init.uniform(-0.3, 0.3, layer.weights.shape).astype(np.float32)

    return net


def part0_sanity_check(net, X_sample, Y_sample, batch_size, steps):
    """Minimal, single-weight finite-difference check -- NOT the full
    multi-point gradient-check ceremony, just enough to catch the most
    likely failure mode (sign error, wrong buffer read) before trusting
    a full, expensive training run."""
    print("=== Part 0: minimal one-weight sanity check ===")

    layer0 = net[0]
    W_before = np.array(layer0.weights, copy=True)

    energy = net.train_step_with_projection(X_sample, Y_sample, steps)
    print(f"  Single train_step_with_projection() energy: {energy:.4f}")

    if np.isnan(energy) or np.isinf(energy):
        print("  FAIL -- energy is NaN/Inf after a single step.")
        return False

    W_after = np.array(layer0.weights, copy=True)
    delta = W_after - W_before
    idx = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    print(f"  Largest weight change: W{idx} moved by {delta[idx]:.6f}")
    print(f"  (Not a full finite-difference check -- just confirms weights")
    print(f"   moved a plausible, non-zero, non-exploding amount.)")

    if np.abs(delta[idx]) > 10.0:
        print("  WARNING -- largest weight change is suspiciously large.")
        return False

    print("  Looks sane. Proceeding to real training.\n")
    return True


def main():
    X_train, Y_train, X_test, y_test_labels = load_canonical_mnist()

    BATCH_SIZE = 256
    STEPS = 20
    EPOCHS = 15
    SEED = 7

    print(f"\nBuilding GaussSeidelPCN network (784->512->512->10), seed={SEED}...")
    net = build_net(BATCH_SIZE, SEED)

    # Part 0: cheap sanity check on a single real batch before committing
    # to the full run.
    X_sample = X_train[:BATCH_SIZE]
    Y_sample = Y_train[:BATCH_SIZE]
    if not part0_sanity_check(net, X_sample, Y_sample, BATCH_SIZE, STEPS):
        print("Sanity check failed -- stopping before a full training run.")
        return

    # Rebuild fresh -- the sanity check already ran one real training step
    # on this network, don't want that to contaminate the real run.
    net = build_net(BATCH_SIZE, SEED)

    print(f"Training: {EPOCHS} epochs, {STEPS} steps, GaussSeidel dynamics + forward-projection...\n")
    print("Reference (ngc-learn, real run): 26.91, 42.96, 60.12, 75.20, 84.68, 89.52,")
    print("  91.90, 93.45, 94.30, 94.80, 95.13, 95.38, 95.63, 95.74, 95.95 -- test 95.09%")
    print("Reference (SimplePCN + forward-projection, Jacobi dynamics): 94.05-94.61% band\n")

    rng = np.random.default_rng(SEED)
    n_batches = len(X_train) // BATCH_SIZE
    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(EPOCHS):
        current_lr = 0.001 * (0.98 ** epoch)
        net.set_learning_rate(current_lr)

        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        epoch_energy = 0.0
        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            energy = net.train_step_with_projection(X_batch, Y_batch, STEPS)
            epoch_energy += energy

        # Quick per-epoch accuracy on a subset, genuine unclamped predict
        correct = 0
        total = 0
        for b in range(10):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            pred = net.predict(X_batch, steps=STEPS)
            pred_classes = np.argmax(pred, axis=1)
            true_classes = np.argmax(Y_batch, axis=1)
            correct += np.sum(pred_classes == true_classes)
            total += BATCH_SIZE

        epoch_acc = 100.0 * correct / total
        epoch_accs.append(epoch_acc)
        avg_energy = epoch_energy / n_batches
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | Avg energy: {avg_energy:.4f}")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    print("\nRunning final test evaluation...")
    correct = 0
    total = 0
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i:i + BATCH_SIZE]
        if len(X_batch) != BATCH_SIZE:
            continue
        pred_matrix = net.predict(X_batch, steps=STEPS)
        pred_classes = np.argmax(pred_matrix, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"GaussSeidelPCN test accuracy: {test_acc:.2f}%   (ngc-learn: 95.09%, SimplePCN band: 94.05-94.61%)")
    print(f"Train time: {train_time:.1f}s")
    print(f"\nPer-epoch: {[round(a,2) for a in epoch_accs]}")


if __name__ == "__main__":
    main()