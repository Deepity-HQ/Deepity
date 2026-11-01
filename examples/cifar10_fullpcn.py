"""
CIFAR-10 on FullPCN, at real depth -- built to answer the question Tutorial
6 (muPC, iPC, and ePC Explained) asks but can't show without a real run:
does a plain, toggle-off FullPCLayer stack actually survive depths that
muPC + residual connections were specifically designed for?

Builds a fixed-width stack of --depth constant-width Linear+ReLU hidden
layers (default 24), the shape muPC's own residual-connection rule
requires (every middle hidden layer must match its neighbor's width).

Usage, for the A/B comparison Tutorial 6 describes:
    python examples/cifar10_fullpcn.py                # plain FullPCLayer
    python examples/cifar10_fullpcn.py --mu-pc         # + muPC scaling/init + residual connections

Same architecture, same hyperparameters, same seed, same data order in
both runs -- --mu-pc is the only thing that changes, so the two runs'
energy/accuracy curves isolate exactly what that pairing buys at this
depth. --mu-pc toggles use_mu_pc_scaling and use_residual_connections
together, matching how the muPC paper presents them; each is actually an
independent configure() kwarg if you want to split them.
"""
import os

# See mnist.py's identical comment: keeps OpenMP worker threads spinning
# instead of sleeping between this engine's many short parallel regions
# per settling step. Must run before any import that can pull in libgomp.
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "300000000")

import argparse
import pickle
import sys
import tarfile
import urllib.request
from time import perf_counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from deepity import FullPCN  # noqa: E402
from deepity.layer import Linear, ReLU  # noqa: E402


def load_full_cifar10():
    print("Fetching CIFAR-10 (python pickle format)...")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    # data_batch_5 specifically, not just extract_dir's existence: a
    # truncated download (seen in practice -- urlretrieve doesn't always
    # raise on a cut-off connection) extracts a partial archive without
    # error, silently leaving only some of the 5 train batches behind,
    # which a directory-existence check alone would wrongly call done.
    last_batch = os.path.join(extract_dir, "data_batch_5")

    if not os.path.exists(last_batch):
        if os.path.exists(tar_path) and os.path.getsize(tar_path) < 160_000_000:
            os.remove(tar_path)  # remove a previous truncated download
        if not os.path.exists(tar_path):
            print("Downloading CIFAR-10 (~163MB)...")
            urllib.request.urlretrieve(url, tar_path)
        print("Extracting CIFAR-10...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
        if not os.path.exists(last_batch):
            raise RuntimeError(
                f"CIFAR-10 extraction incomplete: {last_batch} still missing after "
                "extracting -- the download is likely truncated again, delete "
                f"{tar_path} and {extract_dir} and retry.")

    def load_batch(fname):
        with open(fname, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        X = d[b"data"].astype(np.float32) / 255.0
        y = np.array(d[b"labels"], dtype=np.int64)
        return X, y

    X_parts, y_parts = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(extract_dir, f"data_batch_{i}"))
        X_parts.append(X)
        y_parts.append(y)
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)

    X_test, y_test_labels = load_batch(os.path.join(extract_dir, "test_batch"))

    eps = 0.001
    Y_train = np.full((y_train.shape[0], 10), eps, dtype=np.float32)
    Y_train[np.arange(y_train.shape[0]), y_train] = 1.0 - eps

    return X_train, Y_train, X_test, y_test_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu-pc", action="store_true",
                        help="enable use_mu_pc_scaling + use_residual_connections")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=25, help="settling steps per batch")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=256, help="constant hidden width")
    parser.add_argument("--depth", type=int, default=24,
                        help="number of constant-width hidden Linear+ReLU layers between "
                             "the input and output layers")
    # lr=0.005/ir=0.1 (mnist_fullpcn.py's 2-layer defaults) diverged to NaN
    # energy at this depth -- WITH muPC too, not just without it. muPC's
    # scaling/init fixes the forward pass's MAGNITUDE at a point in time;
    # it says nothing about ir, the settling loop's Euler-integration step
    # size, which has its own, smaller stability threshold as more layers
    # couple together. Lower ir, more steps to compensate for the smaller
    # per-step movement, and a correspondingly smaller lr.
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--ir", type=float, default=0.01)
    # fl=0 is a clean, total disable of DKP's always-on direct-feedback
    # mechanism: it zeroes the DirectFeedbackUpdate() GEMM's scale factor
    # (the straight injection into W) AND Psi's own Adam learning rate
    # (freezing Psi at its random, fan-scaled, never-muPC-scaled init) --
    # see FullPCLayer::DirectFeedbackUpdate()/UpdateWeights()'s Psi branch.
    parser.add_argument("--fl", type=float, default=1e-4)
    parser.add_argument("--acc-batches", type=int, default=10)
    args = parser.parse_args()

    X_train, Y_train, X_test, y_test_labels = load_full_cifar10()

    layers = [Linear(3072, args.hidden), ReLU()]
    for _ in range(args.depth):
        layers += [Linear(args.hidden, args.hidden), ReLU()]
    layers += [Linear(args.hidden, 10)]

    total_weight_layers = args.depth + 2
    print(f"\nBuilding FullPCN (3072 -> {args.hidden} x{args.depth} -> 10), "
          f"{total_weight_layers} weight-bearing layers, mu_pc={args.mu_pc}, seed={args.seed}...")

    net = FullPCN(*layers, batch_size=args.batch_size)
    net.configure(
        learning_rate=args.lr,
        inference_rate=args.ir,
        feedback_rate=args.fl,
        lmbda=0.0,
        optimizer="ADAM",
        use_cross_entropy=True,
        use_mu_pc_scaling=args.mu_pc,
        use_residual_connections=args.mu_pc,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    n_batches = len(X_train) // args.batch_size
    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(args.epochs):
        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        epoch_energy = 0.0
        for b in range(n_batches):
            X_batch = X_shuf[b * args.batch_size:(b + 1) * args.batch_size]
            Y_batch = Y_shuf[b * args.batch_size:(b + 1) * args.batch_size]
            epoch_energy += net.train_step(X_batch, Y_batch, args.steps)

        correct, total = 0, 0
        for b in range(min(args.acc_batches, n_batches)):
            X_batch = X_shuf[b * args.batch_size:(b + 1) * args.batch_size]
            Y_batch = Y_shuf[b * args.batch_size:(b + 1) * args.batch_size]
            pred = net.predict(X_batch, args.steps)
            correct += np.sum(np.argmax(pred, axis=1) == np.argmax(Y_batch, axis=1))
            total += args.batch_size

        epoch_acc = 100.0 * correct / total
        epoch_accs.append(epoch_acc)
        elapsed = perf_counter() - start_time
        avg_energy = epoch_energy / n_batches
        print(f"Epoch {epoch + 1}/{args.epochs} | Time: {elapsed:.1f}s | "
              f"Acc: {epoch_acc:.2f}% | Avg energy: {avg_energy:.4f}")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    print("\nRunning final test evaluation...")
    correct, total = 0, 0
    for i in range(0, len(X_test), args.batch_size):
        X_batch = X_test[i:i + args.batch_size]
        y_labels_batch = y_test_labels[i:i + args.batch_size]
        if len(X_batch) != args.batch_size:
            continue
        pred = net.predict(X_batch, args.steps)
        correct += np.sum(np.argmax(pred, axis=1) == y_labels_batch)
        total += args.batch_size

    test_acc = 100.0 * correct / total
    print(f"\n=== Result (mu_pc={args.mu_pc}, depth={total_weight_layers}) ===")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"Per-epoch: {[round(a, 2) for a in epoch_accs]}")


if __name__ == "__main__":
    main()
