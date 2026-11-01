"""
CIFAR-10 on FullConvPCNetwork -- the convolutional sibling of FullPCLayer
(same muPC scaling, residual connections, iPC, ePC toggles), which turned
out to already exist in this codebase despite cifar10_fullpcn.py's dense
stack having been the first (wrong) attempt at testing those ideas on
CIFAR-10. Spatial weight sharing is what actually gets CIFAR-10 accuracy
within reach of backprop -- no amount of tuning a dense stack closes that
gap, since the paper's own dense-muPC CIFAR10 result (Fig. A.19, a
128-layer dense ResNet) tops out around 35-37% for exactly that reason.

No high-level Python wrapper exists yet for FullConvPCNetwork (unlike
FullPCN's wrapper around FullPCNetworkT) -- this talks to the raw bound
class via deepity._backend.dy directly, which means every layer's
in_height/in_width must be computed and passed explicitly (no
input_shape= auto-propagation the way ConvolutionalPCN's own wrapper
provides).

Usage, for the same A/B comparison cifar10_fullpcn.py's docstring describes:
    python examples/cifar10_fullconvpcn.py              # plain FullConvPCLayer
    python examples/cifar10_fullconvpcn.py --mu-pc       # + muPC scaling/init + residual connections
"""
import os

# See mnist.py's identical comment: keeps OpenMP worker threads spinning
# instead of sleeping between this engine's many short parallel regions
# per settling step. Must run before any import that can pull in libgomp.
# Conv's im2col/GEMM/col2im pipeline makes far more such calls per step
# than the dense path's one big GEMM, so missing this hurts conv harder.
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

from deepity._backend import dy  # noqa: E402


def load_full_cifar10():
    print("Fetching CIFAR-10 (python pickle format)...")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    last_batch = os.path.join(extract_dir, "data_batch_5")

    if not os.path.exists(last_batch):
        if os.path.exists(tar_path) and os.path.getsize(tar_path) < 160_000_000:
            os.remove(tar_path)
        if not os.path.exists(tar_path):
            print("Downloading CIFAR-10 (~163MB)...")
            urllib.request.urlretrieve(url, tar_path)
        print("Extracting CIFAR-10...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
        if not os.path.exists(last_batch):
            raise RuntimeError(
                f"CIFAR-10 extraction incomplete: {last_batch} still missing -- "
                f"delete {tar_path} and {extract_dir} and retry.")

    def load_batch(fname):
        with open(fname, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        # Each row is R(1024)+G(1024)+B(1024), channel-major -- already
        # NCHW-compatible as a straight reshape, no transpose needed.
        X = d[b"data"].astype(np.float32).reshape(-1, 3, 32, 32) / 255.0
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=12, help="settling steps per batch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=32, help="constant channel count")
    parser.add_argument("--depth", type=int, default=8,
                        help="number of constant-shape 3x3 conv+ReLU middle layers, "
                             "each at full 32x32 resolution and residual-eligible")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--ir", type=float, default=0.1)
    parser.add_argument("--fl", type=float, default=0.0,
                        help="DKP feedback rate; 0 disables it (see cifar10_fullpcn.py's own "
                             "note on why always-on unscaled DKP destabilizes muPC at depth)")
    parser.add_argument("--acc-batches", type=int, default=10)
    args = parser.parse_args()

    X_train, Y_train, X_test, y_test_labels = load_full_cifar10()

    C = args.channels
    net = dy.FullConvPCNetwork(args.batch_size, "cpu")

    # Stem: 3->C, 4x4/stride 4, 32x32 -> 8x8. Aggressive downsample up
    # front, BEFORE any residual-eligible layer -- Compile() requires
    # every middle layer (not layer 1) to match shape exactly under
    # --mu-pc, so the only place to shed resolution is here. A single
    # 32x32-resolution conv layer costs 16x what the same layer costs at
    # 8x8 (conv cost ~ spatial_size^2); keeping all `depth` middle layers
    # at 32x32 made one batch take ~10s (~130min/epoch), confirmed by
    # direct timing. This is the architecture every real deep CNN uses
    # too (a downsampling stem before the residual stack), not a hack.
    net.add_layer(3, C, 32, 32, 4, 4, 4, 4, 0, 0, 0,
                  args.lr, args.ir, args.fl, 1e-4, "relu", "drelu", 1, 1, 1, 1)

    # Middle layers: C->C, 3x3, same padding, 8x8 -> 8x8, constant shape
    # throughout so every one of them is residual-eligible.
    for _ in range(args.depth):
        net.add_layer(C, C, 8, 8, 3, 3, 1, 1, 1, 1, 0,
                      args.lr, args.ir, args.fl, 1e-4, "relu", "drelu", 1, 1, 1, 1)

    # Readout: C->10 via a kernel covering the full 8x8 feature map,
    # collapsing to a single spatial position (the conv analog of a dense
    # Linear(C, 10) head) -- same pattern mnist_convpcn.py's
    # ConvolutionalPCN example uses for its own classification head.
    net.add_layer(C, 10, 8, 8, 8, 8, 1, 1, 0, 0, 0,
                  args.lr, args.ir, args.fl, 1e-4, "relu", "drelu", 1, 1, 1, 1)

    # Terminal marker: out_channels=0, matching ConvolutionalPCN's own
    # convention (see mnist_convpcn.py's identical comment).
    net.add_layer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10,
                  args.lr, args.ir, args.fl, 1e-4, "relu", "drelu", 1, 1, 1, 1)

    total_weight_layers = args.depth + 2
    print(f"\nBuilding FullConvPCNetwork (3x32x32 -> {C}ch x{args.depth} -> 10), "
          f"{total_weight_layers} weight-bearing layers, mu_pc={args.mu_pc}, seed={args.seed}...")

    net.set_use_cross_entropy(True)
    net.set_use_mu_pc_scaling(args.mu_pc)
    net.set_use_residual_connections(args.mu_pc)
    net.set_optimizer("ADAM")
    net.set_psi_optimizer("ADAM")
    net.compile()
    net.randomize_weights(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    n_batches = len(X_train) // args.batch_size
    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(args.epochs):
        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        epoch_energy = 0.0
        for b in range(n_batches):
            X_batch = X_shuf[b * args.batch_size:(b + 1) * args.batch_size].ravel()
            Y_batch = Y_shuf[b * args.batch_size:(b + 1) * args.batch_size].ravel()
            epoch_energy += net.train_step(X_batch, Y_batch, args.steps, True)

        correct, total = 0, 0
        for b in range(min(args.acc_batches, n_batches)):
            X_batch = X_shuf[b * args.batch_size:(b + 1) * args.batch_size].ravel()
            Y_batch = Y_shuf[b * args.batch_size:(b + 1) * args.batch_size]
            pred = np.array(net.predict(X_batch, args.steps))
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
        X_batch = X_test[i:i + args.batch_size].ravel()
        y_labels_batch = y_test_labels[i:i + args.batch_size]
        if len(y_labels_batch) != args.batch_size:
            continue
        pred = np.array(net.predict(X_batch, args.steps))
        correct += np.sum(np.argmax(pred, axis=1) == y_labels_batch)
        total += args.batch_size

    test_acc = 100.0 * correct / total
    print(f"\n=== Result (mu_pc={args.mu_pc}, depth={total_weight_layers}) ===")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"Per-epoch: {[round(a, 2) for a in epoch_accs]}")


if __name__ == "__main__":
    main()
