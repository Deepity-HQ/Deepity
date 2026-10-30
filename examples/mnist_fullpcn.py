"""
MNIST on FullPCN, across storage dtypes -- built to answer one question:
does a quantized FullPCN (today: float32 vs. fp4_e2m1) train to a useful
accuracy at all? Not tuned for peak accuracy the way examples/mnist.py's
SimplePCN setup is (that one carries a float-only, heavily-tuned
forward-projection-init/weight-decay recipe that doesn't apply here --
FullPCLayer/FullPCNetwork is the one family of layers this project has
templated over storage type so far; SimplePCLayer has not been).

Usage:
    python examples/mnist_fullpcn.py --dtype float32
    python examples/mnist_fullpcn.py --dtype fp4_e2m1 --ir 4.0

Run both with the same --seed to get the apples-to-apples comparison:
identical architecture, hyperparameters, batch order -- the only
difference is what FullPCN quantizes W/b/Psi down to internally. NOTE:
fp4_e2m1 needs a much larger --ir than float32's default (0.1) to train
at all -- see --ir's own help text for why. Expect noisier, less stable
convergence than float32 at that larger --ir: this is a real property of
settling on a coarse 4-bit grid, not a leftover bug.
"""
import argparse
import os
import sys
from time import perf_counter

os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "300000000")

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mnist import load_full_mnist  # noqa: E402

from deepity import FullPCN  # noqa: E402
from deepity.layer import Linear, Sigmoid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "fp4_e2m1"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=20, help="settling steps per batch")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--ir", type=float, default=0.1, help="settling/inference rate; "
                        "fp4_e2m1 needs this much larger than float32's default to ever cross "
                        "a grid boundary at all -- see examples/mnist_fullpcn.py's own history "
                        "for why (an activation derivative, and everything computed from it, is "
                        "never itself quantized, but the settling step still writes its result "
                        "into a quantized belief, so a per-step delta smaller than the grid's "
                        "local spacing is silently discarded, same as a too-small gradient would "
                        "be discarded updating a quantized weight)")
    parser.add_argument("--acc-batches", type=int, default=10,
                        help="train-set batches sampled for the per-epoch accuracy readout")
    args = parser.parse_args()

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    print(f"\nBuilding FullPCN(dtype={args.dtype!r}) (784->{args.hidden}->10), "
          f"seed={args.seed}...")
    net = FullPCN(
        Linear(784, args.hidden),
        Sigmoid(),
        Linear(args.hidden, 10),
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    net.configure(
        learning_rate=args.lr,
        inference_rate=args.ir,
        feedback_rate=1e-4,
        lmbda=0.0,  # see examples/mnist.py's own comment: ADAMW+lmbda>0 destabilizes this family
        optimizer="ADAM",
        use_cross_entropy=True,
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

        # Accuracy readout on a sampled subset of train batches (same
        # tradeoff as examples/mnist.py: cheap per-epoch signal, not the
        # final number -- that's the held-out test pass after the loop).
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
    print(f"\n=== Result (dtype={args.dtype}) ===")
    print(f"Peak Test Accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"Per-epoch: {[round(a, 2) for a in epoch_accs]}")


if __name__ == "__main__":
    main()
