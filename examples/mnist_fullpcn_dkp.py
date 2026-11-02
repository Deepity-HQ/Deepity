"""
MNIST on FullPCN at real depth, muPC alone vs. muPC + DKP -- the A/B this
project's depth-scaling investigation (see research/dkp_mupc_depth_scaling/)
was built to answer: once DKP's feedback injection is correctly scaled by
muPC's own per-layer factor `a`, and Psi is trained with real
Kolen-Pollack (ADAMW sharing W's weight decay, not a fixed or plain-SGD
feedback matrix), does DKP let a deep residual-muPC stack settle in fewer
steps per batch without sacrificing accuracy?

Usage, for the A/B comparison:
    python examples/mnist_fullpcn_dkp.py --mode off    # muPC + residual connections alone
    python examples/mnist_fullpcn_dkp.py --mode dkp    # + DKP, correctly scaled and KP-trained

Same architecture, same seed, same data order in both modes -- --mode is
the only thing that changes (and --steps, since fewer settling steps per
batch is the whole point of --mode dkp; override it explicitly to hold
steps fixed instead). --feedback-rate/--lmbda default to the values
confirmed stable over a full 40-epoch run at this depth/width (84.17% acc,
2.07x faster than --mode off's 81.68%) -- retune them if you change
--depth/--hidden, there's no derived closed-form for them the way there
is for muPC's own `a`. lmbda here is DKP's own raw, lr-independent weight
decay (W *= 1-lmbda, gating the undecayed feedback-injection accumulate
into W) -- unrelated in scale to an SGD optimizer's lmbda elsewhere, and
much smaller: it only needs to bound an injection, not drive learning.
"""
import os
import sys
from time import perf_counter

os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "300000000")

import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mnist import load_full_mnist  # noqa: E402

from deepity import FullPCN  # noqa: E402
from deepity.layer import Linear, ReLU  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="dkp", choices=["off", "dkp"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--depth", type=int, default=24, help="weight-bearing layer count")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--ir", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=None,
                        help="settling steps per batch; defaults to 25 for --mode off, "
                        "10 for --mode dkp (fewer steps is DKP's whole point)")
    parser.add_argument("--feedback-rate", type=float, default=0.01)
    parser.add_argument("--lmbda", type=float, default=1e-5)
    parser.add_argument("--acc-batches", type=int, default=10)
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else (10 if args.mode == "dkp" else 25)

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    print(f"\nBuilding FullPCN mode={args.mode!r} (784->[{args.hidden}]*{args.depth - 1}->10), "
          f"steps={steps}, seed={args.seed}...")
    layers = [Linear(784, args.hidden), ReLU()]
    for _ in range(args.depth - 2):
        layers += [Linear(args.hidden, args.hidden), ReLU()]
    layers += [Linear(args.hidden, 10)]

    net = FullPCN(*layers, batch_size=args.batch_size)
    if args.mode == "off":
        net.configure(
            learning_rate=args.lr, inference_rate=args.ir, feedback_rate=0.0, lmbda=0.0,
            optimizer="ADAM", psi_optimizer="SGD",
            use_mu_pc_scaling=True, use_residual_connections=True, use_cross_entropy=True,
            seed=args.seed,
        )
    else:
        net.configure(
            learning_rate=args.lr, inference_rate=args.ir,
            feedback_rate=args.feedback_rate, lmbda=args.lmbda,
            optimizer="ADAMW", psi_optimizer="ADAMW",
            use_mu_pc_scaling=True, use_residual_connections=True, use_cross_entropy=True,
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
            energy = net.train_step(X_batch, Y_batch, steps)
            epoch_energy += energy
            if not np.isfinite(energy):
                print(f"BLEW UP at epoch {epoch} batch {b}")
                return

        correct, total = 0, 0
        for b in range(min(args.acc_batches, n_batches)):
            X_batch = X_shuf[b * args.batch_size:(b + 1) * args.batch_size]
            Y_batch = Y_shuf[b * args.batch_size:(b + 1) * args.batch_size]
            pred = net.predict(X_batch, steps)
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
        pred = net.predict(X_batch, steps)
        correct += np.sum(np.argmax(pred, axis=1) == y_labels_batch)
        total += args.batch_size

    test_acc = 100.0 * correct / total
    print(f"\n=== Result (mode={args.mode}, steps={steps}, depth={args.depth}) ===")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"Per-epoch: {[round(a, 2) for a in epoch_accs]}")


if __name__ == "__main__":
    main()
