import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SimplePCN
from time import perf_counter

# Tests forward-projection initialization -- the real, structural finding
# from reading ngc-learn's actual pcn_model.py source: EVERY batch, they
# run a pure feedforward pass through CURRENT weights first, and seed the
# settling loop's hidden states from THAT, not from zero. This is
# fundamentally different from (and likely much stronger than) I_avg's
# stale class-average caching -- fresh, per-example, current-weights,
# every single batch.
#
# Matches ngc-learn's real settings as closely as possible now that we
# have the actual source: [0,1] normalization, labels clipped to
# [0.001, 0.999] (not raw one-hot), tanh activation (confirmed as their
# actual default), 784->512->512->10, T=20, plain SGD (isolating THIS
# specific change -- forward-projection init -- from the separate,
# already-known Adam question).


def load_full_mnist():
    print("Fetching full MNIST dataset (70,000 images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = X.astype(np.float32) / 255.0  # [0,1], matching ngc-learn
    y = y.astype(int)

    # Clipped one-hot, matching ngc-learn's `jnp.clip(lab, eps, 1-eps)`
    eps = 0.001
    Y = np.full((y.shape[0], 10), eps, dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0 - eps

    X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = train_test_split(
        X, Y, y, train_size=60000, test_size=10000, stratify=y, random_state=42
    )
    return X_train, Y_train, X_test, y_test_labels


def main() -> None:
    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 256
    STEPS = 20            # matches ngc-learn's T=20
    EPOCHS = 15
    LR = 0.001             # Adam-appropriate scale, NOT the SGD-tuned 0.06 --
                             # matches ngc-learn's own stated eta=0.001 directly,
                             # and today's earlier finding that Adam needs a
                             # much smaller rate than SGD on this codebase
    DECAY_RATE = 0.98

    print(f"\nBuilding network (784->512->512->10)...")
    net = SimplePCN(batch_size=BATCH_SIZE)
    net.add_layer(784, 512, lr=LR, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(512, 512, lr=LR, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(512, 10, lr=LR, ir=0.08, act="tanh", lmbda=0.0001)
    # Explicit TERMINAL layer -- outChannels/nextSize=0. Without this,
    # net[-1] is the 512->10 layer itself, whose OWN beliefs are its
    # 512-dim INPUT, not the 10-dim prediction it produces -- exactly
    # the bug that just crashed this script (and was already silently
    # corrupting every batch before that: clamp_state() truncates rather
    # than erroring on a size mismatch).
    net.add_layer(10, 0, lr=LR, ir=0.08, act="linear", lmbda=0.0001)
    net.set_optimizer("ADAMW")  # THE missing paired piece -- ngc-learn hard-codes
                                  # Adam for every synapse. The larger +-0.3 init
                                  # tested ALONE (with SGD) made things WORSE
                                  # (73.31% vs 86.41%) -- likely because SGD's
                                  # step size scales with raw gradient magnitude,
                                  # unlike Adam's per-parameter normalization,
                                  # which can handle a larger weight scale
                                  # gracefully. Init range and optimizer were
                                  # tuned TOGETHER in the original; testing one
                                  # without the other may have been the mistake.
    net.compile()
    net.randomize_weights()

    # Override weight init to match ngc-learn's actual convention: fixed
    # uniform range +-0.3, INDEPENDENT of layer size -- not our own
    # size-scaled Gaussian (std ~0.039 for the 784->512 layer, ~8x
    # smaller). This directly affects how informative the very first
    # forward-projection pass is -- larger initial weights produce more
    # differentiated activations from a single forward pass. Biases
    # already default to 0 in Deepity (never touched by
    # RandomizeWeights()), matching ngc-learn's own bias_init=constant(0).
    rng_init = np.random.default_rng(7)
    for layer in net.layers[:-1]:  # skip the terminal -- it has no weights (nextSize=0)
        w_shape = layer.weights.shape
        layer.weights[:] = rng_init.uniform(-0.3, 0.3, w_shape).astype(np.float32)

    print(f"\nTraining with FORWARD-PROJECTION init: {EPOCHS} epochs, {STEPS} steps, "
          f"lr={LR}, decay_rate={DECAY_RATE}...\n")
    print("Reference (ngc-learn, real run): 26.91, 42.96, 60.12, 75.20, 84.68, 89.52,")
    print("  91.90, 93.45, 94.30, 94.80, 95.13, 95.38, 95.63, 95.74, 95.95 -- test 95.09%\n")

    rng = np.random.default_rng(42)
    n_batches = len(X_train) // BATCH_SIZE
    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(EPOCHS):
        current_lr = LR * (DECAY_RATE ** epoch)
        net.set_learning_rate(current_lr)

        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        correct = 0
        total = 0
        epoch_energy = 0.0

        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            # Manual loop instead of train_step() -- need ProjectForward()
            # inserted between clamp_input() and the settling loop, which
            # the existing train_step() convenience method doesn't do.
            net.reset_state()
            net.clamp_input(X_batch)
            net.project_forward()  # THE new step -- seeds hidden layers
                                     # from a genuine forward pass, not zero
            net[-1].clamp_state(Y_batch)

            energy = 0.0
            for _ in range(STEPS):
                energy += net.calculate_state()
                net.update_state()

            net.update_weights()
            net[-1].unclamp_state()

            epoch_energy += energy

        # REAL accuracy check -- genuine UNCLAMPED settle on a cheap
        # subset, not reading beliefs right after unclamp_state() (which
        # only flips a flag, never resets z -- that was reading back the
        # clamped TARGET itself, trivially "matching" 100% every time).
        N_ACC_BATCHES = 10
        for b in range(min(N_ACC_BATCHES, n_batches)):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            net.reset_state()
            net.clamp_input(X_batch)
            net.project_forward()
            for _ in range(STEPS):  # same step count as training, genuinely unclamped
                net.calculate_state()
                net.update_state()

            terminal_beliefs = np.array(net[-1].beliefs).reshape(BATCH_SIZE, 10)
            pred = np.argmax(terminal_beliefs, axis=1)
            true = np.argmax(Y_batch, axis=1)
            correct += np.sum(pred == true)
            total += BATCH_SIZE

        epoch_acc = 100.0 * correct / total
        epoch_accs.append(epoch_acc)
        avg_energy = epoch_energy / n_batches
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | Avg energy: {avg_energy:.4f}")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    print("\nRunning final test evaluation (with forward-projection init)...")
    correct = 0
    total = 0
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i:i + BATCH_SIZE]
        if len(X_batch) != BATCH_SIZE:
            continue

        net.reset_state()
        net.clamp_input(X_batch)
        net.project_forward()
        for _ in range(300):  # generous settle for final eval readout
            net.calculate_state()
            net.update_state()

        terminal_beliefs = np.array(net[-1].beliefs).reshape(BATCH_SIZE, 10)
        pred_classes = np.argmax(terminal_beliefs, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"Deepity + forward-projection test accuracy: {test_acc:.2f}%   (ngc-learn: 95.09%)")
    print(f"Train time: {train_time:.1f}s")
    print(f"\nDeepity per-epoch: {[round(a,2) for a in epoch_accs]}")
    print(f"ngc-learn per-epoch: [26.91, 42.96, 60.12, 75.20, 84.68, 89.52, 91.90,")
    print(f"                      93.45, 94.30, 94.80, 95.13, 95.38, 95.63, 95.74, 95.95]")


if __name__ == "__main__":
    main()