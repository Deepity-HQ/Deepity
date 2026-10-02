import numpy as np
import os
import sys
from time import perf_counter

from pydeepity import FullPCN, Linear, TanH


def load_full_mnist():
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


def main() -> None:
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    # 8, not the old 2: ePC's whole advantage over plain one-hop settling
    # is that it doesn't need many settling steps per hop of depth, so a
    # tiny step count is exactly where the two modes should look most
    # different on this now-6-weight-layer-deep architecture (see the
    # scratch investigation this session: on a 13-layer-deep net, plain
    # one-hop settling needed ~2000-5000 steps to reach what ePC reached
    # in ~30 -- flip USE_EPC below and rerun to see the gap on THIS
    # architecture).
    INFERENCE_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    LR_OVERRIDE = float(sys.argv[4]) if len(sys.argv) > 4 else None

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 250
    # Retuned for THIS config (use_cross_entropy=True + use_epc=True +
    # ADAMW): the old 0.00373 was tuned for MSE loss, never retuned after
    # cross-entropy was turned on, and collapses by epoch ~6-8 (accuracy
    # craters while "energy" -- now a cross-entropy loss, not MSE --
    # keeps dropping, i.e. the model collapses toward a narrow, overconfident
    # class subset rather than generalizing). Swept via 20-epoch trials:
    # 0.00373/0.0015/0.0006/0.00025/0.0001 all collapse (0.0001 just takes
    # longer -- energy visibly starts RISING again past epoch ~6, the same
    # early-warning sign as the higher LRs, just slower); 0.00007 shows the
    # identical early-warning pattern (energy bottoms at epoch 9, then
    # rises); 0.00004 and 0.00002 both ran the full 20 epochs with energy
    # monotonically decreasing throughout and accuracy still climbing at
    # the final epoch (95.20% and 92.96% respectively) -- 0.00004 is the
    # faster of the two stable points, right at the edge without crossing
    # it. IR/FL/LMBDA/decay_rate were deliberately left untouched (this
    # was an LR-only retuning pass).
    LR = LR_OVERRIDE if LR_OVERRIDE is not None else 0.00004
    IR = 0.15
    FL = 1e-3
    LMBDA = 1e-4
    DECAY_RATE = 0.94

    print(f"\nBuilding FullPCN (784->512->256->128->64->32->10), seed={SEED}...")
    print(f"LR={LR}{' (overridden)' if LR_OVERRIDE is not None else ' (retuned for cross-entropy+ePC+ADAMW)'}")

    USE_EPC = True  # flip to False and rerun to compare against plain
                     # one-hop settling on this same deep architecture

    # Declarative architecture (same pattern SimplePCN/DKPPCN use): each
    # Linear is optionally followed by an Activation instance setting
    # THAT layer's own activation; omitting one (as for the first Linear
    # below) defaults to "linear". The terminal sink layer is appended
    # automatically by configure(), no need to add it by hand.
    net = FullPCN(
        Linear(784, 512),
        Linear(512, 256), TanH(),
        Linear(256, 128), TanH(),
        Linear(128, 64), TanH(),
        Linear(64, 32), TanH(),
        Linear(32, 10), TanH(),
        batch_size=BATCH_SIZE,
    )
    net.configure(
        learning_rate=LR,
        inference_rate=IR,
        feedback_rate=FL,
        lmbda=LMBDA,
        optimizer="ADAMW",
        psi_optimizer="ADAMW",
        use_ipc=False,
        use_cross_entropy=True,
        use_mu_pc_scaling=False,
        use_epc=USE_EPC,
        # use_residual_connections is also available here (fixed to work
        # correctly together with use_epc this session), but requires
        # matching width on every middle hidden layer -- this stack's
        # widths deliberately shrink (512->256->128->64->32), so it isn't
        # eligible without a retuning pass of its own.
        seed=SEED,
    )

    print(f"\n*** FullPCN: 6 weight-layers deep, ePC {'ON' if USE_EPC else 'OFF'} ***")
    print(f"Training: {EPOCHS} epochs, inference_steps={INFERENCE_STEPS}, ")
    print(f"lr={LR}, ir={IR}, lmbda={LMBDA}, decay_rate={DECAY_RATE}...\n")

    rng = np.random.default_rng(SEED)
    n_batches = len(X_train) // BATCH_SIZE
    start_time = perf_counter()

    for epoch in range(EPOCHS):
        current_lr = LR * (DECAY_RATE ** epoch)
        net.set_learning_rate(current_lr)

        current_fl = FL * (DECAY_RATE ** epoch)
        net.set_feedback_rate(current_fl)

        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        correct = 0
        total = 0
        epoch_energy = 0.0

        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            energy = net.train_step(X_batch, Y_batch, INFERENCE_STEPS)
            epoch_energy += energy

        N_ACC_BATCHES = 10
        for b in range(min(N_ACC_BATCHES, n_batches)):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            terminal_beliefs = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, 10)
            pred = np.argmax(terminal_beliefs, axis=1)
            true = np.argmax(Y_batch, axis=1)
            correct += np.sum(pred == true)
            total += BATCH_SIZE

        epoch_acc = 100.0 * correct / total
        avg_energy = epoch_energy / n_batches
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | Avg energy: {avg_energy:.4f}")

        if not np.isfinite(avg_energy):
            print("NON-FINITE ENERGY -- STOPPING")
            return

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

        terminal_beliefs = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, 10)
        pred_classes = np.argmax(terminal_beliefs, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"FullPCN (6 layers deep, ePC {'ON' if USE_EPC else 'OFF'}) test accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")


if __name__ == "__main__":
    main()
