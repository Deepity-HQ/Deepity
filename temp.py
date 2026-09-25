import numpy as np
import os
import sys
from pydeepity import dy
from time import perf_counter


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
    INFERENCE_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    LR_OVERRIDE = float(sys.argv[4]) if len(sys.argv) > 4 else None

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 250
    LR = LR_OVERRIDE if LR_OVERRIDE is not None else 0.00373
    IR = 0.15
    FL = 1e-3
    LMBDA = 1e-4
    DECAY_RATE = 0.94

    print(f"\nBuilding FullPCNetwork (784->512->10), seed={SEED}...")
    print(f"LR={LR}{' (overridden)' if LR_OVERRIDE is not None else ' (original MSE-tuned default)'}")
    print("Features: ALL OFF -- should be bit-identical to DirectKPPCNetwork's own defaults")

    net = dy.FullPCNetwork(batch_size=BATCH_SIZE, device="cpu")
    net.add_layer(784, 512, 10, lr=LR, ir=IR, fl=FL, lmbda=LMBDA, activation="linear", activation_deriv="dlinear")
    net.add_layer(512, 10, 10, lr=LR, ir=IR, fl=FL, lmbda=LMBDA, activation="sigmoid", activation_deriv="dsigmoid")
    net.add_layer(10, 0, 10, lr=LR, ir=IR, fl=FL, lmbda=LMBDA, activation="linear", activation_deriv="dlinear")

    net.set_use_ipc(False)
    net.set_use_cross_entropy(False)
    net.set_use_mu_pc_scaling(False)
    net.set_optimizer("ADAMW")
    net.set_psi_optimizer("ADAMW")
    net.compile()
    net.randomize_weights(SEED)

    print(f"\n*** FullPCNetwork: plain DKP-PC (every toggle off) ***")
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
    print(f"FullPCNetwork (plain DKP-PC, all toggles off) test accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")


if __name__ == "__main__":
    main()