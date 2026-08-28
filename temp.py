import numpy as np
import os
from pydeepity import GaussSeidelPCN
from time import perf_counter

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


BASE_LR = 0.001
BASE_IR = 0.04 
DECAY_RATE = 0.98

def build_net(batch_size, seed):
    net = GaussSeidelPCN(batch_size=batch_size)
    
    # Layer 0 (Input): Linear
    net.add_layer(784, 512, lr=BASE_LR, ir=BASE_IR, act="linear")
    # Layer 1 (Hidden 1): Sigmoid
    net.add_layer(512, 512, lr=BASE_LR, ir=BASE_IR, act="sigmoid")
    # Layer 2 (Hidden 2): Sigmoid
    net.add_layer(512, 10, lr=BASE_LR, ir=BASE_IR, act="sigmoid")
    # Layer 3 (Output): Linear
    net.add_layer(10, 0, lr=BASE_LR, ir=BASE_IR, act="linear")
    
    net.set_optimizer("ADAM")
    net.compile()
    net.randomize_weights()

    rng_init = np.random.default_rng(seed)
    for layer in net.layers[:-1]:
        layer.weights[:] = rng_init.uniform(-0.3, 0.3, layer.weights.shape).astype(np.float32)

    return net

def part0_sanity_check(net, X_sample, Y_sample, steps):
    print("=== Part 0: minimal sanity check ===")

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

    if np.abs(delta[idx]) > 10.0:
        print("  WARNING -- largest weight change is suspiciously large.")
        return False

    print("  Looks sane. Proceeding to real training.\n")
    return True


def main():
    X_train, Y_train, X_test, y_test_labels = load_canonical_mnist()

    BATCH_SIZE = 250
    STEPS = 20
    EPOCHS = 15
    SEED = 1234

    print(f"\nBuilding GaussSeidelPCN network (784->512->512->10), seed={SEED}...")

    net = build_net(BATCH_SIZE, SEED)

    X_sample = X_train[:BATCH_SIZE]
    Y_sample = Y_train[:BATCH_SIZE]
    if not part0_sanity_check(net, X_sample, Y_sample, STEPS):
        print("Sanity check failed -- stopping before a full training run.")
        return

    net = build_net(BATCH_SIZE, SEED)

    print(f"Training: {EPOCHS} epochs, {STEPS} steps...\n")

    rng = np.random.default_rng(SEED)
    n_batches = len(X_train) // BATCH_SIZE
    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(EPOCHS):
        current_lr = BASE_LR * (DECAY_RATE ** epoch)  
        net.set_learning_rate(current_lr)

        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        epoch_energy = 0.0
        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            energy = net.train_step_with_projection(X_batch, Y_batch, STEPS)
            epoch_energy += energy

        correct = 0
        total = 0
        for b in range(10):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            
            # Predict using steps=0 to match ancestral projection
            pred = net.predict(X_batch, steps=0)
            
            pred_classes = np.argmax(pred, axis=1)
            true_classes = np.argmax(Y_batch, axis=1)
            correct += np.sum(pred_classes == true_classes)
            total += BATCH_SIZE

        epoch_acc = 100.0 * correct / total
        epoch_accs.append(epoch_acc)
        avg_energy = epoch_energy / n_batches
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | lr={current_lr:.5f} | "
              f"Acc: {epoch_acc:.2f}% | Avg energy: {avg_energy:.4f}")

        if np.isnan(avg_energy) or np.isinf(avg_energy) or avg_energy > 1e6:
            print("\nSTOPPING -- energy diverged. This configuration is unstable.")
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
            
        # Predict using steps=0 to match ancestral projection
        pred_matrix = net.predict(X_batch, steps=0)
        
        pred_classes = np.argmax(pred_matrix, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"GaussSeidelPCN test accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"\nPer-epoch: {[round(a,2) for a in epoch_accs]}")

if __name__ == "__main__":
    main()
