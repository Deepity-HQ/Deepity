import numpy as np
import os
import sys
from time import perf_counter
from pydeepity import ConvolutionalPCN

def load_full_mnist():
    import gzip
    import urllib.request

    print("Fetching canonical MNIST dataset...")
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
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 250
    STEPS = 20
    LR = 0.001
    IR = 0.08
    DECAY_RATE = 0.98
    LMBDA = 0.0  
    PR = 0.0 # Keep precision weighting off for MNIST 

    print(f"\nBuilding ConvolutionalPCN, seed={SEED}...")
    net = ConvolutionalPCN(batch_size=BATCH_SIZE)
    
    # Layer 1: 1x28x28 -> 16x14x14
    net.add_layer(out_channels=16, kernel_h=4, kernel_w=4,
                  in_channels=1, in_height=28, in_width=28,
                  stride_h=2, stride_w=2, pad_h=1, pad_w=1,
                  lr=LR, ir=IR, pr=PR, act="relu", lmbda=LMBDA)
                  
    # Layer 2: 16x14x14 -> 32x7x7
    net.add_layer(out_channels=32, kernel_h=4, kernel_w=4,
                  stride_h=2, stride_w=2, pad_h=1, pad_w=1,
                  lr=LR, ir=IR, pr=PR, act="relu", lmbda=LMBDA)
                  
    # Layer 3: 32x7x7 -> 10x1x1 (Acts as the fully connected projection)
    net.add_layer(out_channels=10, kernel_h=7, kernel_w=7,
                  stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                  lr=LR, ir=IR, pr=PR, act="linear", lmbda=LMBDA)
                  
    # Terminal Layer: Clamps to the 10-element target vector
    net.add_layer(out_channels=0, kernel_h=1, kernel_w=1,
                  lr=LR, ir=IR, pr=PR, act="linear", lmbda=LMBDA)
    
    # Check if the nanobind module exposes the advanced toggles on the parent class
    try:
        net.set_optimizer("ADAMW")
        net.set_mu_cache_threshold(0.0)
    except AttributeError:
        print("\n[WARNING] Advanced toggles not found. Ensure 'set_optimizer' and 'set_mu_cache_threshold' are bound in your Nanobind configuration for ConvPCNetwork.")
    
    net.compile()
    
    # We will rely on the native C++ RandomizeWeights (He-style variance) 
    # instead of overriding it with the +-0.3 uniform distribution here, 
    # as convolutions are highly sensitive to aggressive variance.
    net.randomize_weights()

    print(f"\nTraining CONVOLUTIONAL PCN: {EPOCHS} epochs, {STEPS} steps, "
          f"lr={LR}, decay_rate={DECAY_RATE}...\n")

    rng = np.random.default_rng(SEED)
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

            # Manually flatten here since we are calling the C++ method directly
            energy = net.train_step_with_projection(X_batch.flatten(), Y_batch.flatten(), STEPS)
            epoch_energy += energy

        N_ACC_BATCHES = 10
        for b in range(min(N_ACC_BATCHES, n_batches)):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            net.reset_state()
            net.clamp_input(X_batch.flatten())
            net.project_forward()
            for _ in range(STEPS): 
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

    print("\nRunning final test evaluation...")
    correct = 0
    total = 0
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i:i + BATCH_SIZE]
        if len(X_batch) != BATCH_SIZE:
            continue

        net.reset_state()
        net.clamp_input(X_batch.flatten())
        flat_beliefs = net.predict_with_projection(X_batch.flatten(), STEPS)
        terminal_beliefs = flat_beliefs.reshape(BATCH_SIZE, 10)

        pred_classes = np.argmax(terminal_beliefs, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"Deepity ConvPCN Test Accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"\nPer-epoch accuracy: {[round(a,2) for a in epoch_accs]}")

if __name__ == "__main__":
    main()
