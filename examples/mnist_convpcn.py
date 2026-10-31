import os

# See mnist.py's identical comment: keeps OpenMP worker threads spinning
# instead of sleeping between this engine's many short parallel regions
# per settling step. Must run before any import that can pull in libgomp.
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "300000000")

import numpy as np
import sys
from time import perf_counter
from deepity import ConvolutionalPCN
from deepity.layer import Convolution, ReLU

def load_full_mnist():
    import gzip
    import urllib.request
    print("Fetching canonical MNIST dataset (idx-ubyte)...")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files: dict[str, str] = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
        "x_test": "t10k-images-idx3-ubyte.gz",
        "y_test": "t10k-labels-idx1-ubyte.gz"
    }
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    paths: dict[str, str] = {}
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
    # (N, 784) stays compatible with input_shape=(1, 28, 28) below: a
    # flattened 1x28x28 image is bit-for-bit the same buffer as a 784-row,
    # so no reshape is needed between this loader and clamp/train calls.
    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0
    eps = 0.001
    Y_train = np.full((y_train_labels.shape[0], 10), eps, dtype=np.float32)
    Y_train[np.arange(y_train_labels.shape[0]), y_train_labels] = 1.0 - eps
    return X_train, Y_train, X_test, y_test_labels

def main() -> None:
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 250
    # benchmark.py's config trains at ~chance level, never validated.
    # These track tests/tConvDiagnose.cpp's proven->=90% config instead
    # (lr=0.02, ir=0.2, pr=0.0, lmbda=1e-4), scaled for this net's depth.
    LR = 0.005
    IR = 0.1
    PR = 0.0
    LMBDA = 1e-4
    DECAY_RATE = 0.9

    print(f"\nBuilding network (1x28x28 -> 16 -> 32 -> 128 -> 10), seed={SEED}...")
    # ConvolutionalPCN (ConvPCNetwork) has no ADAMW option -- SGD only,
    # unlike SimpleConvolutionalPCN's AdamW-capable sibling (see
    # benchmark.py, where only this class skips set_optimizer("ADAMW")).
    # The architecture must end with an explicit out_channels=0 layer:
    # ConvolutionalPCN._build_backend() uses that as its terminal marker,
    # not an auto-appended sink the way SimplePCN/SequentialPCN do.
    net = ConvolutionalPCN(
        Convolution(16, 4, 4, stride_h=2, stride_w=2, pad_h=1, pad_w=1),
        ReLU(),
        Convolution(32, 4, 4, stride_h=2, stride_w=2, pad_h=1, pad_w=1),
        ReLU(),
        Convolution(128, 7, 7, stride_h=1, stride_w=1, pad_h=0, pad_w=0),
        ReLU(),
        Convolution(10, 1, 1, stride_h=1, stride_w=1, pad_h=0, pad_w=0),
        Convolution(0, 1, 1, stride_h=1, stride_w=1, pad_h=0, pad_w=0),
        input_shape=(1, 28, 28),
        batch_size=BATCH_SIZE,
    )
    net.configure(learning_rate=LR, inference_rate=IR, precision_rate=PR, lmbda=LMBDA)
    net.randomize_weights(seed=SEED)

    print(f"\nTraining: {EPOCHS} epochs, {STEPS} steps, lr={LR}, "
          f"precision_rate={PR}, decay_rate={DECAY_RATE}...\n")

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

        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            net.train_step_with_projection(X_batch, Y_batch, STEPS)
            # ConvolutionalPCN.train_step_with_projection() does NOT call
            # this internally (unlike SequentialPCN.train_step()) -- see
            # benchmark.py's identical manual call after every batch.
            net.update_precision()

        # Real accuracy check on a subset, same pattern as mnist.py.
        N_ACC_BATCHES = 10
        for b in range(min(N_ACC_BATCHES, n_batches)):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            flat_beliefs = np.array(net.predict_with_projection(X_batch, STEPS), dtype=np.float32)
            terminal_beliefs = flat_beliefs.reshape(BATCH_SIZE, 10)
            pred = np.argmax(terminal_beliefs, axis=1)
            true = np.argmax(Y_batch, axis=1)
            correct += np.sum(pred == true)
            total += BATCH_SIZE

        epoch_acc = 100.0 * correct / total
        epoch_accs.append(epoch_acc)
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}%")

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

        flat_beliefs = np.array(net.predict_with_projection(X_batch, STEPS), dtype=np.float32)
        terminal_beliefs = flat_beliefs.reshape(BATCH_SIZE, 10)
        pred_classes = np.argmax(terminal_beliefs, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"Deepity ConvolutionalPCN Test Accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"\nPer-epoch: {[round(a, 2) for a in epoch_accs]}")

if __name__ == "__main__":
    main()
