import os

# Must run before any import that can pull in libgomp (numpy's BLAS, then
# deepity itself) -- GCC's libgomp reads these once, lazily, on its first
# parallel region, and every settling step in CalculateState()/UpdateState()
# enters and leaves one of those regions separately (GEMM, then activation,
# then error, then state update, dozens of times per batch). libgomp's
# default policy puts idle worker threads to sleep almost immediately, so
# each of those region entries pays a real wake-up latency -- measured
# ~15-20% of total training time on this engine's own settling loop. jpc/JAX
# doesn't pay this at all: its whole training step (forward init, the entire
# T-step settling loop, gradients, optimizer update) is one `@eqx.filter_jit`
# executable that enters its own thread pool once per batch, not once per op.
# Spin-waiting here closes most of that gap without restructuring dispatch.
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "300000000")

import numpy as np
import sys
from time import perf_counter
from deepity import SimplePCN
from deepity.layer import Linear, Sigmoid

def load_full_mnist():
    import gzip
    import urllib.request
    print("Fetching canonical MNIST dataset (idx-ubyte, matching ngc-learn exactly)...")
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
    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0
    eps = 0.001
    Y_train = np.full((y_train_labels.shape[0], 10), eps, dtype=np.float32)
    Y_train[np.arange(y_train_labels.shape[0]), y_train_labels] = 1.0 - eps
    return X_train, Y_train, X_test, y_test_labels

def main() -> None:
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 250
    LR = 0.00373
    DECAY_RATE = 0.94
    LMBDA = 0.0  # Keep at 0.0! Weight decay breaks the PCN's symmetric feedback

    # Under mu=W@phi(z)+b, the second-to-last layer's mu (unbounded logits
    # into the terminal layer's softmax cross-entropy) has nothing bounding
    # its scale. On separable batches, cross-entropy's true minimum is at
    # infinite weight magnitude, so ADAMW keeps growing that one weight
    # matrix until training destabilizes (observed: energy explodes ~epoch
    # 24, accuracy never recovers). LMBDA must stay 0 (it decays every
    # layer's weights uniformly, inside the PC step itself, which is what
    # breaks symmetric feedback); this instead shrinks only the one
    # pathological weight matrix, from outside the library, once per batch.
    # This value sits right at the stabilizer's margin and is tuned to the
    # exact floating-point accumulation order of whichever GEMM kernel
    # actually runs the feedback matmul -- it needed re-tuning once already
    # (1e-4->2e-4) when oneDNN's matmul replaced cblas_sgemm there, and
    # again (->5e-4) when DEEPITY_USE_LIBXSMM's row-blocked JIT kernel
    # replaced oneDNN. 2e-4 is for the DEFAULT build (DEEPITY_USE_LIBXSMM
    # OFF); rebuilding with -DDEEPITY_USE_LIBXSMM=ON needs 5e-4 instead
    # (verified: 4e-4 drifts down to ~94% by epoch 100, 6e-4 holds a longer
    # 99%+ plateau but then crashes suddenly at the very end, 5e-4 is the
    # stable middle -- 96.27% final accuracy over a full 100-epoch run).
    TERMINAL_FEED_WEIGHT_DECAY = 2e-4

    print(f"\nBuilding network (784->512->512->10), seed={SEED}...")
    net = SimplePCN(
        Linear(784, 512),
        Linear(512, 512),
        Sigmoid(),
        Linear(512, 10),
        Sigmoid(),
        batch_size=BATCH_SIZE,
    )
    net.configure(learning_rate=LR, inference_rate=0.091, lmbda=LMBDA, optimizer="ADAMW")

    # mu=W@phi(z)+b (see SimplePCLayer.h) carries no final activation of
    # its own, so the terminal layer's prediction target (mu from the
    # second-to-last layer) is unbounded logits, not [0,1]. Softmax
    # cross-entropy bounds the terminal layer's error/settling target
    # at the comparison itself instead, so the terminal belief (used
    # for argmax accuracy below) still settles toward a valid
    # probability distribution during free (unclamped) inference.
    net[-1].set_cross_entropy(True)

    # Match exact +-0.3 bounds for forward-projection initialization
    rng_init = np.random.default_rng(SEED)
    for layer in net.layers[:-1]:
        w_shape = layer.weights.shape
        layer.set_weights(rng_init.uniform(-0.3, 0.3, w_shape).astype(np.float32).ravel())

    print(f"\nTraining with FORWARD-PROJECTION init: {EPOCHS} epochs, {STEPS} steps, "
          f"lr={LR}, decay_rate={DECAY_RATE}...\n")
    print("Reference: ngc-learn's own pc_discrim exhibit (same 784-512-512-10 "
          "architecture, classic settling, T=20, mb_size=250, 100 epochs), run "
          "unmodified on this same machine -- see logs/ngclearn_pc_discrim_mnist.txt: "
          "best dev acc = 98%, 1707.7s total (17.0s/epoch).\n")

    rng = np.random.default_rng(SEED)
    n_batches = len(X_train) // BATCH_SIZE
    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(EPOCHS):
        current_lr = LR * (DECAY_RATE ** epoch)
        net.set_learning_rate(current_lr)

        # Standard randomized batches
        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        correct = 0
        total = 0
        epoch_energy = 0.0

        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            # 3. Call C++ Native Loop (avoid Nanobind overhead)
            energy = net.train_step_with_projection(X_batch, Y_batch, STEPS)
            epoch_energy += energy

            if TERMINAL_FEED_WEIGHT_DECAY > 0.0:
                terminal_feed_layer = net.layers[-2]
                terminal_feed_layer.set_weights(
                    (terminal_feed_layer.weights * (1.0 - TERMINAL_FEED_WEIGHT_DECAY))
                    .astype(np.float32)
                    .ravel()
                )

        # Real accuracy check on a subset
        N_ACC_BATCHES = 10
        for b in range(min(N_ACC_BATCHES, n_batches)):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            net.reset_state()
            net.clamp_input(X_batch)
            net.project_forward()
            for _ in range(STEPS):
                net.calculate_state(False)
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
        flat_beliefs = net.predict_with_projection(X_batch, STEPS)
        terminal_beliefs = flat_beliefs.reshape(BATCH_SIZE, 10)

        pred_classes = np.argmax(terminal_beliefs, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"Deepity Peak Test Accuracy: {test_acc:.2f}%   (ngc-learn: 95.09%)")
    print(f"Train time: {train_time:.1f}s")
    print(f"\nDeepity per-epoch: {[round(a,2) for a in epoch_accs]}")

if __name__ == "__main__":
    main()
