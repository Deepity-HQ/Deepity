import numpy as np
import os
from pydeepity import SequentialPCN
from time import perf_counter

def load_full_mnist():
    import gzip
    import urllib.request

    print("Fetching canonical MNIST dataset (idx-ubyte, matching ngc-learn exactly)...")
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
    import sys
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7  # pass a seed on the
                                                            # command line to
                                                            # actually vary it,
                                                            # e.g. `python
                                                            # mnist_final_validated.py 123`

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 250       # CONFIRMED from their canonical train_pcn.py's
                             # mb_size=250, not our own 256
    STEPS = 20              # matches ngc-learn's T=20
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15  # pass a 2nd arg
                                                               # to try more
                                                               # epochs, e.g.
                                                               # `python
                                                               # mnist_final_validated.py 7 25`
    LR = 0.001               # Adam-appropriate scale, matches ngc-learn's own
                              # stated eta=0.001 directly
    DECAY_RATE = 0.98
    LMBDA = 0.0              # CONFIRMED from HebbianSynapse's real source:
                              # prior defaults to ("constant", 0.), never
                              # overridden anywhere in pcn_model.py -- ngc-learn
                              # runs with NO weight decay, not our own 0.0001

    print(f"\nBuilding network (784->512->512->10), seed={SEED}...")
    net = SequentialPCN(batch_size=BATCH_SIZE)
    net.add_layer(784, 512, lr=LR, ir=0.08, pr=0.0, act="linear", lmbda=LMBDA)
    net.add_layer(512, 512, lr=LR, ir=0.08, pr=0.0, act="sigmoid", lmbda=LMBDA)
    net.add_layer(512, 10, lr=LR, ir=0.08, pr=0.0, act="sigmoid", lmbda=LMBDA)
    # Explicit TERMINAL layer -- outChannels/nextSize=0. Without this,
    # net[-1] is the 512->10 layer itself, whose OWN beliefs are its
    # 512-dim INPUT, not the 10-dim prediction it produces -- exactly
    # the bug that just crashed this script (and was already silently
    # corrupting every batch before that: clamp_state() truncates rather
    # than erroring on a size mismatch).
    net.add_layer(10, 0, lr=LR, ir=0.08, pr=0.0, act="linear", lmbda=LMBDA)
    net.set_optimizer("ADAM")  # CONFIRMED from the real Adam source: plain Adam,
                                 # no decoupled weight-decay term at all -- was
                                 # "ADAMW" before, a real, if likely small, mismatch
                                 # now that lmbda=0 anyway (AdamW's decay term would
                                 # be inert at lmbda=0, but matching their exact
                                 # optimizer type removes any doubt).
    net.compile()
    net.randomize_weights()
    # net.set_mu_cache_threshold(0.0)  # REVERTED from 0.05 -- that showed genuine
                                       # instability under real, sustained AdamW
                                       # training (energy trending UP, a 62-point
                                       # accuracy crash at epoch 8), not just noise.
                                       # Likely cause: small caching-approximation
                                       # bias accumulating in Adam's momentum/
                                       # variance buffers over many updates --
                                       # something the earlier single-trajectory,
                                       # no-weight-update speed test structurally
                                       # could not have caught. threshold=0 is
                                       # exact (proven bit-identical to no caching)
                                       # and confirmed safe with this exact stack.

    # Override weight init to match ngc-learn's actual convention: fixed
    # uniform range +-0.3, INDEPENDENT of layer size -- not our own
    # size-scaled Gaussian (std ~0.039 for the 784->512 layer, ~8x
    # smaller). This directly affects how informative the very first
    # forward-projection pass is -- larger initial weights produce more
    # differentiated activations from a single forward pass. Biases
    # already default to 0 in Deepity (never touched by
    # RandomizeWeights()), matching ngc-learn's own bias_init=constant(0).
    rng_init = np.random.default_rng(SEED)
    for layer in net.layers[:-1]:  # skip the terminal -- it has no weights (nextSize=0)
        w_shape = layer.weights.shape
        layer.weights[:] = rng_init.uniform(-0.3, 0.3, w_shape).astype(np.float32)

    print(f"\nTraining with FORWARD-PROJECTION init: {EPOCHS} epochs, {STEPS} steps, "
          f"lr={LR}, decay_rate={DECAY_RATE}...\n")
    print("Reference (ngc-learn, real run): 26.91, 42.96, 60.12, 75.20, 84.68, 89.52,")
    print("  91.90, 93.45, 94.30, 94.80, 95.13, 95.38, 95.63, 95.74, 95.95 -- test 95.09%\n")

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

            energy = net.train_step_with_projection(X_batch, Y_batch, STEPS)
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
        flat_beliefs = net.predict_with_projection(X_batch, STEPS)
        terminal_beliefs = flat_beliefs.reshape(BATCH_SIZE, 10)

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
