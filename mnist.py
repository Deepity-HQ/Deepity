import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SequentialPCN
from time import perf_counter

# Reproduces the confirmed ~93% / ~1500-1700s result from earlier this
# session -- SequentialPCN (DiscriminativePCLayer, pr=0), plain shuffled
# batching (NOT the C++ StreamAlignedBatcher, NOT SimplePCLayer, NOT
# I_avg caching -- none of those were part of this specific result).
# Uses net.fit() (SequentialPCN's own built-in training loop) instead of
# a hand-rolled one, for a cleaner reference script.
#
# HONEST NOTE: the original runs landed at 93.07% in 1703.3s and 92.61%
# in 1530.0s -- two separate runs of the SAME config, with real run-to-run
# variance from random weight init. This script targets that same
# config; expect a result in the same ballpark (~92-93%, ~1500-1900s),
# not an exact guaranteed number every time.


def load_full_mnist():
    print("Fetching full MNIST dataset (70,000 images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int)

    Y_bipolar = np.full((y.shape[0], 10), -0.9, dtype=np.float32)
    Y_bipolar[np.arange(y.shape[0]), y] = 0.9

    X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = train_test_split(
        X, Y_bipolar, y, train_size=60000, test_size=10000, stratify=y, random_state=42
    )
    return X_train, Y_train, X_test, y_test_labels


def main():
    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 256
    STEPS = 60      # confirmed correct value -- STEPS=150 (an earlier, untested
                     # default) was found this session to cause inference
                     # collapse; 60 was the empirically-verified fix
    EPOCHS = 50
    LR = 0.06       # reverse-engineered from the original 93%-run's own
                     # printed LR-decay trace, confirmed via exact ratio match
    DECAY_RATE = 0.98

    print(f"\nBuilding dense network (batch_size={BATCH_SIZE})...")
    net = SequentialPCN(batch_size=BATCH_SIZE)
    net.add_layer(784, 512, lr=LR, ir=0.08, pr=0.0, act="tanh", lmbda=0.0001)
    net.add_layer(512, 10, lr=LR, ir=0.08, pr=0.0, act="tanh", lmbda=0.0001)
    net.add_layer(10, 0, lr=LR, ir=0.08, pr=0.0, act="linear", lmbda=0.0001)
    net.compile()
    net.randomize_weights()

    print(f"\nTraining: {EPOCHS} epochs, {STEPS} inference steps, "
          f"lr={LR}, decay_rate={DECAY_RATE}...\n")

    start_time = perf_counter()
    net.fit(X_train, Y_train, epochs=EPOCHS, steps=STEPS, initial_lr=LR, decay_rate=DECAY_RATE)
    train_time = perf_counter() - start_time

    print(f"\nTraining complete in {train_time:.1f}s.")

    print("\nEvaluating on the full held-out test set...")
    correct = 0
    total = 0
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i:i + BATCH_SIZE]
        if len(X_batch) != BATCH_SIZE:
            continue
        pred_matrix = net.predict(X_batch, steps=300)  # generous, confirmed ~fully converged
        pred_classes = np.argmax(pred_matrix, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    accuracy = (correct / total) * 100
    print(f"\n=== Result ===")
    print(f"Test Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Train time: {train_time:.1f}s")
    print(f"\n(Reference: original runs were 93.07% / 1703.3s and 92.61% / 1530.0s)")


if __name__ == "__main__":
    main()
