# This is the file you can use to train a simple PCN on MNIST! ~93% accuracy in <1000 seconds.

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SimplePCN
from time import perf_counter

def load_full_mnist():
    print("Fetching full MNIST dataset (70,000 images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int) # type: ignore

    Y_bipolar = np.full((y.shape[0], 10), -0.9, dtype=np.float32)
    Y_bipolar[np.arange(y.shape[0]), y] = 0.9

    X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = train_test_split(
        X, Y_bipolar, y, train_size=60000, test_size=10000, stratify=y, random_state=42
    )
    return X_train, Y_train, X_test, y_test_labels, y_train_labels


def main() -> None:
    X_train, Y_train, X_test, y_test_labels, y_train_labels = load_full_mnist()

    PER_CLASS = 25
    NUM_CLASSES = 10
    BATCH_SIZE = PER_CLASS * NUM_CLASSES  # 250 -- I_avg's StreamAlignedBatcher
                                            # needs an exact per-class composition,
                                            # not an arbitrary batch size
    STEPS = 60
    EPOCHS = 10
    LR = 0.06 / 15       # ~0.004
    DECAY_RATE = 0.98
    HIDDEN_LAYER_INDEX = 1  # the 512-unit hidden layer (index 1: 0=784->512
                              # input layer itself, 1=512->10 hidden, 2=10->0
                              # terminal); this is whose beliefs get cached
    HIDDEN_SIZE = 512

    print(f"\nBuilding dense network (batch_size={BATCH_SIZE}, precision-free SimplePCN)...")
    net = SimplePCN(batch_size=BATCH_SIZE)
    net.add_layer(784, 512, lr=LR, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(512, 10, lr=LR, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(10, 0, lr=LR, ir=0.08, act="linear", lmbda=0.0001)
    net.compile()
    net.randomize_weights()

    print(f"\nTraining with I_avg: {EPOCHS} epochs, {STEPS} inference steps, "
          f"lr={LR:.5f} (0.06/15), decay_rate={DECAY_RATE}...\n")

    start_time = perf_counter()
    net.fit_iavg(
        X_train, Y_train, y_train_labels,
        epochs=EPOCHS, steps=STEPS,
        per_class=PER_CLASS, num_classes=NUM_CLASSES,
        hidden_layer_index=HIDDEN_LAYER_INDEX, hidden_size=HIDDEN_SIZE,
        initial_lr=LR, decay_rate=DECAY_RATE,
        reset_cache_per_epoch=True,
    )
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
        pred_matrix = net.predict(X_batch, steps=300)
        pred_classes = np.argmax(pred_matrix, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    accuracy = (correct / total) * 100
    print(f"\n=== Result ===")
    print(f"Test Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Train time: {train_time:.1f}s")


if __name__ == "__main__":
    main()