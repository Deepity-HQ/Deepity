import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SequentialPCN
from time import perf_counter

# Fast diagnostic: SMALL subset of MNIST, few epochs, same hyperparameters
# as the real run -- watch whether energy trends down in a healthy way
# before spending 35-50 minutes finding out on the full dataset. Should
# take on the order of a minute or two, not tens of minutes.


def load_mnist_subset(n_train=3000, n_test=500):
    print("Fetching MNIST (subset)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int)

    Y_bipolar = np.full((y.shape[0], 10), -0.9, dtype=np.float32)
    Y_bipolar[np.arange(y.shape[0]), y] = 0.9

    X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = train_test_split(
        X, Y_bipolar, y, train_size=n_train, test_size=n_test, stratify=y, random_state=42
    )
    return X_train, Y_train, X_test, y_test_labels


def quick_check(X_train, Y_train, X_test, y_test_labels,
                 steps, lr, epochs, batch_size, decay_rate=0.89):

    net = SequentialPCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=lr, ir=0.08, pr=0., act="tanh", lmbda=0.0001)
    net.add_layer(512,  10, lr=lr, ir=0.08, pr=0., act="tanh", lmbda=0.0001)
    net.add_layer( 10,   0, lr=lr, ir=0.08, pr=0., act="linear", lmbda=0.0001)

    net.compile()
    net.randomize_weights()

    print(f"\nQuick check: {epochs} epochs, {steps} steps, batch_size={batch_size}, "
          f"lr={lr}, decay_rate={decay_rate}, n_train={len(X_train)}\n")

    start_time = perf_counter()

    net.fit(
        X_train, Y_train,
        epochs=epochs,
        steps=steps,
        initial_lr=lr,
        decay_rate=decay_rate,
    )

    elapsed = perf_counter() - start_time
    print(f"\nQuick check training done in {elapsed:.1f}s.")

    # Cheap accuracy readout -- not the real test set, just a sanity signal
    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        y_labels_batch = y_test_labels[i:i + batch_size]
        if len(X_batch) != batch_size:
            continue
        pred_matrix = net.predict(X_batch, steps=steps)
        pred_classes = np.argmax(pred_matrix, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += batch_size

    if total > 0:
        acc = 100.0 * correct / total
        print(f"Quick-check accuracy on {total} held-out samples: {acc:.2f}%")
    else:
        print("Not enough test samples for a full batch at this batch_size -- "
              "reduce batch_size or increase n_test to get an accuracy readout.")


if __name__ == "__main__":
    X_train, Y_train, X_test, y_test_labels = load_mnist_subset(n_train=3000, n_test=500)

    quick_check(
        X_train, Y_train, X_test, y_test_labels,
        steps=150,
        lr=0.06,           # confirmed from real trace: 0.06 * 0.98^epoch matches exactly
        epochs=8,          # a handful, not 50 -- just enough to see the trend
        batch_size=256,
        decay_rate=0.98,   # confirmed from real trace (0.0588/0.06 = 0.98 exactly)
    )
