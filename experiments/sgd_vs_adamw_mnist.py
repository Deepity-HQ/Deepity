import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SequentialPCN, SimplePCN
from time import perf_counter

# Full MNIST comparison: confirmed-good SGD baseline (SequentialPCN,
# lr=0.06/decay=0.98/steps=60 -- the exact config that reached 92.72%)
# vs AdamW (SimplePCN, freshly verified this session -- sign-agreement +
# multi-step energy-descent checks both passed).
#
# IMPORTANT CAVEAT: AdamW's lr=1e-3 here is a standard-convention STARTING
# POINT (matches our own PyTorch FFNN baseline's Adam lr earlier this
# session), NOT an empirically-tuned value the way lr=0.06 was for SGD.
# Adam takes ~lr-sized steps every update regardless of gradient
# magnitude (verified via the toy-problem check), so this needs its own
# tuning pass if the first result looks off -- don't assume 1e-3 is
# correct just because it's conventional.
#
# Also applying the SAME decay_rate=0.98 schedule to AdamW, not running
# it undamped -- the toy-problem verification showed real oscillation
# near convergence without decay (energy bottomed out then climbed back
# up over the second half of a 30-step run).


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


def evaluate(net, X_test, y_test_labels, steps, batch_size):
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
    return 100.0 * correct / total


def run_sgd_baseline(X_train, Y_train, X_test, y_test_labels, epochs, batch_size, steps):
    print("\n\n########## SGD baseline (SequentialPCN, confirmed lr=0.06/decay=0.98) ##########")
    net = SequentialPCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=0.06, ir=0.08, pr=0.0, act="tanh", lmbda=0.0001)
    net.add_layer(512, 10, lr=0.06, ir=0.08, pr=0.0, act="tanh", lmbda=0.0001)
    net.add_layer(10, 0, lr=0.06, ir=0.08, pr=0.0, act="linear", lmbda=0.0001)
    net.compile()
    net.randomize_weights()

    start = perf_counter()
    net.fit(X_train, Y_train, epochs=epochs, steps=steps, initial_lr=0.06, decay_rate=0.98)
    train_time = perf_counter() - start

    acc = evaluate(net, X_test, y_test_labels, steps=300, batch_size=batch_size)
    print(f"\nSGD baseline: {acc:.2f}% accuracy in {train_time:.1f}s")
    return acc, train_time


def run_adamw(X_train, Y_train, X_test, y_test_labels, epochs, batch_size, steps):
    print("\n\n########## AdamW (SimplePCN, lr=1e-3 -- unconfirmed starting point) ##########")
    net = SimplePCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=1e-3, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(512, 10, lr=1e-3, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(10, 0, lr=1e-3, ir=0.08, act="linear", lmbda=0.0001)
    net.set_optimizer("ADAMW")
    net.compile()
    net.randomize_weights()

    start = perf_counter()
    net.fit(X_train, Y_train, epochs=epochs, steps=steps, initial_lr=1e-3, decay_rate=0.98)
    train_time = perf_counter() - start

    acc = evaluate(net, X_test, y_test_labels, steps=300, batch_size=batch_size)
    print(f"\nAdamW: {acc:.2f}% accuracy in {train_time:.1f}s")
    return acc, train_time


def main():
    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 256
    STEPS = 60
    EPOCHS = 50  # matching the confirmed SGD baseline's exact epoch count

    sgd_acc, sgd_time = run_sgd_baseline(X_train, Y_train, X_test, y_test_labels, EPOCHS, BATCH_SIZE, STEPS)
    adamw_acc, adamw_time = run_adamw(X_train, Y_train, X_test, y_test_labels, EPOCHS, BATCH_SIZE, STEPS)

    print("\n\n=== Summary ===")
    print(f"SGD  (lr=0.06, confirmed):  {sgd_acc:.2f}% in {sgd_time:.1f}s")
    print(f"AdamW (lr=1e-3, unconfirmed): {adamw_acc:.2f}% in {adamw_time:.1f}s")


if __name__ == "__main__":
    main()
