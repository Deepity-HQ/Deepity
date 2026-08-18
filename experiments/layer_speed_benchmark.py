import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SequentialPCN, SimplePCN
from time import perf_counter

# TIMING-ONLY comparison. Accuracy is NOT compared -- with pr=0 (the only
# value ever used), precision is mathematically inert (verified via
# gradient check earlier: removing a permanent p=1 multiplication changes
# nothing numerically), so DiscriminativePCLayer and SimplePCLayer are
# GUARANTEED to produce identical trajectories given identical weights,
# data, and hyperparameters. There's nothing to test on accuracy. The only
# open question is whether removing the unused p/log_p buffers and the
# inert multiply actually makes SimplePCLayer measurably FASTER per step.


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


def time_discriminative(X_train, Y_train, epochs, batch_size, steps):
    net = SequentialPCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=0.06, ir=0.08, pr=0.0, act="tanh", lmbda=0.0001)
    net.add_layer(512, 10, lr=0.06, ir=0.08, pr=0.0, act="tanh", lmbda=0.0001)
    net.add_layer(10, 0, lr=0.06, ir=0.08, pr=0.0, act="linear", lmbda=0.0001)
    net.compile()
    net.randomize_weights()

    n_batches = len(X_train) // batch_size
    indices = np.arange(len(X_train))  # SAME order both runs -- pure timing, not accuracy

    start = perf_counter()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[indices[i:i + batch_size]]
            Y_batch = Y_train[indices[i:i + batch_size]]
            if len(X_batch) != batch_size:
                continue
            net.train_step(X_batch, Y_batch, steps=steps)
    elapsed = perf_counter() - start
    return elapsed, n_batches


def time_simple(X_train, Y_train, epochs, batch_size, steps):
    net = SimplePCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=0.06, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(512, 10, lr=0.06, ir=0.08, act="tanh", lmbda=0.0001)
    net.add_layer(10, 0, lr=0.06, ir=0.08, act="linear", lmbda=0.0001)
    net.compile()
    net.randomize_weights()

    n_batches = len(X_train) // batch_size
    indices = np.arange(len(X_train))

    start = perf_counter()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[indices[i:i + batch_size]]
            Y_batch = Y_train[indices[i:i + batch_size]]
            if len(X_batch) != batch_size:
                continue
            net.train_step(X_batch, Y_batch, steps=steps)
    elapsed = perf_counter() - start
    return elapsed, n_batches


def main():
    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 256
    STEPS = 60
    EPOCHS = 3  # timing-only -- don't need many epochs, per-batch cost is what matters

    print(f"\nTiming comparison: {EPOCHS} epochs, {STEPS} steps, batch_size={BATCH_SIZE}")
    print("(pr=0 makes these mathematically identical -- SPEED is the only question.)\n")

    print("Running DiscriminativePCLayer (SequentialPCN)...")
    disc_time, n_batches = time_discriminative(X_train, Y_train, EPOCHS, BATCH_SIZE, STEPS)
    print(f"  {disc_time:.1f}s total, {disc_time/(EPOCHS*n_batches)*1000:.2f}ms/batch")

    print("\nRunning SimplePCLayer (SimplePCN)...")
    simple_time, n_batches = time_simple(X_train, Y_train, EPOCHS, BATCH_SIZE, STEPS)
    print(f"  {simple_time:.1f}s total, {simple_time/(EPOCHS*n_batches)*1000:.2f}ms/batch")

    speedup = disc_time / simple_time
    print(f"\n=== Result ===")
    print(f"DiscriminativePCLayer: {disc_time:.1f}s")
    print(f"SimplePCLayer:         {simple_time:.1f}s")
    print(f"Speedup: {speedup:.3f}x  ({'faster' if speedup > 1 else 'SLOWER'})")


if __name__ == "__main__":
    main()
