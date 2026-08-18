import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SequentialPCN
from time import perf_counter

# Tests the leading unconfirmed hypothesis from today's I_avg investigation:
# does caching's benefit scale with network DEPTH? Everything tested so far
# used a single hidden layer (784->512->10); the paper's own architecture
# was a 5-layer MLP. This uses 784->512->512->512->10 (3 genuine hidden
# layers, matching the paper's width=512), with MULTI-layer caching (every
# hidden layer seeded from its own class-average), not just one -- our
# earlier fit_iavg only supported a single cached layer, which would be an
# unfaithful test of the depth hypothesis.


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
    return X_train, Y_train, X_test, y_test_labels, y_train_labels


class StreamAlignedBatcher:
    def __init__(self, X, Y, labels, num_classes, per_class, seed=42):
        self.X = X
        self.Y = Y
        self.labels = labels
        self.num_classes = num_classes
        self.per_class = per_class
        self.batch_size = num_classes * per_class
        self.rng = np.random.default_rng(seed)
        self.class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        self.pointers = [0] * num_classes
        self._reshuffle_all()

    def _reshuffle_all(self):
        for c in range(self.num_classes):
            self.rng.shuffle(self.class_indices[c])
        self.pointers = [0] * self.num_classes

    def _next_class_slice(self, c, n):
        idx = self.class_indices[c]
        p = self.pointers[c]
        if p + n > len(idx):
            self.rng.shuffle(idx)
            p = 0
        self.pointers[c] = p + n
        return idx[p:p + n]

    def num_batches_per_epoch(self):
        return min(len(idx) for idx in self.class_indices) // self.per_class

    def get_batch(self):
        idx_parts = [self._next_class_slice(c, self.per_class) for c in range(self.num_classes)]
        all_idx = np.concatenate(idx_parts)
        return self.X[all_idx], self.Y[all_idx], self.labels[all_idx]


def build_deep_net(batch_size, lr, ir):
    net = SequentialPCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=lr, ir=ir, pr=0.0, act="tanh", lmbda=0.0001)   # layer0: input
    net.add_layer(512, 512, lr=lr, ir=ir, pr=0.0, act="tanh", lmbda=0.0001)   # layer1: hidden 1
    net.add_layer(512, 512, lr=lr, ir=ir, pr=0.0, act="tanh", lmbda=0.0001)   # layer2: hidden 2
    net.add_layer(512, 10, lr=lr, ir=ir, pr=0.0, act="tanh", lmbda=0.0001)    # layer3: hidden 3
    net.add_layer(10, 0, lr=lr, ir=ir, pr=0.0, act="linear", lmbda=0.0001)    # layer4: terminal
    net.compile()
    net.randomize_weights()
    return net


# The three genuine hidden layers whose beliefs get cached -- layer0 is
# input (clamped pixels, do NOT cache), layer4 is terminal (clamped target,
# do NOT cache).
CACHED_LAYERS = [(1, 512), (2, 512), (3, 512)]


def train_deep(X_train, Y_train, y_train_labels, X_test, y_test_labels,
                steps, lr, ir, epochs, per_class, num_classes, decay_rate,
                use_cache, label):

    batch_size = per_class * num_classes
    net = build_deep_net(batch_size, lr, ir)

    batcher = StreamAlignedBatcher(X_train, Y_train, y_train_labels, num_classes, per_class)
    n_batches = batcher.num_batches_per_epoch()

    print(f"\n=== {label}: {epochs} epochs, {n_batches} batches/epoch, "
          f"batch_size={batch_size}, steps={steps}, lr={lr}, use_cache={use_cache} ===")

    # One cache dict PER cached layer.
    caches = {idx: {} for idx, _ in CACHED_LAYERS}

    start = perf_counter()
    for epoch in range(epochs):
        current_lr = lr * (decay_rate ** epoch)
        net.set_learning_rate(current_lr)

        epoch_energy = 0.0
        batches_done = 0

        for _ in range(n_batches):
            X_batch, Y_batch, labels_batch = batcher.get_batch()

            net.reset_state()
            if use_cache:
                for layer_idx, layer_size in CACHED_LAYERS:
                    cache = caches[layer_idx]
                    if cache:
                        init = np.zeros((batch_size, layer_size), dtype=np.float32)
                        for c in range(num_classes):
                            if c in cache:
                                init[c * per_class:(c + 1) * per_class] = cache[c]
                        net[layer_idx].beliefs[:] = init

            net.clamp_input(X_batch.flatten())
            net[-1].clamp_state(Y_batch.flatten())

            energy = 0.0
            for _ in range(steps):
                energy = net.calculate_state()
                net.update_state()

            net.update_weights()

            if use_cache:
                for layer_idx, layer_size in CACHED_LAYERS:
                    settled = net[layer_idx].beliefs.reshape(batch_size, layer_size).copy()
                    cache = caches[layer_idx]
                    for c in range(num_classes):
                        cache[c] = settled[c * per_class:(c + 1) * per_class].mean(axis=0)

            net[-1].unclamp_state()
            epoch_energy += energy
            batches_done += 1

        avg_per_step = (epoch_energy / batches_done) / steps
        elapsed = perf_counter() - start
        print(f"  epoch {epoch+1}/{epochs}  lr={current_lr:.5f}  "
              f"avg/steps={avg_per_step:9.4f}  elapsed={elapsed:.1f}s")

    total_time = perf_counter() - start

    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        y_labels_batch = y_test_labels[i:i + batch_size]
        if len(X_batch) != batch_size:
            continue
        net.reset_state()
        net.clamp_input(X_batch.flatten())
        for _ in range(300):  # generous, ~fully converged readout
            net.calculate_state()
            net.update_state()
        pred = np.array(net[-1].beliefs).reshape(batch_size, 10)
        pred_classes = np.argmax(pred, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += batch_size

    acc = 100.0 * correct / total
    print(f"  -> Test accuracy: {acc:.2f}%  |  Train time: {total_time:.1f}s")
    return acc, total_time


def main():
    X_train, Y_train, X_test, y_test_labels, y_train_labels = load_full_mnist()

    STEPS = 60
    IR = 0.08
    EPOCHS = 15  # deeper net = more compute/epoch -- shorter run than the 50-epoch baselines
    PER_CLASS = 25
    NUM_CLASSES = 10

    print("\n\n########## DEPTH TEST 1: zero-init, deep network (no caching) ##########")
    acc_nocache, t_nocache = train_deep(
        X_train, Y_train, y_train_labels, X_test, y_test_labels,
        steps=STEPS, lr=0.06, ir=IR, epochs=EPOCHS,
        per_class=PER_CLASS, num_classes=NUM_CLASSES, decay_rate=0.98,
        use_cache=False, label="No cache (zero-init)",
    )

    print("\n\n########## DEPTH TEST 2: I_avg, deep network (multi-layer caching) ##########")
    acc_cache, t_cache = train_deep(
        X_train, Y_train, y_train_labels, X_test, y_test_labels,
        steps=STEPS, lr=0.06 / 15, ir=IR, epochs=EPOCHS,
        per_class=PER_CLASS, num_classes=NUM_CLASSES, decay_rate=1.0,
        use_cache=True, label="I_avg (multi-layer cache)",
    )

    print("\n\n=== Summary ===")
    print(f"No cache (deep):  {acc_nocache:.2f}% in {t_nocache:.1f}s")
    print(f"I_avg   (deep):   {acc_cache:.2f}% in {t_cache:.1f}s")
    print("\nIf I_avg now MEETS OR BEATS no-cache at this depth (unlike the 1-hidden-layer")
    print("case, where it consistently trailed by ~2-3 points), that confirms the depth")
    print("hypothesis. If the same gap persists, depth isn't the explanation either.")


if __name__ == "__main__":
    main()
