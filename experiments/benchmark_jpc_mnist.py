"""
Runs jpc (JAX predictive-coding library, github.com/thebuckleylab/jpc) on the
SAME real MNIST task, architecture, and hyperparameters as
examples/mnist.py's own Deepity benchmark (784-512-512-10, batch=250,
T=20 settling steps, 100 epochs, ADAMW, lr=0.00373 decaying 0.94/epoch,
forward-projection weight init in [-0.3, 0.3]) -- a direct wall-clock and
accuracy comparison point, not a synthetic-data throughput probe (see
benchmark_jpc.py for that).

Matched to Deepity's TRAINING protocol exactly: jpc.make_pc_step's
settling phase clamps both input AND output while relaxing hidden
activities (jpc.solve_inference's documented behaviour), the same
convention Deepity's train_step_with_projection uses. Euler integration,
dt=INFERENCE_RATE, max_t1=STEPS*INFERENCE_RATE gives exactly STEPS fixed
steps, matching Deepity's T=20, inference_rate=0.091 exactly.

Deliberately NOT matched on test-time accuracy measurement: Deepity reads
the terminal belief after T=20 *settling* steps at test time too
(predict_with_projection). jpc's API only supports clamping the output
layer to a *known* target during solve_inference -- there's no supported
way to "settle freely toward an unknown label" -- so jpc's own idiomatic
discriminative-PC test protocol (jpc.test_discriminative_pc) is pure
feedforward readout post-training, zero settling steps. Replicating
Deepity's settled-readout eval in jpc would mean hand-rolling something
the library's author didn't intend, so this script uses jpc's own
standard protocol instead and reports that distinction explicitly.
"""

import gzip
import os
import sys
import time
import urllib.request

import diffrax
import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr
import numpy as np
import optax

import jpc

LAYER_SIZES = [784, 512, 512, 10]
BATCH_SIZE = 250
LR = 0.00373
DECAY_RATE = 0.94
STEPS = 20
INFERENCE_RATE = 0.091
WEIGHT_INIT_BOUND = 0.3


def load_full_mnist():
    # Verbatim copy of examples/mnist.py's loader -- same canonical
    # idx-ubyte MNIST, same preprocessing, so both libraries train/test
    # on bit-identical data.
    print("Fetching canonical MNIST dataset (idx-ubyte, matching Deepity's own loader)...")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files: dict[str, str] = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
        "x_test": "t10k-images-idx3-ubyte.gz",
        "y_test": "t10k-labels-idx1-ubyte.gz",
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
    with gzip.open(paths["x_train"], "rb") as f:
        X_train_raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(paths["x_test"], "rb") as f:
        X_test_raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(paths["y_train"], "rb") as f:
        y_train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(paths["y_test"], "rb") as f:
        y_test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0
    eps = 0.001
    Y_train = np.full((y_train_labels.shape[0], 10), eps, dtype=np.float32)
    Y_train[np.arange(y_train_labels.shape[0]), y_train_labels] = 1.0 - eps
    return X_train, Y_train, X_test, y_test_labels


def build_model(key):
    # Edge0 (784->512): identity activation. Edges 1,2: sigmoid. Matches
    # Deepity's own SimplePCN(Linear(784,512), Linear(512,512), Sigmoid(),
    # Linear(512,10), Sigmoid()) layer-list convention exactly (see
    # benchmark_jpc.py's docstring/build_model for the same pattern on a
    # different architecture).
    depth = len(LAYER_SIZES) - 1
    subkeys = jr.split(key, depth)
    layers = []
    for i in range(depth):
        act_fn_l = nn.Identity() if i == 0 else jax.nn.sigmoid
        linear = nn.Linear(LAYER_SIZES[i], LAYER_SIZES[i + 1], use_bias=True, key=subkeys[i])
        # Forward-projection init: uniform(-0.3, 0.3) on every weight
        # matrix, matching examples/mnist.py's rng_init.uniform(-0.3, 0.3)
        # applied to all of Deepity's weight-bearing layers.
        w_key = jr.fold_in(subkeys[i], 1)
        new_w = jr.uniform(w_key, linear.weight.shape, minval=-WEIGHT_INIT_BOUND, maxval=WEIGHT_INIT_BOUND)
        linear = eqx.tree_at(lambda l: l.weight, linear, new_w)
        layers.append(nn.Sequential([nn.Lambda(act_fn_l), linear]))
    return layers


def main() -> None:
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    key = jr.PRNGKey(SEED)
    model_key, _ = jr.split(key)
    model = build_model(model_key)

    n_batches = len(X_train) // BATCH_SIZE
    schedule = optax.exponential_decay(
        init_value=LR, transition_steps=n_batches, decay_rate=DECAY_RATE, staircase=True
    )
    optim = optax.adamw(learning_rate=schedule, weight_decay=0.0)
    opt_state = optim.init((eqx.filter(model, eqx.is_array), None))

    euler_kwargs = dict(
        ode_solver=diffrax.Euler(),
        stepsize_controller=diffrax.ConstantStepSize(),
        dt=INFERENCE_RATE,
        max_t1=STEPS * INFERENCE_RATE,
    )

    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        result = jpc.make_pc_step(
            model=model, optim=optim, opt_state=opt_state, output=y, input=x,
            loss_id="mse", **euler_kwargs,
        )
        return result["model"], result["opt_state"], result["loss"]

    @eqx.filter_jit
    def eval_batch(model, x, y_onehot):
        # jpc's own idiomatic discriminative-PC test protocol: pure
        # feedforward readout, no settling (see module docstring for why
        # this isn't matched to Deepity's settled readout).
        preds = jpc.init_activities_with_ffwd(model=model, input=x)[-1]
        acc = jpc.compute_accuracy(y_onehot, preds)
        return acc

    print(f"Architecture: {LAYER_SIZES}, batch={BATCH_SIZE}, seed={SEED}")
    print(f"Training (jpc, matched to Deepity's own protocol): {EPOCHS} epochs, "
          f"{STEPS} steps, lr={LR}, decay_rate={DECAY_RATE}...\n")

    rng = np.random.default_rng(SEED)
    start_time = time.perf_counter()
    epoch_accs = []

    for epoch in range(EPOCHS):
        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            model, opt_state, _ = train_step(model, opt_state, X_batch, Y_batch)

        N_ACC_BATCHES = 10
        accs = []
        for b in range(min(N_ACC_BATCHES, n_batches)):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            accs.append(float(eval_batch(model, X_batch, Y_batch)))

        epoch_acc = sum(accs) / len(accs)
        epoch_accs.append(epoch_acc)
        elapsed = time.perf_counter() - start_time
        print(f"Epoch {epoch + 1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}%")

    train_time = time.perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    print("\nRunning final test evaluation (jpc feedforward readout)...")
    correct = 0
    total = 0
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i:i + BATCH_SIZE]
        if len(X_batch) != BATCH_SIZE:
            continue
        preds = jpc.init_activities_with_ffwd(model=model, input=X_batch)[-1]
        pred_classes = np.argmax(np.asarray(preds), axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print("\n=== Result ===")
    print(f"jpc Peak Test Accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"\njpc per-epoch: {[round(a, 2) for a in epoch_accs]}")


if __name__ == "__main__":
    main()
