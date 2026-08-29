import numpy as np
import os
import sys
import optuna
from time import perf_counter
from pydeepity import SimplePCN

def load_full_mnist():
    import gzip
    import urllib.request

    print("Fetching canonical MNIST dataset...")
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


def objective(trial, X_train, Y_train, X_test, y_test_labels):
    # 1. Define the dynamic search space
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    ir = trial.suggest_float("ir", 0.02, 0.15)
    steps = trial.suggest_int("steps", 10, 30, step=5)
    decay = trial.suggest_float("decay", 0.90, 1.0)
    
    batch_size = 250
    epochs = 5  # Screening epochs

    net = SimplePCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=lr, ir=ir, act="linear", lmbda=0.0)
    net.add_layer(512, 512, lr=lr, ir=ir, act="sigmoid", lmbda=0.0)
    net.add_layer(512, 10,  lr=lr, ir=ir, act="sigmoid", lmbda=0.0)
    net.add_layer(10, 0,    lr=lr, ir=ir, act="linear", lmbda=0.0)
    
    net.set_optimizer("ADAMW")
    net.compile()
    net.set_mu_cache_threshold(0.0)
    
    net.randomize_weights()
    rng_init = np.random.default_rng(7)
    for layer in net.layers[:-1]:
        w_shape = layer.weights.shape
        layer.weights[:] = rng_init.uniform(-0.3, 0.3, w_shape).astype(np.float32)

    rng = np.random.default_rng(7)
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        current_lr = lr * (decay ** epoch)
        net.set_learning_rate(current_lr)

        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]

        for b in range(n_batches):
            X_batch = X_shuf[b * batch_size:(b + 1) * batch_size]
            Y_batch = Y_shuf[b * batch_size:(b + 1) * batch_size]
            net.train_step_with_projection(X_batch, Y_batch, steps)

        # 2. Fast intermediate evaluation for Pruning
        correct = 0
        total = 0
        # Evaluate on a 2,500 sample subset to keep trial time ultra-low
        val_subset_size = 2500 
        for i in range(0, val_subset_size, batch_size):
            X_batch = X_test[i:i + batch_size]
            y_labels_batch = y_test_labels[i:i + batch_size]

            net.reset_state()
            net.clamp_input(X_batch)
            flat_beliefs = net.predict_with_projection(X_batch, steps)
            terminal_beliefs = flat_beliefs.reshape(batch_size, 10)

            pred_classes = np.argmax(terminal_beliefs, axis=1)
            correct += np.sum(pred_classes == y_labels_batch)
            total += batch_size

        intermediate_acc = 100.0 * correct / total
        
        # 3. Report back to Optuna to see if we should kill this trial early
        trial.report(intermediate_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return intermediate_acc

def main() -> None:
    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    # Create an Optuna study with the MedianPruner
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )
    
    print("\nStarting Optuna Hyperparameter Optimization...")
    start_time = perf_counter()
    
    # Run 30 trials (Optuna is smart enough to find the optimum fast)
    study.optimize(lambda trial: objective(trial, X_train, Y_train, X_test, y_test_labels), n_trials=30)
    
    total_time = perf_counter() - start_time
    print(f"\nOptimization complete in {total_time:.1f}s.")
    
    print("\n=== Best Trial ===")
    print(f"Accuracy: {study.best_trial.value:.2f}%")
    print("Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
