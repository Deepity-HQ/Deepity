import numpy as np
import os
from time import perf_counter
from pydeepity import SimplePCN, GaussSeidelPCN, ConvolutionalPCN, SimpleConvolutionalPCN

def load_full_mnist():
    import gzip
    import urllib.request
    
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

    # Standardized Normalization
    with gzip.open(paths["x_train"], 'rb') as f:
        X_train_raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        X_train = (X_train_raw.astype(np.float32) / 255.0 - 0.1307) / 0.3081
        
    with gzip.open(paths["x_test"], 'rb') as f:
        X_test_raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        X_test = (X_test_raw.astype(np.float32) / 255.0 - 0.1307) / 0.3081
        
    with gzip.open(paths["y_train"], 'rb') as f:
        y_train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(paths["y_test"], 'rb') as f:
        y_test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    eps = 0.001
    Y_train = np.full((y_train_labels.shape[0], 10), eps, dtype=np.float32)
    Y_train[np.arange(y_train_labels.shape[0]), y_train_labels] = 1.0 - eps

    return X_train, Y_train, X_test, y_test_labels

def run_benchmark(name, net, X_train, Y_train, X_test, y_test_labels, lr, steps, epochs=3, decay=0.9):
    print(f"\n[{name}] Starting {epochs}-Epoch Sprint...")
    BATCH_SIZE = 250
    rng = np.random.default_rng(7)
    n_batches = len(X_train) // BATCH_SIZE
    
    start_time = perf_counter()
    
    for epoch in range(epochs):
        current_lr = lr * (decay ** epoch)
        net.set_learning_rate(current_lr)
        
        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]
        
        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            
            net.train_step_with_projection(X_batch, Y_batch, steps)
            
            # If the network requires manual precision updates (like standard ConvPCN)
            if hasattr(net, 'update_precision'):
                net.update_precision()
                
    train_time = perf_counter() - start_time
    
    # Fast Evaluation
    correct = 0
    total = 0
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i:i + BATCH_SIZE]
        if len(X_batch) != BATCH_SIZE:
            continue

        # Safely handle Gauss-Seidel's lack of a separate projection prediction method
        if hasattr(net, 'predict_with_projection'):
            flat_beliefs = np.array(net.predict_with_projection(X_batch, steps), dtype=np.float32)
        else:
            flat_beliefs = np.array(net.predict(X_batch, steps), dtype=np.float32)
            
        terminal_beliefs = flat_beliefs.reshape(BATCH_SIZE, 10)
        
        pred = np.argmax(terminal_beliefs, axis=1)
        correct += np.sum(pred == y_labels_batch)
        total += BATCH_SIZE
        
    accuracy = 100.0 * correct / total
    print(f"[{name}] Finished in {train_time:.1f}s | Test Acc: {accuracy:.2f}%")
    return train_time, accuracy

def main():
    X_train, Y_train, X_test, y_test_labels = load_full_mnist()
    BATCH_SIZE = 250
    results = {}

    # 1. SimplePCN (Jacobi Dense)
    net_simple = SimplePCN(batch_size=BATCH_SIZE)
    net_simple.add_layer(784, 512, lr=0.003, ir=0.1, act="relu")
    net_simple.add_layer(512, 10, lr=0.003, ir=0.1, act="linear")
    net_simple.add_layer(10, 0, lr=0.003, ir=0.1, act="linear")
    net_simple.set_optimizer("ADAMW")
    net_simple.compile()
    net_simple.randomize_weights()
    results["Jacobi Dense"] = run_benchmark("Jacobi Dense", net_simple, X_train, Y_train, X_test, y_test_labels, lr=0.003, steps=30)

    # 2. GaussSeidelPCN (Optimized Dense)
    net_gs = GaussSeidelPCN(batch_size=BATCH_SIZE)
    net_gs.add_layer(784, 512, lr=0.003, ir=0.1, act="relu")
    net_gs.add_layer(512, 10, lr=0.003, ir=0.1, act="linear")
    net_gs.add_layer(10, 0, lr=0.003, ir=0.1, act="linear")
    net_gs.set_optimizer("ADAMW")
    net_gs.compile()
    net_gs.randomize_weights()
    results["Gauss-Seidel Dense"] = run_benchmark("Gauss-Seidel Dense", net_gs, X_train, Y_train, X_test, y_test_labels, lr=0.003, steps=30)

    # 3. SimpleConvPCN (AdamW CNN)
    net_sconv = SimpleConvolutionalPCN(batch_size=BATCH_SIZE)
    net_sconv.add_layer(out_channels=16, kernel_h=4, kernel_w=4, in_channels=1, in_height=28, in_width=28, stride_h=2, stride_w=2, pad_h=1, pad_w=1, lr=0.00105, ir=0.015, act="relu")
    net_sconv.add_layer(out_channels=32, kernel_h=4, kernel_w=4, stride_h=2, stride_w=2, pad_h=1, pad_w=1, lr=0.00105, ir=0.015, act="relu")
    net_sconv.add_layer(out_channels=128, kernel_h=7, kernel_w=7, stride_h=1, stride_w=1, pad_h=0, pad_w=0, lr=0.00105, ir=0.015, act="relu")
    net_sconv.add_layer(out_channels=10, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, lr=0.00105, ir=0.015, act="linear")
    net_sconv.add_layer(out_channels=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, lr=0.00105, ir=0.015, act="linear")
    net_sconv.set_optimizer("ADAMW")
    net_sconv.compile()
    net_sconv.randomize_weights()
    results["AdamW CNN"] = run_benchmark("AdamW CNN", net_sconv, X_train, Y_train, X_test, y_test_labels, lr=0.00105, steps=25)

    # 4. ConvPCN (Standard SGD)
    net_conv = ConvolutionalPCN(batch_size=BATCH_SIZE)
    net_conv.add_layer(out_channels=16, kernel_h=4, kernel_w=4, in_channels=1, in_height=28, in_width=28, stride_h=2, stride_w=2, pad_h=1, pad_w=1, lr=0.0001, ir=0.015, pr=0.001, act="relu")
    net_conv.add_layer(out_channels=32, kernel_h=4, kernel_w=4, stride_h=2, stride_w=2, pad_h=1, pad_w=1, lr=0.0001, ir=0.015, pr=0.001, act="relu")
    net_conv.add_layer(out_channels=128, kernel_h=7, kernel_w=7, stride_h=1, stride_w=1, pad_h=0, pad_w=0, lr=0.0001, ir=0.015, pr=0.001, act="relu")
    net_conv.add_layer(out_channels=10, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, lr=0.0001, ir=0.015, pr=0.001, act="linear")
    net_conv.add_layer(out_channels=0, kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, pad_h=0, pad_w=0, lr=0.0001, ir=0.015, pr=0.001, act="linear")
    net_conv.compile()
    net_conv.randomize_weights()
    results["Standard SGD CNN"] = run_benchmark("Standard SGD CNN", net_conv, X_train, Y_train, X_test, y_test_labels, lr=0.0001, steps=25)

    print("\n" + "="*40)
    print("🏆 DEEPITY CPU BENCHMARK SCOREBOARD 🏆")
    print("="*40)
    print(f"{'Architecture':<20} | {'Time (s)':<10} | {'Test Acc':<10}")
    print("-" * 45)
    for name, (time, acc) in results.items():
        print(f"{name:<20} | {time:<10.1f} | {acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
