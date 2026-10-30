import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from time import perf_counter

# CUDA counterpart to examples/test_ffnn_deep.py -- IDENTICAL architecture,
# batch size, LR/decay schedule, and MSE-vs-one-hot objective; only
# difference is device below. Matches examples/mnist_cuda.py's SimplePCN
# config for a direct backprop-vs-PC wall-clock comparison on the same GPU.


def load_full_mnist():
    import gzip
    import os
    import urllib.request
    print("Fetching canonical MNIST dataset (idx-ubyte, matching ngc-learn exactly)...")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files: dict[str, str] = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
        "x_test": "t10k-images-idx3-ubyte.gz",
        "y_test": "t10k-labels-idx1-ubyte.gz"
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


class FFNN(nn.Module):
    """784 -> 512 -> 512 -> 10. Matches SimplePCN's exact layer/activation
    placement: the first Linear has no activation, the next two each have
    a trailing sigmoid."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def main() -> None:
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    torch.manual_seed(SEED)

    X_train, Y_train, X_test, y_test_labels = load_full_mnist()

    BATCH_SIZE = 250
    LR = 0.00373
    DECAY_RATE = 0.94

    if not torch.cuda.is_available():
        raise RuntimeError("test_ffnn_deep_cuda.py requires a CUDA device; "
                            "use examples/test_ffnn_deep.py for the CPU control.")
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test_labels.astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    model = FFNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
    criterion = nn.MSELoss()

    print(f"\nTraining (backprop control, CUDA): {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"architecture=784->512->512->10 (sigmoid, matches SimplePCN exactly), "
          f"lr={LR}, decay_rate={DECAY_RATE}, MSE loss vs one-hot...\n")

    # Ensures the first batch's H2D transfer + kernel launch/compile latency
    # isn't folded into epoch 1's reported time -- matched against Deepity/
    # jpc's own CUDA warm-up handling for a fair wall-clock comparison.
    torch.cuda.synchronize()
    start_time = perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | avg loss: {epoch_loss/n_batches:.4f} | elapsed: {elapsed:.1f}s")

    torch.cuda.synchronize()
    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.\n")

    print("Running final test evaluation...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X_test_t), BATCH_SIZE):
            Xb = X_test_t[i:i + BATCH_SIZE].to(device)
            yb = y_test_t[i:i + BATCH_SIZE].to(device)
            if len(Xb) != BATCH_SIZE:
                continue
            out = model(Xb)
            pred = torch.argmax(out, dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    test_acc = 100.0 * correct / total

    print(f"\n=== Result ===")
    print(f"Backprop (PyTorch, CUDA) test accuracy: {test_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")


if __name__ == "__main__":
    main()
