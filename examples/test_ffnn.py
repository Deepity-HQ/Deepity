import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from time import perf_counter

# Standard backprop feedforward baseline -- matches SequentialPCN's EXACT
# architecture (784 -> 512 -> 10, single hidden layer, tanh) for a true
# apples-to-apples comparison: same network shape, same data, different
# training algorithm (ordinary backprop vs. predictive coding settling).
#
# Worth knowing going in: PC networks are generally understood in the
# literature to be substantially slower than backprop to reach comparable
# accuracy, because of the iterative settling cost per batch (T_infer steps
# of forward+backward-like computation, vs. backprop's single forward +
# single backward pass). This script exists to measure that gap directly
# for this specific architecture/dataset, not to assume the answer.


def load_full_mnist_torch():
    print("Fetching full MNIST dataset (70,000 images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')

    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, stratify=y, random_state=42
    )

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    return X_train_t, y_train_t, X_test_t, y_test_t


class FFNN(nn.Module):
    """784 -> 512 -> 10, tanh hidden activation -- matches SequentialPCN's
    add_layer(784, 512, act="tanh") / add_layer(512, 10, act="tanh") /
    add_layer(10, 0, act="linear") exactly in shape and hidden activation.
    Output layer is linear (raw logits), standard for CrossEntropyLoss."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, test_loader, device):
    """Full test-set accuracy. Cheap here since inference is a single
    forward pass (no iterative settling like the PC variants) -- running
    this every epoch adds negligible overhead relative to training time."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    model.train()
    return 100.0 * correct / total


def main():
    X_train_t, y_train_t, X_test_t, y_test_t = load_full_mnist_torch()

    BATCH_SIZE = 256
    EPOCHS = 50  # matching the PC run's epoch count for direct comparison

    print(f"\nBuilding PyTorch DataLoader (batch_size={BATCH_SIZE})...")
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FFNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining: {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"architecture=784->512->10 (matches SequentialPCN exactly)...\n")

    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        epoch_acc = evaluate(model, test_loader, device)
        epoch_accs.append(epoch_acc)
        elapsed = perf_counter() - start_time

        print(f"  Epoch {epoch+1}/{EPOCHS} | avg loss: {avg_loss:.4f} | "
              f"test acc: {epoch_acc:.2f}% | elapsed: {elapsed:.1f}s")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.\n")

    final_acc = epoch_accs[-1]
    print(f"\nTest Accuracy: {final_acc:.2f}%")
    print(f"\n--- Summary ---")
    print(f"Architecture: 784 -> 512 -> 10 (tanh hidden)")
    print(f"Training time: {train_time:.1f}s for {EPOCHS} epochs")
    print(f"Test accuracy: {final_acc:.2f}%")
    print(f"\nPer-epoch test accuracy: {[round(a, 2) for a in epoch_accs]}")


if __name__ == "__main__":
    main()
