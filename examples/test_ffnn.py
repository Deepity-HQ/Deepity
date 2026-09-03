import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from time import perf_counter

# Standard backprop feedforward baseline -- matches DKPPCN EXACTLY
# architecture (784 -> 512 -> 10, single hidden layer, sigmoid) for a true
# apples-to-apples comparison: same network shape, same data, same LR decay,
# same batch size, but ordinary backprop vs. predictive coding settling.

def load_full_mnist_torch():
    print("Fetching full MNIST dataset (70,000 images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')

    # MATCHED: Scale from 0 to 1 exactly like the C++ script
    X = X.astype(np.float32) / 255.0
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
    """784 -> 512 -> 10, sigmoid hidden activation."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.Sigmoid(), # MATCHED: act="sigmoid" from DKPPCN
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, test_loader, device):
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

    BATCH_SIZE = 250 # MATCHED: DKPPCN uses 250
    EPOCHS = 50 

    print(f"\nBuilding PyTorch DataLoader (batch_size={BATCH_SIZE})...")
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    # MATCHED: Hardcode to CPU to compare against C++ CPU performance
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model = FFNN().to(device)
    
    # MATCHED: Initial LR and Decay Rate from DKPPCN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00373)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining: {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"architecture=784->512->10 (matches DKPPCN exactly)...\n")

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
            
        # MATCHED: Step the LR decay per epoch
        scheduler.step()

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
    print(f"Architecture: 784 -> 512 -> 10 (sigmoid hidden)")
    print(f"Training time: {train_time:.1f}s for {EPOCHS} epochs")
    print(f"Test accuracy: {final_acc:.2f}%")
    print(f"\nPer-epoch test accuracy: {[round(a, 2) for a in epoch_accs]}")

if __name__ == "__main__":
    main()
