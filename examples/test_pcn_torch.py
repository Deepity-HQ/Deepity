import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Import the pcn-torch modules based on your setup
from pcn_torch import PredictiveCodingNetwork, TrainConfig, train_pcn, test_pcn, RichCallback

def load_full_mnist_torch():
    print("Fetching full MNIST dataset (70,000 images)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Keep the same normalization: [-1.0, 1.0]
    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int)

    # pcn-torch 'classification' mode expects standard class indices (0-9), 
    # so we skip the manual Y_bipolar mapping here.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=60000, test_size=10000, stratify=y, random_state=42
    )

    # Convert directly to PyTorch tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    return X_train_t, y_train_t, X_test_t, y_test_t

def main():
    X_train_t, y_train_t, X_test_t, y_test_t = load_full_mnist_torch()

    BATCH_SIZE = 256
    EPOCHS = 3
    
    print(f"\nBuilding PyTorch DataLoader (batch_size={BATCH_SIZE})...")
    # PyTorch DataLoaders handle the batch slicing and shuffling natively
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    print("\nInitializing pcn-torch model...")
    # Structuring a dense network: 784 (input) -> 256 (hidden) -> 128 (top latent) -> 10 (output)
    model = PredictiveCodingNetwork(
        dims=[784, 512],   
        activation="tanh",      # Matching the activation from the pydeepity setup
        output_dim=10,
        mode="classification",
    )

    print("\nConfiguring training parameters...")
    config = TrainConfig(
        task="classification",
        T_infer=60,            # Matches your 150 inference steps
        lr_infer=0.05,          
        lr_learn=0.005,         # Equivalent to your global INITIAL_LR
        num_epochs=EPOCHS,
        callback=RichCallback(),
    )

    print("\nStarting pcn-torch training...")
    history = train_pcn(model, train_loader, config)

    print("\nEvaluating on held-out test set...")
    results = test_pcn(model, test_loader, config)
    
    print(f"\nTest Accuracy: {results['accuracy']:.2%}")

if __name__ == "__main__":
    main()
