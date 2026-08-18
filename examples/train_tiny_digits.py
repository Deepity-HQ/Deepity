from typing import cast
import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from pydeepity import SequentialPCN
from time import perf_counter

def load_tiny_digits() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32],
    npt.NDArray[np.float32], npt.NDArray[np.int64],
]:
    print("Loading tiny 8x8 Digits dataset (1,797 images)...")
    # load_digits()'s stub return type is an unconditional Union covering
    # both the Bunch and (X, y) tuple shapes regardless of arguments, so
    # pyright can't narrow it on its own; cast to the shape we actually get
    # at runtime here (no return_X_y passed -> Bunch).
    digits = cast(Bunch, load_digits())
    
    # Digits are 0-16 natively. Scale to [-1.0, 1.0] for tanh
    X = (digits.data.astype(np.float32) / 8.0) - 1.0 
    y = digits.target
    
    # Soft-Bipolar Targets [-0.9, 0.9]
    Y_bipolar = np.full((y.shape[0], 10), -0.9, dtype=np.float32)
    Y_bipolar[np.arange(y.shape[0]), y] = 0.9

    X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = train_test_split(
        X, Y_bipolar, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return cast(
        "tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int64]]",
        (X_train, Y_train, X_test, y_test_labels),
    )

def main():
    X_train, Y_train, X_test, y_test_labels = load_tiny_digits()
    
    # Smaller batch size for the tiny dataset
    BATCH_SIZE = 32 
    net = SequentialPCN(batch_size=BATCH_SIZE)
    
    # === SCHEDULE SETUP ===
    initial_lr = 0.06
    decay_rate = 0.98 
    
    # Input is now 64 (8x8 pixels) -> 32 hidden -> 10 classes
    net.add_layer(64, 32, lr=initial_lr, ir=0.08, pr=0., act="tanh", lmbda=0.0001)
    net.add_layer(32, 10, lr=initial_lr, ir=0.08, pr=0., act="tanh", lmbda=0.0001)
    net.add_layer(10,  0, lr=initial_lr, ir=0.08, pr=0., act="linear", lmbda=0.0001)

    net.compile()
    net.randomize_weights()

    epochs = 15 # Only 15 epochs needed for this tiny set
    current_steps = 60
    total_batches = len(X_train) // BATCH_SIZE
    
    print(f"\nStarting Rapid PCN Training ({len(X_train)} samples, Batch Size: {BATCH_SIZE})...")
    start = perf_counter()

    for epoch in range(epochs):
        current_lr = initial_lr * (decay_rate ** epoch)
        net.set_learning_rate(current_lr)
        total_energy = 0.0
        
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        batches_processed = 0
        
        for i in range(0, len(X_train_shuffled), BATCH_SIZE):
            X_batch = X_train_shuffled[i : i + BATCH_SIZE]
            Y_batch = Y_train_shuffled[i : i + BATCH_SIZE]
            
            if len(X_batch) != BATCH_SIZE:
                continue
                
            energy = net.train_step(X_batch.flatten(), Y_batch.flatten(), steps=current_steps)
            total_energy += energy
            batches_processed += 1
            
        # Print every epoch since it is so fast
        print(f"  Epoch {epoch+1:02d}/{epochs}"
              f" | LR: {current_lr:.6f}"
              f" | Avg Batch Energy: {total_energy / (batches_processed * current_steps):.4f}"
              f" | Time: {(perf_counter() - start):.5f} sec")
        start = perf_counter()
                
    print("\n=== Evaluating on Test Set ===")
    correct = 0
    total = 0
    
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i : i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i : i + BATCH_SIZE]
        
        if len(X_batch) != BATCH_SIZE:
            continue
            
        pred_matrix = net.predict(X_batch, steps=current_steps)
        pred_classes = np.argmax(pred_matrix, axis=1)
        
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    # Save to verify the ActivationType bug is fixed!
    print("\nSaving checkpoint to 'checkpoints/fast_test_pcn'...")
    net.save("checkpoints/fast_test_pcn")
    print("Done! Go check manifest.json.")

if __name__ == "__main__":
    main()