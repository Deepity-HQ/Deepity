import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SequentialPCN
from time import perf_counter
import itertools

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

def train_and_evaluate(X_train, Y_train, X_test, y_test_labels, 
                       steps, lr, epochs, batch_size, decay_rate=0.89):
    
    # Isolate network initialization for a fair, fresh start per run
    net = SequentialPCN(batch_size=batch_size)
    net.add_layer(784, 512, lr=lr, ir=0.08, pr=0., act="tanh", lmbda=0.0001)
    net.add_layer(512,  10, lr=lr, ir=0.08, pr=0., act="tanh", lmbda=0.0001)
    net.add_layer( 10,   0, lr=lr, ir=0.08, pr=0., act="linear", lmbda=0.0001)
    
    net.compile()
    net.randomize_weights()

    start_time = perf_counter()
    
    for epoch in range(epochs):
        current_lr = lr * (decay_rate ** epoch)
        net.set_learning_rate(current_lr)
        
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled[i : i + batch_size]
            Y_batch = Y_train_shuffled[i : i + batch_size]
            
            if len(X_batch) != batch_size:
                continue
                
            net.train_step(X_batch.flatten(), Y_batch.flatten(), steps=steps)

    train_time = perf_counter() - start_time
    
    # Evaluation phase
    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i : i + batch_size]
        y_labels_batch = y_test_labels[i : i + batch_size]
        
        if len(X_batch) != batch_size:
            continue
            
        pred_matrix = net.predict(X_batch, steps=steps)
        pred_classes = np.argmax(pred_matrix, axis=1)
        
        correct += np.sum(pred_classes == y_labels_batch)
        total += batch_size

    accuracy = (correct / total) * 100
    return train_time, accuracy

def main():
    X_train, Y_train, X_test, y_test_labels = load_full_mnist()
    
    # ==========================================
    # EXPERIMENT 1: STEP SWEEP
    # ==========================================
    print("\n--- EXPERIMENT 1: RELAXATION STEPS ---")
    steps_to_test = [55, 60, 65, 70, 80, 90, 100, 120]
    exp1_results = []
    
    for s in steps_to_test:
        print(f"Testing {s} steps...")
        t, acc = train_and_evaluate(X_train, Y_train, X_test, y_test_labels, 
                                    steps=s, lr=0.06, epochs=1, batch_size=256)
        exp1_results.append((s, t, acc))
        print(f"  -> Time: {t:.2f}s | Acc: {acc:.2f}%")
        
    print("\nExperiment 1 Summary:")
    print(f"{'Steps':<10} | {'Time (s)':<15} | {'Accuracy (%)':<15}")
    print("-" * 45)
    for s, t, acc in exp1_results:
        print(f"{s:<10} | {t:<15.2f} | {acc:<15.2f}")
        
    # ==========================================
    # FUTURE EXPERIMENTS (Uncomment as needed)
    # ==========================================
    
    """
    # EXPERIMENT 2: LR x Steps Grid
    print("\n--- EXPERIMENT 2: LR x STEPS ---")
    best_steps = [60, 80] # TODO: Update this list based on Exp 1 peaks
    lrs = [0.02, 0.03, 0.04, 0.06]
    
    for s, lr in itertools.product(best_steps, lrs):
        t, acc = train_and_evaluate(X_train, Y_train, X_test, y_test_labels, 
                                    steps=s, lr=lr, epochs=1, batch_size=256)
        print(f"Steps: {s:<4} | LR: {lr:<5} -> Acc: {acc:.2f}%")
        
    # EXPERIMENT 3: Epoch Scaling
    print("\n--- EXPERIMENT 3: EPOCHS ---")
    best_s = 60    # TODO: Update based on Exp 2
    best_lr = 0.06 # TODO: Update based on Exp 2
    epochs_to_test = [1, 2, 3, 5, 10]
    
    for e in epochs_to_test:
        t, acc = train_and_evaluate(X_train, Y_train, X_test, y_test_labels, 
                                    steps=best_s, lr=best_lr, epochs=e, batch_size=256)
        print(f"Epochs: {e:<4} -> Time: {t:.2f}s | Acc: {acc:.2f}%")
        
    # EXPERIMENT 4: Batch Size Scaling
    print("\n--- EXPERIMENT 4: BATCH SIZE ---")
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    
    for b in batch_sizes:
        t, acc = train_and_evaluate(X_train, Y_train, X_test, y_test_labels, 
                                    steps=best_s, lr=best_lr, epochs=1, batch_size=b)
        print(f"Batch: {b:<5} -> Time: {t:.2f}s | Acc: {acc:.2f}%")
    """

if __name__ == "__main__":
    main()