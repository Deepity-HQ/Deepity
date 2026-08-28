import numpy as np
import os
from time import perf_counter

def load_canonical_mnist():
    import gzip
    import urllib.request
    print("Fetching canonical MNIST dataset (idx-ubyte)...")
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

class NumpyPCN:
    def __init__(self, layer_sizes, seed=1234):
        self.L = len(layer_sizes) - 1
        self.W = []
        self.b = []
        rng = np.random.default_rng(seed)
        
        # Matches ngc-learn's exact initialization: uniform(-0.3, 0.3)
        for i in range(self.L):
            self.W.append(rng.uniform(-0.3, 0.3, (layer_sizes[i], layer_sizes[i+1])).astype(np.float32))
            self.b.append(np.zeros((layer_sizes[i+1],), dtype=np.float32))
        
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        
    def process(self, X, Y=None, steps=20, ir=0.04, lr=0.001, train=True):
        B = X.shape[0]
        z = [None] * (self.L + 1)
        mu = [None] * (self.L + 1)
        e = [None] * (self.L + 1)
        
        # --- PROJECTION PASS ---
        z[0] = X
        for i in range(self.L):
            phi_z = z[i] if i == 0 else sigmoid(z[i])
            mu[i+1] = phi_z @ self.W[i] + self.b[i]
            z[i+1] = mu[i+1].copy()
            
        if Y is not None:
            z[-1] = Y
            
        for i in range(1, self.L):
            e[i] = np.zeros_like(z[i])
        e[-1] = z[-1] - mu[-1]
        
        # --- E-STEP (SETTLING) ---
        for step in range(steps):
            dz = [None] * self.L
            
            # 1. Update states (Linearized approximation: NO phi_prime!)
            for i in range(1, self.L):
                feedback = e[i+1] @ self.W[i].T
                dz[i] = -z[i] - e[i] + feedback
                
            for i in range(1, self.L):
                z[i] += ir * dz[i]
                
            # 2. Compute predictions 
            for i in range(self.L):
                phi_z = z[i] if i == 0 else sigmoid(z[i])
                mu[i+1] = phi_z @ self.W[i] + self.b[i]
                
            # 3. Compute errors
            for i in range(1, self.L + 1):
                e[i] = z[i] - mu[i]
                
        energy = sum(np.sum(e[i]**2) for i in range(1, self.L + 1)) / (2.0 * B)
        
        # --- M-STEP (WEIGHT UPDATE) ---
        if train:
            self.t += 1
            for i in range(self.L):
                phi_z = z[i] if i == 0 else sigmoid(z[i])
                
                # Un-normalized gradients to match ngc-learn's HebbianSynapse raw sum
                grad_W = -(phi_z.T @ e[i+1]) 
                grad_b = -np.sum(e[i+1], axis=0)
                
                # Adam
                self.mW[i] = 0.9 * self.mW[i] + 0.1 * grad_W
                self.vW[i] = 0.999 * self.vW[i] + 0.001 * (grad_W**2)
                m_hat = self.mW[i] / (1.0 - 0.9**self.t)
                v_hat = self.vW[i] / (1.0 - 0.999**self.t)
                
                self.W[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                
                self.mb[i] = 0.9 * self.mb[i] + 0.1 * grad_b
                self.vb[i] = 0.999 * self.vb[i] + 0.001 * (grad_b**2)
                mb_hat = self.mb[i] / (1.0 - 0.9**self.t)
                vb_hat = self.vb[i] / (1.0 - 0.999**self.t)
                
                self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
                
        return mu[-1], energy

def main():
    X_train, Y_train, X_test, y_test_labels = load_canonical_mnist()
    
    BATCH_SIZE = 250 # Match train_pcn.py's mb_size=250 exactly
    STEPS = 20
    EPOCHS = 15
    
    print("\nBuilding pure NumPy PCN (Linearized + Raw Gradients)...")
    net = NumpyPCN([784, 512, 512, 10])
    
    rng = np.random.default_rng(1234)
    n_batches = len(X_train) // BATCH_SIZE
    start_time = perf_counter()
    
    for epoch in range(EPOCHS):
        indices = rng.permutation(len(X_train))
        X_shuf, Y_shuf = X_train[indices], Y_train[indices]
        
        epoch_energy = 0.0
        for b in range(n_batches):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            
            _, energy = net.process(X_batch, Y_batch, steps=STEPS, train=True)
            epoch_energy += energy
            
        correct = 0
        total = 0
        for b in range(10):
            X_batch = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Y_batch = Y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            
            # Predict uses steps=0 to measure ancestral projection like ngc-learn
            pred, _ = net.process(X_batch, None, steps=0, train=False)
            
            pred_classes = np.argmax(pred, axis=1)
            true_classes = np.argmax(Y_batch, axis=1)
            correct += np.sum(pred_classes == true_classes)
            total += BATCH_SIZE
            
        epoch_acc = 100.0 * correct / total
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | Avg Energy: {epoch_energy/n_batches:.4f}")

    print("\nRunning final test evaluation...")
    correct = 0
    total = 0
    for i in range(0, len(X_test), BATCH_SIZE):
        X_batch = X_test[i:i + BATCH_SIZE]
        y_labels_batch = y_test_labels[i:i + BATCH_SIZE]
        if len(X_batch) != BATCH_SIZE:
            continue
            
        pred_matrix, _ = net.process(X_batch, None, steps=0, train=False)
        pred_classes = np.argmax(pred_matrix, axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += BATCH_SIZE

    test_acc = 100.0 * correct / total
    print(f"NumPy PCN test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
