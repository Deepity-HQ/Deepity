import pydeepity as pd
import numpy as np
import time

X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float32)
Y = np.array([[-1], [1], [1], [-1]], dtype=np.float32)

SETTLING_STEPS = 100

if __name__ == "__main__":
    pcn = pd.SimplePCN(
        pd.layer.Linear(2, 8),
        pd.layer.TanH(),
        pd.layer.Linear(8, 1),
        pd.layer.TanH(),
        batch_size=4
    )

    pcn.configure(
        learning_rate=0.1,
        inference_rate=0.15,
        lmbda=0.0,
        optimizer="SGD",
        seed=42
    )

    start = time.perf_counter()
    for epoch in range(1500):
        pcn.train_step(X, Y, SETTLING_STEPS)
    print(f"Elapsed: {time.perf_counter() - start:.4f}s\n")

    preds = pcn.predict(X, SETTLING_STEPS)
    print("Predictions:")
    for i in range(4):
        print(f"Input: {X[i]}, Target: {Y[i][0]} -> Prediction: {preds[i][0]:.4f}")
