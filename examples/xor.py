import numpy as np
from pydeepity import SequentialPCN, Linear, TanH


def main():
    # 1. Setup the Network
    #
    # Architecture is declared with Linear/Activation objects; the final
    # Linear(1, 0) is the terminal "sink" layer that holds the output
    # belief itself (clamped to Y during training), per this codebase's
    # convention of one extra weight-free layer past the last real weight
    # matrix.
    net = SequentialPCN(
        Linear(2, 8), TanH(),
        Linear(8, 1), TanH(),
        Linear(1, 0),
        batch_size=1,
    )

    # Hyperparameters and backend construction happen in configure(), not
    # the constructor -- it calls compile() before randomize_weights()
    # internally (required ordering, see CONTRIBUTING.md).
    net.configure(
        learning_rate=0.05,
        inference_rate=0.3,
        precision_rate=0.0,
        lmbda=0.0001,
        optimizer="SGD",
    )

    # 2. Data
    X = np.array([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0]
    ], dtype=np.float32)

    Y = np.array([
        [-1.0],
        [ 1.0],
        [ 1.0],
        [-1.0]
    ], dtype=np.float32)

    # 3. Training Loop
    epochs = 2000
    inference_steps = 50
    report_every = 500

    print("Starting Discriminative PC XOR Test...")

    for epoch in range(epochs):
        total_energy = 0.0

        # Shuffle the training order to prevent oscillation
        indices = np.random.permutation(4)

        for idx in indices:
            total_energy += net.train_step(X[idx], Y[idx], steps=inference_steps)

        if epoch % report_every == 0:
            avg_energy = total_energy / (4 * inference_steps)
            print(f"Epoch {epoch:5d} | Energy: {avg_energy:.4f}")

    # 4. Testing
    print("\n=== Predictions ===")
    correct = 0

    for i in range(len(X)):
        x_sample = X[i]
        target = Y[i][0]

        # predict() returns a (batch_size, output_size) beliefs array, i.e.
        # shape (1, 1) here -- pull out the scalar before using it as a
        # number (an array can't be fed straight into an f-string ':.4f').
        pred = float(net.predict(x_sample, steps=inference_steps).reshape(-1)[0])

        sign_correct = (pred > 0 and target > 0) or (pred < 0 and target < 0)
        if sign_correct:
            correct += 1

        status = "OK" if sign_correct else "WRONG"
        print(f"Input: [{x_sample[0]:>4.1f}, {x_sample[1]:>4.1f}] | Target: {target:>4.1f} | Pred: {pred:>6.4f} {status}")

    print(f"\nAccuracy: {correct}/4")
    if correct == 4:
        print("XOR LEARNED SUCCESSFULLY!")
    else:
        print("XOR NOT FULLY LEARNED.")

if __name__ == "__main__":
    main()
