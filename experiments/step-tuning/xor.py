import numpy as np
from deepitybind.sequential_pcn import SequentialPCN

def main():
    # 1. Setup the Network
    net = SequentialPCN(batch_size=1)
    
    # Using the abstracted add_layer (automatically handles 'd' + act)
    net.add_layer(2, 8, lr=0.05, ir=0.3, act="tanh", lmbda=0.0001)
    net.add_layer(8, 1, lr=0.05, ir=0.3, act="tanh", lmbda=0.0001)
    net.add_layer(1, 0, lr=0.05, ir=0.3, act="linear", lmbda=0.0001)

    net.randomize_weights()

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
    epochs = 5000
    inference_steps = 50
    report_every = 500

    print("Starting Discriminative PC XOR Test...")

    for epoch in range(epochs):
        total_energy = 0.0
        
        # Shuffle the training order to prevent oscillation
        indices = np.random.permutation(4)
        
        for idx in indices:
            # Using the abstracted train_step
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
        
        # Using the abstracted predict
        pred = net.predict(x_sample, steps=inference_steps)
        
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