import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import ConvolutionalPCN

# Same architecture as test.py, but a much smaller, much cheaper diagnostic:
# does energy for a SINGLE batch actually decrease as inference steps grow?
# If flat, MNIST's real issue isn't "not enough steps" -- something else
# needs investigating (rates, or a scale-specific bug not caught by the
# small-toy gradient checks).

def load_small_mnist(n=512):
    print("Fetching a small MNIST slice...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int)
    X_train, _, y_train, _ = train_test_split(X, y, train_size=n, random_state=42, stratify=y)

    Y_bipolar = np.full((n, 10), -0.9, dtype=np.float32)
    Y_bipolar[np.arange(n), y_train] = 0.9
    return X_train, Y_bipolar


def build_net(batch_size):
    net = ConvolutionalPCN(batch_size=batch_size)
    net.add_layer(out_channels=16, kernel_h=5, kernel_w=5, in_channels=1, in_height=28, in_width=28,
                  stride_h=2, stride_w=2, pad_h=2, pad_w=2, lr=0.01, ir=0.05, pr=0.0, lmbda=0.0001, act="tanh")
    net.add_layer(out_channels=32, kernel_h=5, kernel_w=5, stride_h=2, stride_w=2, pad_h=2, pad_w=2,
                  lr=0.01, ir=0.05, pr=0.0, lmbda=0.0001, act="tanh")
    net.add_layer(out_channels=10, kernel_h=7, kernel_w=7, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                  lr=0.01, ir=0.05, pr=0.0, lmbda=0.0001, act="tanh")
    net.add_layer(out_channels=0, kernel_h=1, kernel_w=1, lr=0.01, ir=0.05, pr=0.0, lmbda=0.0001, act="linear")
    net.compile()
    net.randomize_weights()
    return net


def main():
    BATCH_SIZE = 32
    X, Y = load_small_mnist(n=512)
    X_batch = X[:BATCH_SIZE]
    Y_batch = Y[:BATCH_SIZE]

    print(f"\n=== Diagnostic: does energy converge with more settling steps? ===")
    print(f"(Single fixed batch, WEIGHTS FROZEN -- no update_weights call -- "
          f"isolating pure inference/settling dynamics from learning.)\n")

    net = build_net(BATCH_SIZE)

    # Manually drive clamp/settle WITHOUT calling train_step, so weights
    # never update -- isolates whether settling itself converges, same
    # "weights frozen" isolation test used earlier this session for the
    # dense network.
    net.reset_state()
    net.clamp_input(X_batch.flatten())
    net[-1].clamp_state(Y_batch.flatten())

    STEPS_TO_CHECK = [10, 30, 60, 100, 150, 250, 400]
    last_step = 0
    for target_step in STEPS_TO_CHECK:
        for _ in range(target_step - last_step):
            energy = net.calculate_state()
            net.update_state()
        last_step = target_step
        print(f"  after {target_step:4d} total steps: energy = {energy:12.4f}")

    print("\nIf energy is still falling sharply by step 400, 30 steps was "
          "far too few -- that's the fix. If it plateaus early (by ~60-100) "
          "at a value still this large, the issue is likely elsewhere "
          "(rates, or something scale-specific the toy gradient checks "
          "couldn't have caught).")


if __name__ == "__main__":
    main()
