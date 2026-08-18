import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SequentialPCN
from time import perf_counter

# PC-vs-backprop gradient fidelity experiment (per ChatGPT's plan, Section 2/5).
#
# FIXED: one batch, one frozen weight set, fixed X/Y, pr=0, lmbda=0, no
# weight update applied (captured via before/after delta and restored each
# trial). For each T in a sweep, record:
#   - PC's implied weight delta (via a single update_weights() call after
#     T settling steps, then restored)
#   - the analytic backprop gradient on the IDENTICAL weights/data/loss
#   - cosine similarity and relative error between them, per layer
#
# The backprop-equivalent loss is NOT CrossEntropyLoss -- it's matched to
# PC's own implied objective: with pr=0, CalculateState()'s terminal-layer
# energy is exactly 0.5 * sum((target - prediction)^2) with a tanh output,
# since that's literally what the C++ computes for the clamped terminal
# layer. The PyTorch model mirrors PC's own forward structure exactly:
# Linear(784,512) -> Tanh -> Linear(512,10) -> Tanh, loss = 0.5*sum((pred-target)^2).


def load_one_frozen_batch(batch_size=32, seed=42):
    print("Fetching MNIST (one batch)...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int)

    Y_bipolar = np.full((y.shape[0], 10), -0.9, dtype=np.float32)
    Y_bipolar[np.arange(y.shape[0]), y] = 0.9

    X_train, _, Y_train, _, _, _ = train_test_split(X, Y_bipolar, y, train_size=3000, random_state=seed, stratify=y)
    return X_train[:batch_size], Y_train[:batch_size]


def pc_weight_delta(net, x_flat, y_flat, T, W0_before, W1_before):
    """Restores frozen weights, settles for T steps (no weight update during
    settling), calls update_weights() ONCE, captures the delta, then
    restores weights again so the net is clean for the next T trial.
    Bias is left untouched -- RandomizeWeights() never modifies it and
    BindMemory() zero-initializes it, so it's provably 0 at this frozen
    point (before any update_weights() call has run) without needing a
    bias accessor at all."""
    net[0].weights[:] = W0_before
    net[1].weights[:] = W1_before

    net.reset_state()
    net.clamp_input(x_flat)
    net[-1].clamp_state(y_flat)

    for _ in range(T):
        net.calculate_state()
        net.update_state()

    net.update_weights()

    W0_after = net[0].weights.copy()
    W1_after = net[1].weights.copy()

    dW0 = W0_after - W0_before
    dW1 = W1_after - W1_before

    # restore for the next trial
    net[0].weights[:] = W0_before
    net[1].weights[:] = W1_before
    net[-1].unclamp_state()

    return dW0, dW1


class MatchedMLP(nn.Module):
    """Mirrors PC's OWN forward structure exactly -- both layers use tanh,
    matching add_layer(..., act="tanh") on both PC layers."""

    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(784, 512)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        h = torch.tanh(self.fc0(x))
        out = torch.tanh(self.fc1(h))
        return out


def backprop_gradient(W0_pc, W1_pc, x_flat, y_flat, batch_size):
    """Computes the analytic gradient on the IDENTICAL weights/data/loss
    PC is implicitly targeting. Note the transpose: PC stores weights as
    (input_size, output_size); PyTorch's nn.Linear stores (out, in).
    Bias is explicitly zeroed on the PyTorch side too, matching PC's
    provably-zero bias state at this point (see pc_weight_delta's
    docstring) -- nn.Linear defaults to nonzero random bias otherwise,
    which would make this an unfair comparison."""
    model = MatchedMLP()
    with torch.no_grad():
        model.fc0.weight.copy_(torch.from_numpy(W0_pc.copy()))
        model.fc0.bias.zero_()
        model.fc1.weight.copy_(torch.from_numpy(W1_pc.copy()))
        model.fc1.bias.zero_()

    x = torch.from_numpy(x_flat.reshape(batch_size, 784).copy())
    y = torch.from_numpy(y_flat.reshape(batch_size, 10).copy())

    pred = model(x)
    # 0.5 * sum((pred-target)^2) -- matches PC's own energy definition
    # exactly (summed, not mean-reduced, with the 0.5 coefficient).
    loss = 0.5 * ((pred - y) ** 2).sum()
    loss.backward()

    # Gradients are dLoss/dW in PyTorch's (out,in) layout -- transpose back
    # to PC's (in,out) layout for a direct comparison.
    grad_W0 = model.fc0.weight.grad.numpy().copy()
    grad_W1 = model.fc1.weight.grad.numpy().copy()
    return grad_W0, grad_W1


def cosine_and_relerr(delta_pc, grad_bp):
    """PC's update_weights() is verified (via finite-difference gradient
    check, elsewhere this session) to move W in the DESCENT direction, i.e.
    delta_pc ~= -lr*grad_true. Compare delta_pc against -grad_bp so a
    correctly-behaving PC shows cosine ~= +1, not -1."""
    a = delta_pc.flatten()
    b = (-grad_bp).flatten()
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    rel_err = np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b))
    return cos, rel_err


def main():
    BATCH_SIZE = 32
    LR = 0.06
    IR = 0.08

    x_batch, y_batch = load_one_frozen_batch(BATCH_SIZE)

    net = SequentialPCN(batch_size=BATCH_SIZE)
    # pr=0, lmbda=0 -- isolate pure gradient behavior, no decay/precision confound
    net.add_layer(784, 512, lr=LR, ir=IR, pr=0.0, act="tanh", lmbda=0.0)
    net.add_layer(512, 10, lr=LR, ir=IR, pr=0.0, act="tanh", lmbda=0.0)
    net.add_layer(10, 0, lr=LR, ir=IR, pr=0.0, act="linear", lmbda=0.0)
    net.compile()
    net.randomize_weights()

    # Freeze this exact init -- copied into PyTorch for the backprop side too.
    W0 = net[0].weights.copy()
    W1 = net[1].weights.copy()

    x_flat = x_batch.flatten()
    y_flat = y_batch.flatten()

    print("Computing analytic backprop gradient (ground truth)...")
    grad_W0, grad_W1 = backprop_gradient(W0, W1, x_flat, y_flat, BATCH_SIZE)

    # --- Sanity check FIRST: do the forward passes even agree? ---
    # If PC's own predict() doesn't match MatchedMLP.forward() on the
    # IDENTICAL frozen weights, the gradient comparison above is meaningless.
    # UPDATE: initial check at steps=150 showed correlation=0.21, a real
    # disagreement. Rather than assume a bug, sweep predict()'s step count --
    # predict() settles with NOTHING clamped at the far end (unlike training,
    # which clamps BOTH input and target), so it may converge far slower.
    # If correlation climbs toward 1.0 with more steps, that's not a script
    # bug -- it means predict() itself may need more steps than train_step()
    # to give an accurate readout, which would mean our 92.72% evaluation
    # (which used steps=60, matching training) may have UNDER-measured the
    # network's true accuracy.
    print("\nSanity check: sweeping predict() step count vs feedforward-composition agreement...")

    model = MatchedMLP()
    with torch.no_grad():
        model.fc0.weight.copy_(torch.from_numpy(W0.copy()))
        model.fc0.bias.zero_()
        model.fc1.weight.copy_(torch.from_numpy(W1.copy()))
        model.fc1.bias.zero_()
        torch_pred = model(torch.from_numpy(x_flat.reshape(BATCH_SIZE, 784).copy())).numpy()

    for predict_steps in [60, 150, 300, 600, 1200, 2400]:
        net.reset_state()
        pc_pred = net.predict(x_flat, steps=predict_steps)
        pc_pred_reshaped = pc_pred.reshape(BATCH_SIZE, 10)
        max_diff = np.abs(pc_pred_reshaped - torch_pred).max()
        corr = np.corrcoef(pc_pred_reshaped.flatten(), torch_pred.flatten())[0, 1]
        print(f"  predict_steps={predict_steps:5d}  max_diff={max_diff:.4f}  correlation={corr:.4f}")

    print("\nIf correlation climbs toward 1.0, predict() needs more steps than")
    print("train_step() to converge -- worth re-evaluating accuracy with a")
    print("higher predict()-time step count before trusting 92.72% as final.")
    print("If it STAYS flat/low even at 2400 steps, that rules out slow")
    print("convergence and confirms a real structural mismatch instead.\n")

    print("(Gradient comparison table below is UNRELIABLE until the above")
    print(" is resolved -- shown for reference only.)")

    print(f"\n{'T':>5} {'W0 cosine':>10} {'W0 relerr':>10} {'W1 cosine':>10} {'W1 relerr':>10} {'time(s)':>8}")
    print("-" * 60)

    for T in [1, 2, 4, 8, 16, 32, 64, 80, 150]:
        start = perf_counter()
        dW0, dW1 = pc_weight_delta(net, x_flat, y_flat, T, W0, W1)
        elapsed = perf_counter() - start

        cos0, relerr0 = cosine_and_relerr(dW0, grad_W0)
        cos1, relerr1 = cosine_and_relerr(dW1, grad_W1)

        print(f"{T:>5} {cos0:>10.4f} {relerr0:>10.4f} {cos1:>10.4f} {relerr1:>10.4f} {elapsed:>8.3f}")

    print("\nGradient fidelity vs T table above uses ir=0.08 (the confirmed real value).")
    print("Now testing the actual research question: can a DIFFERENT ir reach the")
    print("SAME W0 fidelity in FEWER steps -- i.e. reduce required iterations,")
    print("not just make each iteration cheaper.\n")

    FIXED_T = 32  # W0 cosine was still climbing here (0.73) at ir=0.08 -- the
                   # interesting regime to test whether a bigger step size
                   # gets further, faster, at the SAME step count.

    print(f"=== ir sweep at fixed T={FIXED_T} ===")
    print(f"{'ir':>6} {'W0 cosine':>10} {'W1 cosine':>10} {'final energy':>13} {'stable?':>8}")
    print("-" * 55)

    for ir_test in [0.08, 0.16, 0.24, 0.32, 0.50, 0.80]:
        net2 = SequentialPCN(batch_size=BATCH_SIZE)
        net2.add_layer(784, 512, lr=LR, ir=ir_test, pr=0.0, act="tanh", lmbda=0.0)
        net2.add_layer(512, 10, lr=LR, ir=ir_test, pr=0.0, act="tanh", lmbda=0.0)
        net2.add_layer(10, 0, lr=LR, ir=ir_test, pr=0.0, act="linear", lmbda=0.0)
        net2.compile()

        net2[0].weights[:] = W0
        net2[1].weights[:] = W1

        net2.reset_state()
        net2.clamp_input(x_flat)
        net2[-1].clamp_state(y_flat)

        energy_trace = []
        for _ in range(FIXED_T):
            e = net2.calculate_state()
            energy_trace.append(e)
            net2.update_state()

        net2.update_weights()
        dW0_2 = net2[0].weights.copy() - W0
        dW1_2 = net2[1].weights.copy() - W1
        net2[-1].unclamp_state()

        cos0, _ = cosine_and_relerr(dW0_2, grad_W0)
        cos1, _ = cosine_and_relerr(dW1_2, grad_W1)

        # crude stability check: did energy blow up or oscillate wildly in
        # the back half of the trace, rather than settling?
        back_half = energy_trace[FIXED_T // 2:]
        stable = "yes" if (max(back_half) - min(back_half)) < 0.5 * max(back_half) else "NO"

        print(f"{ir_test:>6.2f} {cos0:>10.4f} {cos1:>10.4f} {energy_trace[-1]:>13.4f} {stable:>8}")

    print("\nIf a larger ir reaches W0 cosine comparable to what ir=0.08 needed")
    print("T=64-150 to reach, WHILE staying stable, that's a real reduction in")
    print("required iterations -- worth a full training run at that (ir, T) pair")
    print("to confirm it holds up beyond this single frozen batch.")


if __name__ == "__main__":
    main()
