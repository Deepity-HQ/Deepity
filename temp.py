"""
Minimal, direct test for the new, declarative SimplePCN wrapper API
(layer.py's Linear/Activation classes + SimplePCN.py). Exercises the
full real lifecycle: construction, configure(), randomize_weights()
(the method just fixed -- was calling the bound C++ function with an
argument it doesn't accept), a few training steps on synthetic data,
and a prediction call.

Not a formal correctness check (no known-right answer to compare
against) -- this just confirms the wrapper actually runs end-to-end
without crashing, and that shapes/types look sane.
"""
import numpy as np
from pydeepity import SimplePCN
from pydeepity.layer import Linear, ReLU

BATCH_SIZE = 4
IN_DIM = 4
HIDDEN_DIM = 8
OUT_DIM = 2

print("Building network via declarative API...")
net = SimplePCN(
    Linear(IN_DIM, HIDDEN_DIM),
    ReLU(),
    Linear(HIDDEN_DIM, OUT_DIM),
    batch_size=BATCH_SIZE,
    device="cpu",
)
print(f"  Architecture: {len(net.architecture)} components")

print("\nConfiguring (builds backend layers, sets optimizer, compiles)...")
net.configure(learning_rate=0.01, inference_rate=0.1, lmbda=0.0001, optimizer="ADAM")
print(f"  Backend layer count: {len(net)}")

print("\nExplicitly calling randomize_weights() -- the method that was just fixed...")
net.randomize_weights()
print("  OK, no exception raised.")

rng = np.random.default_rng(42)
X = rng.standard_normal((BATCH_SIZE, IN_DIM)).astype(np.float32)
Y = rng.standard_normal((BATCH_SIZE, OUT_DIM)).astype(np.float32)

print("\nRunning a few training steps on synthetic data...")
for step in range(5):
    energy = net.train_step(X, Y, steps=10)
    print(f"  step {step}: energy={energy:.4f}")

print("\nRunning predict()...")
pred = net.predict(X, steps=10)
print(f"  predict() output shape: {pred.shape} (expected: ({BATCH_SIZE}, {OUT_DIM}) or flattened)")
print(f"  predict() output dtype: {pred.dtype}")
print(f"  sample values: {pred.flatten()[:4]}")

print("\nChecking terminal layer's next_size, given _build_backend() adds every")
print("Linear layer uniformly (including the last one, with its own out_n as")
print("next_size) rather than a separate, explicit next_size=0 terminal layer:")
terminal = net[-1]
print(f"  terminal layer input_size={terminal.input_size}, output_size={terminal.output_size}")
print("  (if output_size > 0 here, the 'terminal' layer still computes an unused")
print("   outgoing mu prediction -- likely harmless given UpdateWeights() guards")
print("   on layerAbove being null regardless, but worth confirming energy/predict")
print("   values above still look sane, not NaN/exploding)")

print("\n" + "=" * 50)
print("PASS: ran end-to-end without exceptions.")
print("=" * 50)