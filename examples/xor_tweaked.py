import deepity as pd
import numpy as np
import time

X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float32)
Y = np.array([[-1], [1], [1], [-1]], dtype=np.float32)

SETTLING_STEPS = 100

if __name__ == "__main__":
    # SequentialPCN wraps DiscriminativePCLayer instead of SimplePCLayer:
    # every layer carries its own learned precision p (one scalar per unit,
    # initialized to 1), which rescales that unit's prediction error before
    # it is used for either the energy or the feedback into the layer
    # below -- e_weighted = p * (z - mu). A unit whose error stays small
    # and consistent gets trusted more (p climbs); a unit whose error
    # stays large or noisy gets trusted less (p falls), so it stops
    # dominating the feedback the lower layer settles against.
    pcn = pd.SequentialPCN(
        pd.layer.Linear(2, 8),
        pd.layer.TanH(),
        pd.layer.Linear(8, 1),
        pd.layer.TanH(),
        batch_size=4,
    )

    pcn.configure(
        learning_rate=0.1,
        inference_rate=0.15,
        # precision_rate: how fast each unit's log-precision adapts
        # (log_p -= pr * 0.5*(p*err^2 - 1), hard-clamped to +-5 --
        # CPUBackend::UpdatePrecisionFromError). The output layer here
        # is a single unit whose error can stay largeish for a while
        # early in training (it hasn't learned XOR yet); too high a
        # precision_rate risks driving log_p into its -5 floor and
        # pinning it there (p ~ exp(-5) ~ 0.0067), which would collapse
        # the feedback this layer sends down to the hidden layer almost
        # to zero. 0.002 keeps that floor comfortably far away for a
        # run this length, well below the library's 0.01 default.
        precision_rate=0.002,
        # lmbda: L2 weight decay, applied as W *= (1 - lmbda) once per
        # epoch (DiscriminativePCLayer::UpdateWeights(), SGD branch) --
        # NOT scaled by epoch count, so it compounds: over 1500 epochs,
        # 1e-3 multiplies out to (1-1e-3)^1500 = 0.22 (weights dragged
        # to ~22% of their undecayed scale), while 1e-4 compounds to
        # only (1-1e-4)^1500 = 0.86 (~14% shrink). 1e-4 is mild enough
        # to matter as regularization without fighting the already
        # weak signal from a 4-row dataset.
        lmbda=1e-4,
        optimizer="SGD",
    )

    # Match xor.py's actual init: xor.py passes seed=42 into configure(),
    # which seeds a single normal-distribution randomize_weights() call
    # across every layer at once. SequentialPCN.configure() has no seed
    # parameter, so that call is made explicitly here instead -- this is
    # what configure() itself calls internally, just with a fixed seed.
    pcn.randomize_weights(seed=42, distribution="normal")

    start = time.perf_counter()
    for epoch in range(1500):
        # train_step() already calls update_precision() once per step
        # internally, right after update_weights() -- see
        # SequentialPCN.train_step() in deepity/SequentialPCN.py.
        pcn.train_step(X, Y, SETTLING_STEPS)
    print(f"Elapsed: {time.perf_counter() - start:.4f}s\n")

    preds = pcn.predict(X, SETTLING_STEPS)
    print("Predictions:")
    for i in range(4):
        print(f"Input: {X[i]}, Target: {Y[i][0]} -> Prediction: {preds[i][0]:.4f}")
