"""
Backprop control experiment for imagenet.py's FullConvPCNetwork architecture.

Reuses imagenet.py's exact data pipeline, architecture, and muPC-scaled
initialization (build_network()) unchanged, with the one combination of
settings that makes ePC's single settling step mathematically equivalent to
plain backprop on this architecture:

    inference_steps = 1
    ir               = 1.0   (every layer -- forces e := -adjoint exactly,
                               not a blend toward it; see EPCStep()/
                               ComputeAdjoint() in FullConvPCLayer.cpp)
    fl               = 0.0   (disables DirectFeedbackUpdate()'s contribution,
                               which otherwise writes fl/batchSize * (DFA
                               projection) directly into W on top of whatever
                               UpdateWeights() does -- see that function's
                               own matmul call in FullConvPCLayer.cpp)
    useEPC           = True  (forced below, regardless of USE_EPC env var)

With those four things true, one TrainStep() call runs ProjectForward()
(the ordinary forward pass), one EPCStep() (which computes each hidden
layer's error as exactly minus the backprop adjoint at the current forward
states, since ir=1.0), then one UpdateWeights() using that error -- i.e.
the resulting weight update IS the backprop gradient on this exact
architecture, initialization, classifier head, and data, with none of the
PC-specific settling dynamics (multi-step relaxation, DKP feedback
alignment) in the loop at all.

This isolates one question: can the architecture AS CURRENTLY BUILT --
including whatever isn't yet fixed in muPC scaling's fan-in calculation or
the classifier head -- reach reasonable Tiny ImageNet accuracy via a method
with zero PC-specific failure modes to blame. Two outcomes:

  - Reaches roughly 35-45% (PCX's own published VGG-7 backprop baseline on
    Tiny ImageNet: https://arxiv.org/html/2407.01163v2) -> the architecture
    and setup are fine. Every failure seen in imagenet.py's real (ePC-
    settling, multi-step) runs is specifically about PC mechanics: ir's
    CUDA-graph-capture staleness (fixed in FullConvPCNetwork.cpp as of this
    commit), the muPC scaling fan-in miscalibration (confirmed: middle
    layers use channel count as fan-in, not channels*kernel_h*kernel_w,
    overscaling by ~3x for 3x3 convs -- not yet fixed), Predict()'s
    unclamped-terminal bias at prediction time (not yet fixed), or
    settling-step instability at step counts > 1 (not yet investigated).
  - Also fails to learn (stuck near the 0.5% random-guess floor) or blows
    up -> the problem is upstream of any PC-specific mechanism, most likely
    the muPC scaling bug (it affects the forward pass regardless of how
    weights get updated) or the classifier head (tanh squashes logits to
    [-1,1] before 200-way cross-entropy, capping achievable confidence far
    below what 200 classes need -- fine for 10-way MNIST, not for this).

Usage (same positional style as imagenet.py, minus INFERENCE_STEPS/IR/FL,
which this control fixes by definition):

    python imagenet_bp_control.py EPOCHS LR BATCH_SIZE SEED LMBDA
"""
import numpy as np
import os
import sys
from time import perf_counter

from imagenet import (
    DATA_DIR,
    load_wnids,
    load_train_set_cached,
    load_val_set_cached,
    compute_normalization_stats,
    to_float_batch,
    to_one_hot,
    pick_diagnostic_indices,
    run_collapse_diagnostic,
    build_network,
)

INFERENCE_STEPS = 1  # fixed by the control's own definition, see module docstring


def main() -> None:
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
    BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    LMBDA = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' not found in the current directory -- "
            f"run this script from wherever you unzipped tiny-imagenet-200.zip."
        )

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    mean, std = compute_normalization_stats(X_train_u8, seed=SEED)
    print(f"Per-channel normalization (computed once, from a 2,000-image training "
          f"sample): mean={mean}, std={std}")

    print(f"\n*** TINY IMAGENET, BP CONTROL (ePC, inference_steps=1, ir=1.0, fl=0.0) ***")
    print(f"If this doesn't reach roughly 35-45% (PCX's published backprop baseline "
          f"for a similar depth on Tiny ImageNet), the architecture/setup is the "
          f"problem, not PC-specific settling mechanics. If it does, PC mechanics are "
          f"the remaining suspect.")
    print(f"Training: {EPOCHS} epochs, inference_steps={INFERENCE_STEPS} (fixed), "
          f"ir=1.0 (fixed), fl=0.0 (fixed), lr={LR}, lmbda={LMBDA}, "
          f"batch_size={BATCH_SIZE}, seed={SEED}\n")

    net = build_network(BATCH_SIZE, N_CLASSES, LR, ir=1.0, fl=0.0, lmbda=LMBDA,
                        device="gpu", use_momentum=False, use_ipc=False)
    # Forced regardless of the USE_EPC env var -- ePC being on is what makes
    # this a backprop-equivalence experiment at all, see module docstring.
    net.set_use_epc(True)
    net.randomize_weights(SEED)

    diag_indices = pick_diagnostic_indices(y_val_idx)
    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, mean, std,
                            BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "PRE-training")

    rng = np.random.default_rng(SEED)
    n_train = len(X_train_u8)
    n_batches = n_train // BATCH_SIZE
    start_time = perf_counter()

    # No ir/fl decay loop here, unlike imagenet.py's main(): ir is pinned to
    # 1.0 and fl to 0.0 for the entire run by the control's own definition
    # (changing either would break the backprop-equivalence property this
    # script exists to test). lr still decays -- that's an ordinary
    # gradient-descent schedule choice, orthogonal to the PC-vs-BP question.
    DECAY_RATE = 0.94

    for epoch in range(EPOCHS):
        current_lr = LR * (DECAY_RATE ** epoch)
        net.set_learning_rate(current_lr)

        indices = rng.permutation(n_train)
        epoch_energy = 0.0

        for b in range(n_batches):
            batch_idx = indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            X_batch = to_float_batch(X_train_u8[batch_idx], mean, std)
            Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES)

            energy = net.train_step(X_batch, Y_batch, INFERENCE_STEPS)
            if not np.isfinite(energy):
                print(f"Epoch {epoch+1}, batch {b}/{n_batches}: NON-FINITE ENERGY -- STOPPING "
                      f"(previous batch's energy was {epoch_energy / b if b > 0 else 'N/A (first batch of epoch)'})")
                return
            epoch_energy += energy
            if b % 50 == 0:
                print(f"  epoch {epoch+1}, batch {b}/{n_batches}: energy={energy:.4f}")

        avg_energy = epoch_energy / n_batches

        N_ACC_BATCHES = 20
        correct, total = 0, 0
        for b in range(min(N_ACC_BATCHES, len(X_val_u8) // BATCH_SIZE)):
            X_batch = to_float_batch(X_val_u8[b * BATCH_SIZE:(b + 1) * BATCH_SIZE], mean, std)
            y_batch = y_val_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            preds = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, N_CLASSES)
            pred_classes = np.argmax(preds, axis=1)
            correct += np.sum(pred_classes == y_batch)
            total += BATCH_SIZE
        epoch_acc = 100.0 * correct / total

        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | "
              f"Avg energy: {avg_energy:.4f} | lr={current_lr:.2e}")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    correct, total = 0, 0
    for i in range(0, len(X_val_u8), BATCH_SIZE):
        X_batch_u8 = X_val_u8[i:i + BATCH_SIZE]
        y_batch = y_val_idx[i:i + BATCH_SIZE]
        if len(X_batch_u8) != BATCH_SIZE:
            continue
        X_batch = to_float_batch(X_batch_u8, mean, std)
        preds = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, N_CLASSES)
        pred_classes = np.argmax(preds, axis=1)
        correct += np.sum(pred_classes == y_batch)
        total += BATCH_SIZE

    val_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"BP control (FullConvPCNetwork architecture, exact backprop gradient) "
          f"Tiny ImageNet validation accuracy: {val_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"Reference: PCX's published VGG-7 backprop baseline on Tiny ImageNet is "
          f"~46% (vs. ~41% for PC). If this run landed well below that range, the "
          f"architecture/head/scaling -- not PC settling mechanics -- is the next "
          f"thing to fix.")

    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, mean, std,
                            BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "POST-training")


if __name__ == "__main__":
    main()
