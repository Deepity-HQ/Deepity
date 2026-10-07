"""
Backprop control experiment for imagenet.py's FullConvPCNetwork architecture.

Uses the same data pipeline and overall layer shape as imagenet.py's
build_network(), but as a deliberately SIMPLE, standard baseline rather than
that function's full recipe -- the goal here is the most boring, best-tested
configuration possible, so a failure can't be blamed on anything exotic:

    ir               = 1.0   (every layer -- forces e := -adjoint exactly,
                               not a blend toward it; see EPCStep()/
                               ComputeAdjoint() in FullConvPCLayer.cpp)
    fl               = 0.0   (disables DirectFeedbackUpdate()'s contribution,
                               which otherwise writes fl/batchSize * (DFA
                               projection) directly into W on top of whatever
                               UpdateWeights() does -- see that function's
                               own matmul call in FullConvPCLayer.cpp)
    inference_steps  = 1
    useEPC           = True  (forced below, regardless of USE_EPC env var)

With those four things true, one TrainStep() call runs ProjectForward()
(the ordinary forward pass), one EPCStep() (which computes each hidden
layer's error as exactly minus the backprop adjoint at the current forward
states, since ir=1.0), then one UpdateWeights() using that error -- i.e. the
resulting weight update IS the backprop gradient on this architecture, with
none of the PC-specific settling dynamics (multi-step relaxation, DKP
feedback alignment) in the loop at all.

On top of that BP-equivalence baseline, this version makes five further
changes relative to imagenet.py's own build_network(), each one targeting a
specific, independently-confirmed problem found by inspecting this exact
architecture and the previous (failed, 0.50% accuracy, saturated-at-+1-logits)
run of this script:

  1. ReLU hidden layers, not GELU. Activations.h's own doc comment admits
     ActivationDerivativeFromActivatedScalar's dGELU case is wrong -- it
     evaluates the derivative formula AT THE ACTIVATED OUTPUT, not the
     pre-activation, because GELU's derivative can't be recovered from its
     output alone (unlike ReLU's, which only needs the output's sign).
     Every GELU-activated hidden layer's gradient has been wrong since GELU
     became the default. ReLU's dRELU formula has no such bug.
  2. Linear classifier logits, not tanh. tanh bounds logits to [-1,1];
     with 200 classes, max achievable softmax confidence from that range is
     a few percent, and cross-entropy's pressure to push the correct
     class's logit up (see point 3) drives EVERY logit toward tanh's +1
     saturation plateau, where its gradient is ~0. The previous run's
     post-training diagnostic showed exactly this: every output in
     [0.899, 1.0].
  3. Properly-summing one-hot targets (fixed in imagenet.py's to_one_hot()
     itself, shared by both scripts) -- eps used to land on top of the true
     class AND flat across every other class, summing to 1.198 instead of
     1.0. That uniform positive bias, applied every sample and step, is
     what Adam turned into the steady climb toward tanh saturation in
     point 2.
  4. muPC scaling OFF, plain Kaiming-normal init instead (RandomizeWeights()
     already falls back to limit=sqrt(2/fan_in) normal init with a=1.0
     when useMuPCInit is false -- no new C++ needed, just don't call
     set_use_mu_pc_scaling(True)). Sidesteps two separate, not-yet-fully-
     resolved issues at once: muPC's per-layer scale was computed from the
     WIDEST middle layer's fan-in and applied uniformly to every middle
     layer, under-scaling every layer that isn't the widest one (confirmed:
     64-channel layers end up with effective gain 0.5, 128-channel layers
     0.71, only the 256-channel layers get the intended gain of 1.0); and
     separately, muPC's per-layer `a` is baked into the GRADIENT's
     magnitude (grad_scale = -a/batchSize in UpdateWeights()), but Adam's
     m/sqrt(v) normalization is scale-invariant to that magnitude, washing
     out muPC's intended per-layer differentiation from the weight UPDATE
     step (though not from the forward pass) -- correctly preserving it
     under Adam needs per-layer learning rates, not just per-layer `a`,
     which is real, separate work this baseline sidesteps entirely.
  5. Adam (not ADAMW) at lr~1e-3, no muPC to interact with it -- removes
     lmbda from the picture too (plain Adam, unlike AdamW, has no weight-
     decay term at all, see UpdateWeights()'s optimizer switch).

This isolates one question: can the architecture's basic shape -- the conv/
spatial/channel structure itself, independent of every PC-specific choice
AND independent of the GELU/tanh/muPC/target-sum issues above -- reach
reasonable Tiny ImageNet accuracy via the most standard setup possible.
Two outcomes:

  - Reaches double digits within a few epochs -> the underlying architecture
    is fine, and the real work is re-introducing PC-specific settling
    (multi-step inference, ePC, DKP) and the removed techniques (muPC,
    GELU, tanh+cross-entropy) one at a time against this now-working
    baseline, to see which of them hold up and which need their own fix
    before going back in.
  - Still fails to learn -> something more fundamental is wrong (one
    concrete next step from here: a one-batch gradient comparison against
    PyTorch autograd on the same weights, since the existing C++
    gradient-verify tests may share this exact derivative-from-activated
    path and not catch it either).

Usage:

    python imagenet_bp_control.py EPOCHS LR BATCH_SIZE SEED
"""
import numpy as np
import os
import sys
from time import perf_counter
from pydeepity import dy  # pyright: ignore[reportAttributeAccessIssue] -- see imagenet.py's own import of this for why

from imagenet import (
    DATA_DIR,
    IMG_SIZE,
    N_CHANNELS,
    load_wnids,
    load_train_set_cached,
    load_val_set_cached,
    compute_normalization_stats,
    to_float_batch,
    to_one_hot,
    pick_diagnostic_indices,
    run_collapse_diagnostic,
)

INFERENCE_STEPS = 1  # fixed by the control's own definition, see module docstring


def augment_batch(X_uint8_batch, rng, pad=4):
    """Standard pad-then-random-crop + random horizontal flip (the CIFAR/
    ImageNet-style recipe: reflect-pad by `pad` on each side, crop back to
    the original size at a random offset). Training batches only -- Tiny
    ImageNet overfits heavily without this, and it's one of the standard
    backprop-world tricks this run exists to apply before comparing against
    a published baseline. Input/output are both (n, N_CHANNELS*IMG_SIZE*
    IMG_SIZE) flat CHW uint8."""
    n = len(X_uint8_batch)
    X = X_uint8_batch.reshape(n, N_CHANNELS, IMG_SIZE, IMG_SIZE)
    padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    out = np.empty_like(X)
    offsets_h = rng.integers(0, 2 * pad + 1, size=n)
    offsets_w = rng.integers(0, 2 * pad + 1, size=n)
    flips = rng.random(n) < 0.5

    for i in range(n):
        oh, ow = offsets_h[i], offsets_w[i]
        crop = padded[i, :, oh:oh + IMG_SIZE, ow:ow + IMG_SIZE]
        out[i] = crop[:, :, ::-1] if flips[i] else crop

    return out.reshape(n, -1)


def build_sane_baseline_network(batch_size, n_classes, lr, lmbda=1e-4):
    """Same conv/spatial/channel shape as imagenet.py's build_network(), but
    ReLU hidden layers, linear classifier logits, muPC off (plain Kaiming
    init instead) -- see module docstring for why each of these differs
    from build_network()'s own recipe. Optimizer is AdamW (decoupled weight
    decay, lmbda~1e-4 is a standard, gentle CNN-image-classification value,
    not the aggressive "counteract runaway growth" magnitude earlier
    sessions needed under muPC's unit-variance init -- this network's
    weights are now ordinary Kaiming-small, so ordinary AdamW decay
    conventions apply)."""
    net = dy.FullConvPCNetwork(batch_size=batch_size, device="gpu")

    def conv(in_c, out_c, hw, k, stride, pad, activation="relu", activation_deriv="drelu"):
        net.add_layer(in_c, out_c, hw, hw, k, k, stride_h=stride, stride_w=stride,
                     pad_h=pad, pad_w=pad, terminal_size=n_classes,
                     lr=lr, ir=1.0, fl=0.0, lmbda=lmbda,
                     activation=activation, activation_deriv=activation_deriv)

    conv(N_CHANNELS, 64, IMG_SIZE, k=5, stride=2, pad=2)
    conv(64, 64, 32, k=3, stride=1, pad=1)
    conv(64, 64, 32, k=3, stride=1, pad=1)
    conv(64, 128, 32, k=3, stride=2, pad=1)
    conv(128, 128, 16, k=3, stride=1, pad=1)
    conv(128, 128, 16, k=3, stride=1, pad=1)
    conv(128, 256, 16, k=3, stride=2, pad=1)
    conv(256, 256, 8, k=3, stride=1, pad=1)
    conv(256, 256, 8, k=3, stride=1, pad=1)
    # Linear, not tanh -- see module docstring point 2.
    conv(256, n_classes, 8, k=8, stride=1, pad=0, activation="linear",
         activation_deriv="dlinear")

    net.add_layer(n_classes, 0, 1, 1, 1, 1, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                 terminal_size=n_classes, lr=lr, ir=1.0, fl=0.0, lmbda=lmbda,
                 activation="linear", activation_deriv="dlinear")

    # muPC scaling deliberately left OFF -- see module docstring point 4.
    # RandomizeWeights() falls back to Kaiming-normal init (limit=
    # sqrt(2/fan_in), a=1.0) automatically when this is never turned on.
    net.set_use_residual_connections(False)
    net.compile()

    net.set_use_ipc(False)
    net.set_use_epc(True)  # forced -- this is what makes it a BP control at all
    net.set_use_momentum(False, 0.9)
    net.set_use_cross_entropy(True)
    net.set_optimizer("ADAMW")
    net.set_psi_optimizer("ADAMW")

    return net


def main() -> None:
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7

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

    print(f"\n*** TINY IMAGENET, BP CONTROL v3 (ReLU, linear head, no muPC, AdamW, "
          f"augmentation) ***")
    print(f"Same sane-baseline shape as v2 (which reached 22.76% in 5 epochs), plus "
          f"three standard backprop-world additions to see how close that gets to "
          f"PCX's ~41-46% published baseline before touching anything PC-specific: "
          f"AdamW (decoupled weight decay) instead of plain Adam, the same per-epoch "
          f"lr decay schedule imagenet.py's own recipe uses, and random-crop + "
          f"horizontal-flip augmentation (Tiny ImageNet overfits heavily without it).")
    print(f"Training: {EPOCHS} epochs, inference_steps={INFERENCE_STEPS} (fixed), "
          f"ir=1.0 (fixed), fl=0.0 (fixed), lr={LR} (decaying), optimizer=ADAMW, "
          f"batch_size={BATCH_SIZE}, seed={SEED}\n")

    net = build_sane_baseline_network(BATCH_SIZE, N_CLASSES, LR)
    net.randomize_weights(SEED)

    diag_indices = pick_diagnostic_indices(y_val_idx)
    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, mean, std,
                            BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "PRE-training")

    rng = np.random.default_rng(SEED)
    n_train = len(X_train_u8)
    n_batches = n_train // BATCH_SIZE
    start_time = perf_counter()

    # Same per-epoch exponential decay imagenet.py's own build_network()
    # recipe uses -- proven reasonable there, no reason to pick differently
    # here.
    DECAY_RATE = 0.94

    for epoch in range(EPOCHS):
        current_lr = LR * (DECAY_RATE ** epoch)
        net.set_learning_rate(current_lr)

        indices = rng.permutation(n_train)
        epoch_energy = 0.0

        for b in range(n_batches):
            batch_idx = indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            X_batch_u8 = augment_batch(X_train_u8[batch_idx], rng)
            X_batch = to_float_batch(X_batch_u8, mean, std)
            Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES)

            energy = net.train_step(X_batch, Y_batch, INFERENCE_STEPS)
            if not np.isfinite(energy):
                print(f"Epoch {epoch+1}, batch {b}/{n_batches}: NON-FINITE ENERGY -- STOPPING "
                      f"(previous batch's energy was {epoch_energy / b if b > 0 else 'N/A (first batch of epoch)'})")
                return
            epoch_energy += energy
            if b % 10 == 0:
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
    print(f"BP control v3 (ReLU, linear head, no muPC, AdamW, augmentation) Tiny "
          f"ImageNet validation accuracy: {val_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")
    print(f"Reference: PCX's published VGG-7 backprop baseline on Tiny ImageNet is "
          f"~46% (vs. ~41% for PC).")

    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, mean, std,
                            BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "POST-training")


if __name__ == "__main__":
    main()
