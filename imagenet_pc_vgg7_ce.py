"""
Genuine multi-step predictive coding on the exact VGG-7 architecture
imagenet_bp_control_vgg7_ce.py validated as a backprop control (38.78%
best top-1, full 50 epochs).

Earlier attempts in this history (see git log) ported PCX's own
VGG_PCN_CE_tinyimagenet.yaml numbers directly (T=12, ir=0.0223,
momentum=0.55, AdamW lr=7.96e-5) under classic settling -- every one
collapsed to a near-constant, input-independent output (~0.5% val top1)
or exploded outright, across the full ir range from 0.0223 to 0.5.
Direct per-layer instrumentation traced this to real signal starvation:
the early conv layers received essentially zero error (printing as
exactly 0.0 to 6 decimal places), while Adam's per-parameter
normalization still gave them full-size, systematically-biased steps
anyway -- a real, measured mechanism, not a guess.

This version switches to muPC's OWN parameterization and recipe
(Innocenti et al., "muPC: Scaling Predictive Coding to 100+ Layer
Networks", https://arxiv.org/abs/2505.13124) instead of PCX's, since
that paper is specifically about fixing signal imbalance across depth
in PC networks:
  - useMuPCScaling=True: each layer's forward multiplier `a` (Table 1:
    a_1=N_0^-0.5, a_hidden=(N*L)^-0.5 or N^-0.5 depending on residual
    use, a_L=1/N) and unit-variance init, replacing Kaiming.
  - No momentum on state settling (the paper's own experiments use
    plain gradient descent for inference, "no other optimisation
    techniques such as momentum, weight decay, and nudging").
  - T = number of hidden layers (the paper's own stated rule), not
    PCX's T=12.
  - ir and the weight learning rate both moved into the paper's OWN
    searched range (activity lr in {1e-2...1e3}, weight lr in
    {1e-2...5e-1}) -- both 100-1000x larger than PCX's numbers, which
    were tuned for a completely different (non-muPC) setup and were
    never going to transfer as-is.
  - Plain Adam, no weight decay, matching the paper's own optimizer
    choice (not AdamW).

A real bug was found and fixed getting here: FullConvPCLayer's muPC
chain-rule derivative was missing the `a` factor entirely (mu =
Activation(a*preact+b), so d(mu)/d(preact) needs a*Activation'(.), not
just Activation'(.)) in EnsureMuHoldsDerivative(), UpdateState()'s fused
path, AND ComputeAdjoint(). A SEPARATE bug this exposed: UpdateWeights()
was ALSO separately multiplying by `a` in lr_batch/grad_scale, double-
counting the same factor once the chain rule was fixed. Both fixed and
verified via finite differences across multiple seeds, T=1 and T=4,
with and without pooling. Both are no-ops when useMuPCScaling is off,
so every earlier (non-muPC) verified configuration is unaffected.

CPU toy-network testing, first on a small 5-weight-layer/2-3-channel
net, then on a 6-weight-layer+terminal net matching VGG-7's actual
DEPTH (channels still much smaller: 1-16, not 128-512): the earlier,
shallower toy found ir=10-100 avoided the permanent divergence ir=1000
showed, but that network was a worse match for VGG-7's actual depth.
Repeating the ir sweep on the depth-matched network gave a different,
more specific answer: ir=10 leaves energy stuck at billions with wildly
imbalanced per-layer signal (some layers in the thousands, others
starved toward zero) -- "not infinite" but not healthy either. ir=0.5
and ir=1.0 keep energy small and bounded (~4, not billions) WITH
genuinely growing (not collapsing, not exploding) hidden-layer signal
over 30 batches; ir=2 decays back toward starvation; ir=5 jumps back to
energy ~33,000. ir=1.0 (also one of the paper's own grid points) is
this script's default. Channel counts are still 8-30x smaller than
VGG-7's real ones -- this run is the actual test of whether that gap
matters.

Usage:
    python imagenet_pc_vgg7_ce.py [EPOCHS] [BATCH_SIZE] [W_LR] [IR]
(defaults below are the muPC-paper-scale values, not PCX's)
"""
import sys
import numpy as np
from time import perf_counter
from pydeepity import dy  # pyright: ignore[reportAttributeAccessIssue]

from imagenet import DATA_DIR, IMG_SIZE, N_CHANNELS, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices

CROP_SIZE = 56
T = 7            # VGG-7's weight-bearing layer count, matching muPC paper's "T = hidden layers"
IR = 1.0         # one of the paper's own grid points; a 7-layer-deep CPU toy sweep found
                 # ir=0.5-1.0 keeps energy small/bounded with genuinely GROWING hidden-layer
                 # signal, while ir=2 decays and ir=5-10 blow up into the thousands/billions
MOMENTUM = 0.0   # muPC paper's own recipe: no momentum on state settling
W_LR = 0.02      # mid-range of muPC paper's own weight-lr search {1e-2...5e-1}
W_WD = 0.0       # muPC paper's own recipe: no weight decay
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_and_normalize(X_u8_batch, train: bool, rng):
    """Matches imagenet_bp_control_vgg7_ce.py's own function exactly."""
    n = len(X_u8_batch)
    X = X_u8_batch.reshape(n, N_CHANNELS, IMG_SIZE, IMG_SIZE)
    margin = IMG_SIZE - CROP_SIZE
    out = np.empty((n, N_CHANNELS, CROP_SIZE, CROP_SIZE), dtype=np.float32)

    for i in range(n):
        if train:
            oh = rng.integers(0, margin + 1)
            ow = rng.integers(0, margin + 1)
            crop = X[i, :, oh:oh + CROP_SIZE, ow:ow + CROP_SIZE]
            if rng.random() < 0.5:
                crop = crop[:, :, ::-1]
        else:
            oh = ow = margin // 2
            crop = X[i, :, oh:oh + CROP_SIZE, ow:ow + CROP_SIZE]
        out[i] = crop

    out /= 255.0
    out -= MEAN[None, :, None, None]
    out /= STD[None, :, None, None]
    return out.reshape(n, -1)


def run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, batch_size, n_classes, label):
    """Same diagnostic imagenet_bp_control_vgg7_ce.py uses."""
    rng = np.random.default_rng(0)
    diag_X = crop_and_normalize(X_val_u8[diag_indices], train=False, rng=rng)
    diag_true = y_val_idx[diag_indices]

    pad_n = batch_size - len(diag_X)
    if pad_n > 0:
        filler = crop_and_normalize(X_val_u8[len(diag_indices):len(diag_indices) + pad_n],
                                    train=False, rng=rng)
        diag_X_padded = np.vstack([diag_X, filler])
    else:
        diag_X_padded = diag_X[:batch_size]

    # 0 settling steps: cross-entropy's predict() always overrides the
    # terminal's belief with the layer-below's mu after the settle loop
    # regardless of step count, so this is the plain feedforward readout
    # of the CURRENT weights -- the natural way to read out a trained (or
    # training) network's prediction, independent of how many settling
    # steps training itself uses. Would silently break under SE loss.
    preds = net.predict(diag_X_padded, 0).reshape(batch_size, n_classes)
    n_real = len(diag_indices)
    pred_classes = np.argmax(preds[:n_real], axis=1)
    n_distinct = len(set(pred_classes.tolist()))

    # Distinct argmax count is NOT a reliable collapse signal on its own:
    # near-constant logits still argmax to different classes depending on
    # which position has the largest (tiny) noise, which can look like
    # "5/5 distinct" on an actually-collapsed network. The real signal is
    # whether the logits THEMSELVES vary across different inputs: for
    # each of the n_classes logit positions, std across these n_real
    # images -- near zero means the output doesn't depend on the input at
    # all, regardless of what argmax picks.
    cross_input_std = preds[:n_real].std(axis=0).mean()

    print(f"\n=== DIAGNOSTIC ({label}): are predictions input-dependent? ===")
    print(f"True classes:      {diag_true}")
    print(f"Predicted classes: {pred_classes}")
    print(f"Distinct predicted classes: {n_distinct} out of {n_real} inputs (weak signal, see below)")
    print(f"Mean per-logit std ACROSS these {n_real} inputs: {cross_input_std:.6f} "
          f"(near 0 = output doesn't depend on input, regardless of argmax)")
    if cross_input_std < 1e-3:
        print(">>> COLLAPSED: output is essentially input-independent.")
    else:
        print(">>> Output genuinely varies with input.")


def quick_val_accuracy(net, X_val_u8, y_val_idx, batch_size, n_classes, n_batches=10):
    """Real top-1 accuracy over n_batches held-out batches -- the robust
    signal Opus asked for in place of energy or distinct-argmax-count for
    judging whether a config is actually training, not just not-crashing."""
    rng = np.random.default_rng(0)
    correct, total = 0, 0
    for b in range(n_batches):
        X_batch = crop_and_normalize(X_val_u8[b * batch_size:(b + 1) * batch_size],
                                     train=False, rng=rng)
        y_batch = y_val_idx[b * batch_size:(b + 1) * batch_size]
        preds = net.predict(X_batch, 0).reshape(batch_size, n_classes)
        correct += np.sum(np.argmax(preds, axis=1) == y_batch)
        total += batch_size
    return 100.0 * correct / total


def build_vgg7_network(batch_size, n_classes, lr):
    net = dy.FullConvPCNetwork(batch_size=batch_size, device="gpu")

    def conv(in_c, out_c, hw, pad, pool=1, activation="relu", activation_deriv="drelu"):
        net.add_layer(in_c, out_c, hw, hw, 3, 3, stride_h=1, stride_w=1, pad_h=pad, pad_w=pad,
                     terminal_size=n_classes, lr=lr, ir=IR, fl=0.0, lmbda=W_WD,
                     activation=activation, activation_deriv=activation_deriv,
                     pool_h=pool, pool_w=pool, pool_stride_h=pool, pool_stride_w=pool)

    conv(N_CHANNELS, 128, CROP_SIZE, pad=1, pool=2)      # 56 -> 56 -> 28
    conv(128, 128, 28, pad=1)                            # 28 -> 28
    conv(128, 256, 28, pad=1, pool=2)                    # 28 -> 28 -> 14
    conv(256, 256, 14, pad=0)                             # 14 -> 12
    conv(256, 512, 12, pad=1, pool=2)                    # 12 -> 12 -> 6
    conv(512, 512, 6, pad=0)                              # 6 -> 4

    net.add_layer(512, n_classes, 4, 4, 4, 4, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                 terminal_size=n_classes, lr=lr, ir=IR, fl=0.0, lmbda=W_WD,
                 activation="linear", activation_deriv="dlinear")

    net.add_layer(n_classes, 0, 1, 1, 1, 1, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                 terminal_size=n_classes, lr=lr, ir=IR, fl=0.0, lmbda=W_WD,
                 activation="linear", activation_deriv="dlinear")

    net.set_use_residual_connections(False)
    # Must precede compile() (and randomize_weights(), called later in
    # main()): compile() is what actually computes and applies each
    # layer's `a` from the full architecture, and the init-variance
    # change only takes effect on the NEXT randomize_weights() call.
    net.set_use_mu_pc_scaling(True)
    net.compile()

    net.set_use_ipc(False)
    net.set_use_epc(False)  # classic PC settling, NOT the ePC adjoint trick
    net.set_use_momentum(MOMENTUM > 0.0, MOMENTUM)  # paper's recipe: off
    net.set_use_cross_entropy(True)
    net.set_optimizer("ADAM")       # paper's own optimizer, not AdamW
    net.set_psi_optimizer("ADAM")

    return net


def main():
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    LR = float(sys.argv[3]) if len(sys.argv) > 3 else W_LR
    # muPC-paper-scale ir (see module docstring) -- 100-1000x larger than
    # any PCX-derived value tried before. Pass a 4th CLI arg to override.
    ir = float(sys.argv[4]) if len(sys.argv) > 4 else IR
    peak_lr = 1.1 * LR
    end_lr = 0.1 * LR

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    print(f"\n*** TINY IMAGENET, VGG-7 GENUINE PC, muPC recipe (T={T}, ir={ir}, momentum={MOMENTUM}) ***")
    print(f"Comparable to: PCX's own PC-NN {39.49}+-{2.69}% (Table 1) and this codebase's own "
          f"BP-equivalent baseline, 38.78% (imagenet_bp_control_vgg7_ce.py, full 50 epochs).")
    print(f"Training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR:.6g} "
          f"(warmup-cosine to peak {peak_lr:.6g}, end {end_lr:.6g}), Adam (no weight decay), "
          f"cross-entropy, muPC scaling on, 56x56 crops, pooling on, {T} settling steps/batch, "
          f"ir={ir}.\n")

    net = build_vgg7_network(BATCH_SIZE, N_CLASSES, LR)
    net.set_inference_rate(ir)
    net.randomize_weights(7)

    diag_indices = pick_diagnostic_indices(y_val_idx)
    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, BATCH_SIZE, N_CLASSES,
                            "PRE-training")

    rng = np.random.default_rng(7)
    n_train = len(X_train_u8)
    n_batches = n_train // BATCH_SIZE
    total_steps = n_batches * EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_at_step(step):
        if step < warmup_steps:
            return LR + (peak_lr - LR) * (step / warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cosine

    global_step = 0
    best_acc = 0.0
    start_time = perf_counter()

    for epoch in range(EPOCHS):
        indices = rng.permutation(n_train)
        epoch_energy = 0.0
        n_energy_samples = 0

        for b in range(n_batches):
            batch_idx = indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            X_batch = crop_and_normalize(X_train_u8[batch_idx], train=True, rng=rng)
            Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES, eps=0.0)

            want_energy = (b % 50 == 0)
            net.set_learning_rate(lr_at_step(global_step))
            energy = net.train_step(X_batch, Y_batch, T, want_energy)
            if want_energy:
                if not np.isfinite(energy):
                    print(f"Epoch {epoch+1}, batch {b}/{n_batches}: NON-FINITE ENERGY -- STOPPING")
                    return
                epoch_energy += energy
                n_energy_samples += 1
                print(f"  epoch {epoch+1}, batch {b}/{n_batches}: energy={energy:.4f}")
            global_step += 1

        correct, total = 0, 0
        for b in range(len(X_val_u8) // BATCH_SIZE):
            X_batch = crop_and_normalize(X_val_u8[b * BATCH_SIZE:(b + 1) * BATCH_SIZE],
                                         train=False, rng=rng)
            y_batch = y_val_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            preds = net.predict(X_batch, 0).reshape(BATCH_SIZE, N_CLASSES)
            correct += np.sum(np.argmax(preds, axis=1) == y_batch)
            total += BATCH_SIZE

        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | "
              f"Avg energy: {epoch_energy/n_energy_samples:.4f} | Val top1: {acc:.2f}% | "
              f"Best top1: {best_acc:.2f}% | lr={lr_at_step(global_step):.2e}")

        if epoch == 0:
            run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, BATCH_SIZE, N_CLASSES,
                                    "POST-epoch-1")

    print(f"\n=== Result ===")
    print(f"Deepity VGG-7 genuine PC (T={T}, ir={IR}): best top-1 = {best_acc:.2f}%")
    print(f"Reference points: this codebase's own BP-equivalent baseline = 38.78%, "
          f"PCX's published PC-NN = 39.49+-2.69%.")


if __name__ == "__main__":
    main()
