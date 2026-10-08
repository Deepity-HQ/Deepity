"""
Genuine multi-step predictive coding on the exact VGG-7 architecture
imagenet_bp_control_vgg7_ce.py validated as a backprop control (38.78%
best top-1, full 50 epochs). That script forced inference_steps=1,
ir=1.0, ePC -- a literal backprop gradient computed through the PC
machinery, no real settling at all. This script drops that trick
entirely: useEPC=False (classic predictive coding, not Deepity's ePC
adjoint acceleration), inference_steps=12, ir=0.0223, momentum=0.55 on
the state updates -- taken directly from PCX's own
VGG_PCN_CE_tinyimagenet.yaml (T, optim.x.lr, optim.x.momentum), the
hyperparameters PCX actually used for standard, non-incremental PC on
this architecture. optim.w (lr=7.96e-5, wd=3.53e-5) comes from the same
yaml.

PCX's own PC-NN number (39.49+-2.69%, Table 1) is the headline PC result
to compare against, but it isn't an exact target: "NN" uses a soft
negative-nudge clamp (beta=-0.9: h = u + 0.9*(u-y), not a hard clamp) on
the terminal belief, a nudging/contrastive technique this codebase's
hard ClampState() doesn't implement. "CE" uses beta=1.0 (h=y exactly --
a genuine hard clamp, exactly what ClampState() already does), so CE's
hyperparameters are the faithful match for what Deepity can actually
run, even though 39.49% was reported for NN specifically, not CE.
Treat 39.49+-2.69% as the best available PC reference point, and
38.78% (this same architecture's own BP-equivalent result) as the more
directly comparable baseline, since it isolates "does real multi-step
PC settling match/beat backprop on THIS codebase" from "does Deepity's
PC implementation match PCX's own."

Verification before trusting this: every finite-difference gradient
check earlier this session used inferenceSteps=1 (the BP trick), which
never exercises genuine multi-step settling. Checking it directly
surfaced a real bug in FullConvPCNetwork::TrainStep(): classic Step()
computes e=z-mu THEN updates z, so after the last settling iteration e
is one step stale relative to the now-updated z, and UpdateWeights()
(called right after) was reading that stale e. Fixed with a resync
(recompute e from the final z, reconvert mu to derivative form) before
the weight update -- confirmed via finite differences on a 3-layer
pooled network (T=4, ir=0.3) after the fix, ratios 0.986-1.0024. ePC
itself needed no change: its own per-iteration CalculateState() call
already keeps it current by construction.

Architecture, pooling, cross-entropy, ReLU-instead-of-GELU deviation,
56x56 crops and ImageNet normalization: all identical to
imagenet_bp_control_vgg7_ce.py, see that script's own docstring.

Usage:
    python imagenet_pc_vgg7_ce.py [EPOCHS] [BATCH_SIZE]
(defaults: 50, 128, matching PCX exactly)
"""
import sys
import numpy as np
from time import perf_counter
from pydeepity import dy  # pyright: ignore[reportAttributeAccessIssue]

from imagenet import DATA_DIR, IMG_SIZE, N_CHANNELS, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices

CROP_SIZE = 56
T = 12          # PCX's VGG_PCN_CE_tinyimagenet.yaml hp.T
IR = 0.0223381085942421   # PCX's optim.x.lr
MOMENTUM = 0.55           # PCX's optim.x.momentum
W_LR = 7.961890843897934e-05   # PCX's optim.w.lr
W_WD = 3.5299055677719896e-05  # PCX's optim.w.wd
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

    print(f"\n=== DIAGNOSTIC ({label}): are predictions input-dependent? ===")
    print(f"True classes:      {diag_true}")
    print(f"Predicted classes: {pred_classes}")
    print(f"Distinct predicted classes: {n_distinct} out of {n_real} inputs")
    print(f"First image's output std: {preds[0].std():.6f}, range: [{preds[0].min():.4f}, {preds[0].max():.4f}]")
    if n_distinct == 1:
        print(">>> COLLAPSED: same class predicted regardless of input.")
    else:
        print(">>> Predictions DO vary across inputs.")


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
    net.compile()

    net.set_use_ipc(False)
    net.set_use_epc(False)  # classic PC settling, NOT the ePC adjoint trick
    net.set_use_momentum(True, MOMENTUM)
    net.set_use_cross_entropy(True)
    net.set_optimizer("ADAMW")
    net.set_psi_optimizer("ADAMW")

    return net


def main():
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    LR = float(sys.argv[3]) if len(sys.argv) > 3 else W_LR
    peak_lr = 1.1 * LR
    end_lr = 0.1 * LR

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    print(f"\n*** TINY IMAGENET, VGG-7 GENUINE PC (T={T}, ir={IR}, momentum={MOMENTUM}) ***")
    print(f"Comparable to: PCX's own PC-NN {39.49}+-{2.69}% (Table 1) and this codebase's own "
          f"BP-equivalent baseline, 38.78% (imagenet_bp_control_vgg7_ce.py, full 50 epochs).")
    print(f"Training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR:.6g} "
          f"(warmup-cosine to peak {peak_lr:.6g}, end {end_lr:.6g}), AdamW, cross-entropy, "
          f"56x56 crops, pooling on, {T} settling steps/batch.\n")

    net = build_vgg7_network(BATCH_SIZE, N_CLASSES, LR)
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
