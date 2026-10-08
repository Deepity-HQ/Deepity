"""
Exact VGG-7 replica of PCX's BP_CE backprop baseline on Tiny ImageNet --
the cross-entropy sibling of imagenet_bp_control_vgg7.py's SE-loss
version. Built after that SE version converged, cleanly and repeatably,
to a confirmed-not-a-bug degenerate solution: predicting the dataset's
marginal class frequency (1/200) for every input regardless of content.
Checked directly: 0.5*[(1/200-1)^2 + 199*(1/200-0)^2] summed over a
128-batch = 63.68, matching the observed energy plateau (~64-65) almost
exactly, every single run, fast collapse or slow convergence alike.
Squared error has no structural pressure against this -- unlike
cross-entropy, where softmax's normalization creates competitive,
relative pressure between classes. Every EARLIER working Deepity run
this session (imagenet_bp_control.py's v2/v3, 22-33%) used cross-entropy;
this was the first to try SE specifically, matching PCX's own separate
"BP_SE" vs "BP_CE" split.

Same ir=1.0/fl=0.0/inference_steps=1/ePC-forced trick every BP-control
script in this session uses: one TrainStep() call computes the literal
backprop gradient on this architecture, no PC-specific settling at all.

Architecture, pooling, 56x56 crops and ImageNet normalization are
unchanged from the SE version (and were never the problem -- see that
script's own history). Two deviations from PCX's own BP_CE recipe,
both deliberate:
  1. ReLU instead of PCX's own choice of GELU for this specific variant
     (confirmed via their yaml) -- this codebase's GELU-derivative-from-
     activated path is documented-but-unfixed-wrong (see
     imagenet_bp_control.py's own history), and ReLU is already proven
     to work fine under cross-entropy in this exact codebase.
  2. lr/wd: starting from PCX's own BP_CE search values (lr~2.975e-4,
     wd~5.813e-4) rather than the SE variant's (which were tuned for a
     different loss and activation entirely).

  Conv(3->128,   k3,s1,p1) -> ReLU -> MaxPool(2,2)   # 56->28
  Conv(128->128, k3,s1,p1) -> ReLU                   # 28
  Conv(128->256, k3,s1,p1) -> ReLU -> MaxPool(2,2)   # 28->14
  Conv(256->256, k3,s1,p0) -> ReLU                   # 14->12
  Conv(256->512, k3,s1,p1) -> ReLU -> MaxPool(2,2)   # 12->6
  Conv(512->512, k3,s1,p0) -> ReLU                   # 6->4
  Conv(512->200, k4,s1,p0)                           # 4->1, linear logits --
    the conv-layer equivalent of flatten+Linear(512*4*4,200): a
    full-spatial-extent kernel collapses the whole feature map to 1x1 in
    one GEMM, same trick imagenet_bp_control.py's own classifier uses.
    Cross-entropy is computed from these raw logits internally
    (set_use_cross_entropy(True) below) -- no softmax needed here.

Loss: cross-entropy against a hard one-hot (imagenet.py's own
to_one_hot() with eps=0.0 -- matching PCX's own ce_loss(), no label
smoothing). muPC scaling off (Kaiming-normal init instead, same
reasoning as imagenet_bp_control.py). AdamW with the exact warmup-cosine
schedule shape PCX uses, seeded from their BP_CE search values. 56x56
random-crop+flip (train) / center-crop (val), standard ImageNet mean/std
-- NOT the full 64x64 image, see imagenet_pytorch_baseline.py's own notes
on why.

Usage:
    python imagenet_bp_control_vgg7.py [EPOCHS] [BATCH_SIZE]
(defaults: 50, 128, matching PCX exactly)
"""
import sys
import numpy as np
from time import perf_counter
from pydeepity import dy  # pyright: ignore[reportAttributeAccessIssue] -- see imagenet.py's own import of this for why

from imagenet import DATA_DIR, IMG_SIZE, N_CHANNELS, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices

CROP_SIZE = 56
INFERENCE_STEPS = 1  # fixed by the control's own definition, see module docstring
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_and_normalize(X_u8_batch, train: bool, rng):
    """Matches imagenet_pytorch_baseline.py's TinyImageNetDataset exactly:
    random 56x56 crop + random horizontal flip for train, center crop for
    val, both normalized with plain ImageNet mean/std."""
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
    """Same diagnostic imagenet.py/imagenet_bp_control.py use, adapted to
    this script's own crop_and_normalize() instead of to_float_batch()."""
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

    # 0 settling steps, not INFERENCE_STEPS: cross-entropy's predict() always
    # overrides the terminal's belief with the layer-below's mu after the
    # settle loop regardless of how many steps ran, so an ePC adjoint pass
    # here changes nothing but burns the GPU time for it. This relies on
    # cross-entropy's override -- would silently break (always predict 0)
    # under SE loss, which has no such override.
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
                     terminal_size=n_classes, lr=lr, ir=1.0, fl=0.0, lmbda=5.813e-4,
                     activation=activation, activation_deriv=activation_deriv,
                     pool_h=pool, pool_w=pool, pool_stride_h=pool, pool_stride_w=pool)

    conv(N_CHANNELS, 128, CROP_SIZE, pad=1, pool=2)      # 56 -> 56 -> 28
    conv(128, 128, 28, pad=1)                            # 28 -> 28
    conv(128, 256, 28, pad=1, pool=2)                    # 28 -> 28 -> 14
    conv(256, 256, 14, pad=0)                             # 14 -> 12
    conv(256, 512, 12, pad=1, pool=2)                    # 12 -> 12 -> 6
    conv(512, 512, 6, pad=0)                              # 6 -> 4

    # Full-spatial-extent kernel (k=4 on a 4x4 input): the conv-layer
    # equivalent of flatten+Linear(512*4*4, n_classes), matching
    # imagenet_bp_control.py's own classifier-collapse trick. Linear, not
    # ReLU -- these are PCX's raw SE-loss logits.
    net.add_layer(512, n_classes, 4, 4, 4, 4, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                 terminal_size=n_classes, lr=lr, ir=1.0, fl=0.0, lmbda=5.813e-4,
                 activation="linear", activation_deriv="dlinear")

    net.add_layer(n_classes, 0, 1, 1, 1, 1, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                 terminal_size=n_classes, lr=lr, ir=1.0, fl=0.0, lmbda=5.813e-4,
                 activation="linear", activation_deriv="dlinear")

    net.set_use_residual_connections(False)
    net.compile()

    net.set_use_ipc(False)
    net.set_use_epc(True)  # forced -- this is what makes it a BP control at all
    net.set_use_momentum(False, 0.9)
    net.set_use_cross_entropy(True)  # cross-entropy, matching PCX's BP_CE
    net.set_optimizer("ADAMW")
    net.set_psi_optimizer("ADAMW")

    return net


def main():
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    LR = float(sys.argv[3]) if len(sys.argv) > 3 else 2.975e-4
    peak_lr = 1.1 * LR
    end_lr = 0.1 * LR

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    print(f"\n*** TINY IMAGENET, VGG-7 BP CONTROL (exact PCX architecture, pooling, cross-entropy) ***")
    print(f"Comparable to: PCX's own BP_CE baseline (SE variant converged to the 63.68 trivial minimum -- see this script's own docstring).")
    print(f"Deviation: ReLU instead of PCX's GELU for this variant (no working GELU derivative in this codebase).")
    print(f"Training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR:.6g} "
          f"(warmup-cosine to peak {peak_lr:.6g}, end {end_lr:.6g}), AdamW, cross-entropy, "
          f"56x56 crops, pooling on.\n")

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

            # Energy readout costs a cublasSdot + cudaStreamSynchronize PER
            # LAYER on GPU, purely to produce a number we only print every
            # 50 batches anyway -- skip it on batches we don't log. Tradeoff:
            # a blowup can go undetected for up to 49 extra batches instead
            # of being caught immediately.
            want_energy = (b % 50 == 0)
            net.set_learning_rate(lr_at_step(global_step))
            energy = net.train_step(X_batch, Y_batch, INFERENCE_STEPS, want_energy)
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
            preds = net.predict(X_batch, 0).reshape(BATCH_SIZE, N_CLASSES)  # see diagnostic's note above
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
    print(f"Deepity VGG-7 BP control, cross-entropy (exact PCX architecture): "
          f"best top-1 = {best_acc:.2f}%")
    print(f"63.68 would mean it fell into the same trivial minimum the SE variant did -- "
          f"anything meaningfully above the ~0.5% random floor means cross-entropy's "
          f"competitive pressure between classes avoided it.")


if __name__ == "__main__":
    main()
