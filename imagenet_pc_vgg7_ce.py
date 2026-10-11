"""
Genuine multi-step predictive coding on the exact VGG-7 architecture
imagenet_bp_control_vgg7_ce.py validated as a backprop control (38.78%
best top-1, full 50 epochs).

History (see git log for the full detail of each): porting PCX's own
VGG_PCN_CE_tinyimagenet.yaml numbers directly under classic settling
collapsed or exploded across every ir tried (0.0223 to 0.5, both
momentum conventions). Per-layer instrumentation traced this to real
signal starvation with ReLU: early layers measured essentially zero
error while Adam still gave them full-size steps anyway, and the
affected layer's dead-ReLU fraction climbed to ~100% in lockstep with
the collapse. Switching to muPC's own parameterization (fixing a real
chain-rule bug along the way, see git log) showed the SAME layer stuck
at zero error regardless of ir, momentum, or T on a depth-matched toy
network -- never conclusively tested at VGG-7's real scale.

THIS version instead replicates PCX's actual released PC-CE code
(pcax v0.6.1) line for line, correcting a real, specific deviation:
this script was using ReLU (borrowed from the backprop control, which
needed it because of a documented GELU-derivative bug), but PCX's own
VGG_PCN_CE_tinyimagenet.yaml specifies act_fn: hard_tanh for PC-CE, not
ReLU, GELU, or leaky_relu (those are PCX's OTHER PC variants'
activations, not this one's). ReLU is unbounded above, so activations
can grow without limit -- exactly the shape of the 1e8-1e9 energy
blow-ups seen earlier. hard_tanh is bounded to [-1,1] by construction,
and PCX's own (small) init keeps most units in its linear region
initially, which a dead ReLU unit's permanently-zero gradient never
gets the chance to do. The per-layer data already in hand (L5 dead
fraction -> ~100%) is exactly the failure mode hard_tanh's boundedness
is meant to prevent.

Three changes from the previous (muPC) attempt, all matching PCX's
PC-CE exactly, no deviations:
  - HARD_TANH added to Deepity (one enum value, CPU+CUDA kernels,
    exact derivative -- no GELU-style from-activated ambiguity, since
    whether clamping happened is recoverable from the activated value
    alone). Verified via finite differences before use (see git log).
  - PCX's own init added (SetUsePCXInit): weights uniform in
    +-1/sqrt(fan_in), not Kaiming-normal (biases were already zero by
    default either way). Verified the resulting weights actually
    respect that bound before use.
  - muPC scaling OFF (PCX doesn't use it), momentum back ON (PCX's
    own trace-form momentum=0.55, matching optax's own convention --
    see the earlier momentum-fix commit), AdamW restored (PCX's own
    optimizer, not plain Adam), and Adam's epsilon back to its
    default (the eps mitigation was MY OWN addition from a different
    line of reasoning, not PCX's own recipe -- removed here so this is
    a clean test of the activation-function hypothesis specifically,
    not a mix of several changes at once).

Back to PCX's own exact numbers: T=12, ir=0.0223, momentum=0.55,
weight lr=7.96e-5, weight decay=3.53e-5.

How to read the outcome: if this trains, the activation-function
hypothesis was right, and the ~20 batches/sec classic-settling path is
confirmed correct. If it STILL collapses with PCX's literal code
replicated this closely, the remaining difference is a genuine Deepity
bug in the classic settling path, not a config/hyperparameter gap --
and the next real step is a one-batch, identical-weights comparison
against pcax itself to find exactly where the two diverge.

Usage:
    python imagenet_pc_vgg7_ce.py [EPOCHS] [BATCH_SIZE] [W_LR] [IR] [CHECKPOINT_PREFIX]
(defaults below are PCX's own PC-CE numbers; CHECKPOINT_PREFIX defaults
to "checkpoint" -- a "<prefix>.safetensors" + "<prefix>_meta.json" pair
written after every epoch. Rerunning the exact same command resumes
from it automatically if it exists: weights, Adam moments, epoch,
global_step, best_acc, and the data-shuffling RNG state all carry over,
so the warmup-cosine LR schedule and data order continue exactly where
they left off rather than restarting cold. At most one epoch of
progress is lost to a mid-epoch kill, since checkpoints are only
written at epoch boundaries.)
"""
import json
import os
import sys
import numpy as np
from time import perf_counter
from pydeepity import dy  # pyright: ignore[reportAttributeAccessIssue]

from imagenet import DATA_DIR, IMG_SIZE, N_CHANNELS, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices

CROP_SIZE = 56
# PCX's own VGG_PCN_CE_tinyimagenet.yaml numbers, verbatim.
T = 12
IR = 0.0223381085942421
MOMENTUM = 0.55           # trace-form (optax convention, see git log), not EMA
W_LR = 7.961890843897934e-05
W_WD = 3.5299055677719896e-05
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


def checkpoint_paths(prefix):
    return prefix + ".safetensors", prefix + "_meta.json"


def save_checkpoint(net, prefix, epoch, global_step, best_acc, elapsed, rng):
    """Weights + Adam moments go through net.save() (safetensors, see
    FullConvPCLayer::GetStateDict()); everything net.save() can't know
    about -- how far the training LOOP got, not the network itself --
    goes in a small JSON sidecar next to it. Written at epoch boundaries
    only (not mid-epoch): resuming replays at most one epoch, in
    exchange for not needing to serialize the in-epoch batch index or
    permutation as well."""
    weights_path, meta_path = checkpoint_paths(prefix)
    net.save(weights_path)
    meta = {
        "epoch": epoch,
        "global_step": global_step,
        "best_acc": best_acc,
        "elapsed": elapsed,
        "rng_state": rng.bit_generator.state,
    }
    # Write to a temp file then rename: a kill mid-write (the exact
    # scenario this exists for) must never leave a half-written,
    # unparseable meta file behind for the next resume to trip over.
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_path, meta_path)


def try_load_checkpoint(net, prefix):
    """Returns (epoch, global_step, best_acc, rng) if a checkpoint from
    save_checkpoint() exists at `prefix`, having already loaded its
    weights into `net`; returns None (net left untouched) if not."""
    weights_path, meta_path = checkpoint_paths(prefix)
    if not (os.path.exists(weights_path) and os.path.exists(meta_path)):
        return None
    if not net.load(weights_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    rng = np.random.default_rng(7)
    rng.bit_generator.state = meta["rng_state"]
    return meta["epoch"], meta["global_step"], meta["best_acc"], meta["elapsed"], rng


def build_vgg7_network(batch_size, n_classes, lr):
    net = dy.FullConvPCNetwork(batch_size=batch_size, device="gpu")

    def conv(in_c, out_c, hw, pad, pool=1, activation="hard_tanh", activation_deriv="dhard_tanh"):
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
    # muPC scaling OFF: PCX's own PC-CE doesn't use it. PCX-style init
    # must precede compile() (and the randomize_weights() call later in
    # main()): the init-variance change only takes effect on the NEXT
    # randomize_weights() call.
    net.set_use_mu_pc_scaling(False)
    net.set_use_pcx_init(True)
    net.compile()

    net.set_use_ipc(False)
    net.set_use_epc(False)  # classic PC settling, NOT the ePC adjoint trick
    net.set_use_momentum(True, MOMENTUM)
    net.set_use_cross_entropy(True)
    net.set_optimizer("ADAMW")       # PCX's own optimizer
    net.set_psi_optimizer("ADAMW")

    return net


def main():
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    LR = float(sys.argv[3]) if len(sys.argv) > 3 else W_LR
    ir = float(sys.argv[4]) if len(sys.argv) > 4 else IR
    CHECKPOINT_PREFIX = sys.argv[5] if len(sys.argv) > 5 else "checkpoint"
    peak_lr = 1.1 * LR
    end_lr = 0.1 * LR

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    print(f"\n*** TINY IMAGENET, VGG-7 GENUINE PC, PCX PC-CE replica (T={T}, ir={ir}, "
          f"momentum={MOMENTUM}, hard_tanh, PCX init) ***")
    print(f"Comparable to: PCX's own PC-CE 39.49+-2.69% (Table 1) and this codebase's own "
          f"BP-equivalent baseline, 38.78% (imagenet_bp_control_vgg7_ce.py, full 50 epochs).")
    print(f"Training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR:.6g} "
          f"(warmup-cosine to peak {peak_lr:.6g}, end {end_lr:.6g}), AdamW wd={W_WD:.3g}, "
          f"cross-entropy, muPC scaling off, PCX uniform init, hard_tanh, 56x56 crops, "
          f"pooling on, {T} settling steps/batch, ir={ir}.\n")

    net = build_vgg7_network(BATCH_SIZE, N_CLASSES, LR)
    net.set_inference_rate(ir)

    diag_indices = pick_diagnostic_indices(y_val_idx)

    resumed = try_load_checkpoint(net, CHECKPOINT_PREFIX)
    if resumed is not None:
        start_epoch, global_step, best_acc, elapsed_before, rng = resumed
        print(f"Resuming from checkpoint '{CHECKPOINT_PREFIX}': "
              f"epoch={start_epoch}, global_step={global_step}, best_acc={best_acc:.2f}%\n")
    else:
        net.randomize_weights(7)
        run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, BATCH_SIZE, N_CLASSES,
                                "PRE-training")
        start_epoch, global_step, best_acc, elapsed_before = 0, 0, 0.0, 0.0
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

    start_time = perf_counter()

    for epoch in range(start_epoch, EPOCHS):
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
        elapsed = elapsed_before + (perf_counter() - start_time)
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | "
              f"Avg energy: {epoch_energy/n_energy_samples:.4f} | Val top1: {acc:.2f}% | "
              f"Best top1: {best_acc:.2f}% | lr={lr_at_step(global_step):.2e}")

        save_checkpoint(net, CHECKPOINT_PREFIX, epoch + 1, global_step, best_acc, elapsed, rng)

        if epoch == 0:
            run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, BATCH_SIZE, N_CLASSES,
                                    "POST-epoch-1")

    print(f"\n=== Result ===")
    print(f"Deepity VGG-7 genuine PC, PCX PC-CE replica (T={T}, ir={ir}): "
          f"best top-1 = {best_acc:.2f}%")
    print(f"Reference points: this codebase's own BP-equivalent baseline = 38.78%, "
          f"PCX's published PC-CE = 39.49+-2.69%.")


if __name__ == "__main__":
    main()
