"""
Opus's experiment #1 and #3 together: per-layer error mean/RMS, weight
norm, and dead-mu fraction logged every 10 batches at the ORIGINAL
ir=0.0223 (the exact config that collapsed), using the new
get_all_layer_diagnostics() binding -- plus a second run with Adam's
epsilon raised from its hardcoded default (1e-8) to 1e-4, via the new
set_adam_epsilon(), to test whether capping how much a near-zero
gradient gets amplified is enough to stabilize training on its own.

What to look for in the baseline (default epsilon) run:
  - A layer whose error MEAN is comparable in magnitude to its error RMS
    (not just small, but consistently signed) is carrying a systematic
    bias, not just noise -- Opus's "lower layers starve, then Adam
    amplifies a consistent bias" mechanism specifically needs this.
  - Rising dead-mu fraction in any ReLU layer, especially if it climbs
    right before the energy spike / logit-std collapse.
  - Whether the weight norm of any layer grows sharply right before the
    collapse (the "update-to-weight-norm ratio" Opus asked for, without
    needing a separate before/after capture -- a sharp jump in the norm
    ITSELF across consecutive checkpoints is the same signal).

Usage:
    python diagnose_pc_layer_stats.py [N_BATCHES]
(default: 150)
"""
import sys
import numpy as np

from imagenet import DATA_DIR, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices
from imagenet_pc_vgg7_ce import build_vgg7_network, crop_and_normalize, quick_val_accuracy, T, W_LR, IR

BATCH_SIZE = 128
CHECK_EVERY = 10
LAYER_NAMES = [f"L{i}" for i in range(7)] + ["terminal"]  # 6 conv + 1x1 classifier + terminal


def run_one(label, X_train_u8, y_train_idx, X_val_u8, y_val_idx, diag_indices, N_CLASSES,
           n_batches, adam_epsilon):
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    net = build_vgg7_network(BATCH_SIZE, N_CLASSES, lr=W_LR)
    net.randomize_weights(7)
    net.set_inference_rate(IR)
    net.set_adam_epsilon(adam_epsilon)

    rng = np.random.default_rng(7)
    n_train = len(X_train_u8)
    indices = rng.permutation(n_train)

    for b in range(n_batches):
        batch_idx = indices[(b * BATCH_SIZE) % n_train:(b * BATCH_SIZE) % n_train + BATCH_SIZE]
        X_batch = crop_and_normalize(X_train_u8[batch_idx], train=True, rng=rng)
        Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES, eps=0.0)

        energy = net.train_step(X_batch, Y_batch, T, False)
        if not np.isfinite(energy):
            print(f"  NON-FINITE at batch {b} -- stopping this config.")
            break

        if b % CHECK_EVERY == 0:
            diag = net.get_all_layer_diagnostics()  # [layer, 4]: mean, rms, norm, dead_frac

            diag_X = crop_and_normalize(X_val_u8[diag_indices], train=False,
                                        rng=np.random.default_rng(0))
            pad_n = BATCH_SIZE - len(diag_X)
            filler = crop_and_normalize(X_val_u8[len(diag_indices):len(diag_indices) + pad_n],
                                        train=False, rng=np.random.default_rng(0))
            diag_X_padded = np.vstack([diag_X, filler])
            preds = net.predict(diag_X_padded, 0).reshape(BATCH_SIZE, N_CLASSES)
            cross_input_std = preds[:len(diag_indices)].std(axis=0).mean()

            print(f"\n  --- batch {b} --- (cross-input logit std = {cross_input_std:.6f})")
            for i, name in enumerate(LAYER_NAMES[:len(diag)]):
                mean, rms, norm, dead = diag[i]
                sign_ratio = abs(mean) / rms if rms > 1e-12 else 0.0
                print(f"    {name:9s}: error_mean={mean:+.6f} error_rms={rms:.6f} "
                      f"|mean|/rms={sign_ratio:.3f} weight_norm={norm:.4f} dead_frac={dead:.4f}")

    acc = quick_val_accuracy(net, X_val_u8, y_val_idx, BATCH_SIZE, N_CLASSES, n_batches=10)
    print(f"\n--- {label}: final val top1 (10 batches) = {acc:.2f}% ---")


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)
    diag_indices = pick_diagnostic_indices(y_val_idx)

    run_one(f"ir={IR:.4g}, AdamW, default epsilon=1e-8 (baseline, should reproduce the collapse)",
           X_train_u8, y_train_idx, X_val_u8, y_val_idx, diag_indices, N_CLASSES, n_batches,
           adam_epsilon=1e-8)

    run_one(f"ir={IR:.4g}, AdamW, epsilon=1e-4 (Opus's cheap mitigation)",
           X_train_u8, y_train_idx, X_val_u8, y_val_idx, diag_indices, N_CLASSES, n_batches,
           adam_epsilon=1e-4)

    print("\nDone. In the baseline run: does any layer's |error_mean|/rms stay elevated "
          "(systematic bias, not just noise) in the batches before logit std collapses? Does "
          "any layer's weight_norm or dead_frac jump sharply right before that? In the "
          "epsilon=1e-4 run: does raising epsilon alone keep logit std from collapsing?")


if __name__ == "__main__":
    main()
