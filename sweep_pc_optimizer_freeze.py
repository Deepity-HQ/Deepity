"""
v2, correcting two methodology issues Opus flagged in the first run:

1. The SGD comparison wasn't fair -- it reused W_LR (8e-5), a rate tuned
   for Adam's per-parameter normalization. Plain SGD at that rate barely
   moves at all, so "it stayed stable" proved nothing about whether the
   gradient itself was fine; it just proved the network was nearly
   frozen. This version gives SGD its own, much larger learning rate
   (SGD_LR below) so it actually has a chance to learn or visibly
   misbehave.
2. "N/5 distinct argmax classes" is a weak collapse signal -- near-
   constant logits still argmax to different classes depending on which
   position has the largest noise. This version uses
   run_collapse_diagnostic()'s corrected cross-input logit std (near 0 =
   output doesn't depend on input, regardless of argmax) plus real
   validation accuracy over several batches, both from
   imagenet_pc_vgg7_ce.py's own (now fixed) helpers.

Tests, all at the ORIGINAL ir=0.0223 (T=12, momentum=0.55, cross-entropy):
  - AdamW baseline (should reproduce the collapse)
  - SGD at a real SGD learning rate
  - AdamW with the bottom 4 of 7 weight-bearing layers frozen (lr=0)

Usage:
    python sweep_pc_optimizer_freeze.py [N_BATCHES]
(default: 150)
"""
import sys
import numpy as np

from imagenet import DATA_DIR, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices
from imagenet_pc_vgg7_ce import (build_vgg7_network, crop_and_normalize, run_collapse_diagnostic,
                                 quick_val_accuracy, T, W_LR, IR)

BATCH_SIZE = 128
CHECK_AT = (0, 50, 100, 150)
N_FROZEN_LAYERS = 4  # bottom 4 of 7 weight-bearing layers
SGD_LR = 0.02         # a real SGD rate, NOT W_LR -- see module docstring


def run_one(label, X_train_u8, y_train_idx, X_val_u8, y_val_idx, diag_indices, N_CLASSES,
           n_batches, configure_fn):
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    net = build_vgg7_network(BATCH_SIZE, N_CLASSES, lr=W_LR)
    net.randomize_weights(7)
    net.set_inference_rate(IR)
    configure_fn(net)

    rng = np.random.default_rng(7)
    n_train = len(X_train_u8)
    indices = rng.permutation(n_train)

    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, BATCH_SIZE, N_CLASSES,
                            f"{label}, PRE-training")

    for b in range(n_batches):
        batch_idx = indices[(b * BATCH_SIZE) % n_train:(b * BATCH_SIZE) % n_train + BATCH_SIZE]
        X_batch = crop_and_normalize(X_train_u8[batch_idx], train=True, rng=rng)
        Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES, eps=0.0)

        want_energy = (b % 20 == 0)
        energy = net.train_step(X_batch, Y_batch, T, want_energy)
        if want_energy:
            print(f"  batch {b:4d}: energy={energy:.4f} (reference only, see Opus's note on why "
                  f"this isn't a reliable cross-config stability signal)")

        if not np.isfinite(energy):
            print(f"  NON-FINITE at batch {b} -- stopping this config.")
            break

        if b + 1 in CHECK_AT:
            diag_X = crop_and_normalize(X_val_u8[diag_indices], train=False,
                                        rng=np.random.default_rng(0))
            pad_n = BATCH_SIZE - len(diag_X)
            filler = crop_and_normalize(X_val_u8[len(diag_indices):len(diag_indices) + pad_n],
                                        train=False, rng=np.random.default_rng(0))
            diag_X_padded = np.vstack([diag_X, filler])
            preds = net.predict(diag_X_padded, 0).reshape(BATCH_SIZE, N_CLASSES)
            cross_input_std = preds[:len(diag_indices)].std(axis=0).mean()
            acc = quick_val_accuracy(net, X_val_u8, y_val_idx, BATCH_SIZE, N_CLASSES, n_batches=10)
            print(f"  batch {b+1:4d}: cross-input logit std = {cross_input_std:.6f}, "
                  f"val top1 (10 batches) = {acc:.2f}%")

    print(f"--- {label}: done ---")


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)
    diag_indices = pick_diagnostic_indices(y_val_idx)

    def baseline(net):
        pass  # AdamW, no freezing -- the original failing config, for reference

    def sgd(net):
        net.set_optimizer("SGD")
        for layer in net.layers:
            layer.set_learning_rate(SGD_LR)

    def freeze_bottom(net):
        for layer in net.layers[:N_FROZEN_LAYERS]:
            layer.set_learning_rate(0.0)

    configs = [
        (f"ir={IR:.4g}, AdamW, no freezing (baseline, should reproduce the collapse)", baseline),
        (f"ir={IR:.4g}, SGD, lr={SGD_LR} (a REAL SGD rate, not W_LR)", sgd),
        (f"ir={IR:.4g}, AdamW, bottom {N_FROZEN_LAYERS} layers frozen (lr=0)", freeze_bottom),
    ]

    for label, configure_fn in configs:
        run_one(label, X_train_u8, y_train_idx, X_val_u8, y_val_idx, diag_indices, N_CLASSES,
               n_batches, configure_fn)

    print("\nDone. Judge each config by cross-input logit std and val accuracy, not energy or "
          "distinct-argmax-count. If SGD's accuracy rises above chance (0.5%) while AdamW's "
          "logit std collapses toward 0, that supports Adam's per-parameter normalization as "
          "the amplifier. If freezing the bottom layers ALSO fails to keep logit std up, the "
          "bad updates aren't confined to those layers.")


if __name__ == "__main__":
    main()
