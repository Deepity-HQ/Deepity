"""
Separates two candidate explanations for the ir=0.0223 collapse, both
zero-new-code using bindings already in place (set_optimizer("SGD") was
already supported; net.layers exposes individual FullConvPCLayer
objects, each with set_learning_rate() already bound):

Opus's revised mechanism: starved lower layers get a WEAK, possibly
systematically-biased error signal under settling, but Adam normalizes
every parameter's step to ~lr regardless of gradient magnitude -- so a
starved layer's biased gradient still gets a full-size, coherent step
every batch. Compounded over many batches, pre-activations drift,
ReLUs die, outputs go flat. Two ways to test this directly:

  1. Same run, SGD instead of AdamW for the weights. SGD's step size
     scales WITH the gradient magnitude, so a genuinely starved
     (near-zero-gradient) layer should barely move at all under SGD,
     even if it moves under Adam. If SGD stays stable (even if it
     learns slowly), that points at Adam's normalization as the
     amplifier, not the settling/gradient computation itself.
  2. Same run, AdamW, but the bottom 4 (of 7) layers' lr forced to 0 --
     they can't update at all, regardless of what their gradient looks
     like. If THIS stays stable, the bad updates are specifically
     coming from those starved layers.

All three configs below run at the ORIGINAL ir=0.0223 (the exact
config that collapsed), T=12, momentum=0.55, cross-entropy -- only the
optimizer/frozen-layers differ.

Usage:
    python sweep_pc_optimizer_freeze.py [N_BATCHES]
(default: 150, matching sweep_pc_ir.py)
"""
import sys
import numpy as np

from imagenet import DATA_DIR, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices
from imagenet_pc_vgg7_ce import build_vgg7_network, crop_and_normalize, run_collapse_diagnostic, T, W_LR, IR

BATCH_SIZE = 128
CHECK_AT = (0, 50, 100, 150)
N_FROZEN_LAYERS = 4  # bottom 4 of 7 weight-bearing layers


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

    collapsed_at = None
    for b in range(n_batches):
        batch_idx = indices[(b * BATCH_SIZE) % n_train:(b * BATCH_SIZE) % n_train + BATCH_SIZE]
        X_batch = crop_and_normalize(X_train_u8[batch_idx], train=True, rng=rng)
        Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES, eps=0.0)

        want_energy = (b % 20 == 0)
        energy = net.train_step(X_batch, Y_batch, T, want_energy)
        if want_energy:
            print(f"  batch {b:4d}: energy={energy:.4f}")

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
            n_distinct = len(set(np.argmax(preds[:len(diag_indices)], axis=1).tolist()))
            print(f"  batch {b+1:4d}: distinct predicted classes = {n_distinct}/5, "
                  f"output std = {preds[0].std():.6f}")
            if n_distinct == 1 and collapsed_at is None:
                collapsed_at = b + 1

    status = f"COLLAPSED by batch {collapsed_at}" if collapsed_at else "did not collapse"
    print(f"\n--- {label}: {status} within {n_batches} batches ---")


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

    def freeze_bottom(net):
        for layer in net.layers[:N_FROZEN_LAYERS]:
            layer.set_learning_rate(0.0)

    configs = [
        (f"ir={IR:.4g}, AdamW, no freezing (baseline, should reproduce the collapse)", baseline),
        (f"ir={IR:.4g}, SGD instead of AdamW", sgd),
        (f"ir={IR:.4g}, AdamW, bottom {N_FROZEN_LAYERS} layers frozen (lr=0)", freeze_bottom),
    ]

    for label, configure_fn in configs:
        run_one(label, X_train_u8, y_train_idx, X_val_u8, y_val_idx, diag_indices, N_CLASSES,
               n_batches, configure_fn)

    print("\nDone. If SGD stays stable while AdamW collapses, Adam's per-parameter "
          "normalization is amplifying a weak/biased signal from starved layers. If freezing "
          "the bottom layers ALSO stabilizes things, that confirms those specific layers are "
          "the source of the bad updates.")


if __name__ == "__main__":
    main()
