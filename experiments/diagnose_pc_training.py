"""
Follow-up to diagnose_pc_settling.py's finding: settling on FRESH, random
weights is stable for every ir candidate tested, including 0.0223 (the
value actually used in the run that collapsed). So the instability isn't
in the settling dynamics at init -- it has to emerge DURING real
training, as AdamW evolves the weights, somewhere in roughly the first
100 batches (the original run's energy dropped 117170 -> 823 by batch
100, then the collapse). This script watches that directly: runs the
REAL training loop (train_step(), weights updating, actual ir=0.0223)
for the first several hundred batches, and every few batches, ALSO
calls debug_settle_energy_trace() on a FRESH held-out validation batch
with the CURRENT (evolving) weights -- no interference with training
state, since that call never touches weights. If settling-on-a-fresh-
batch degrades (grows/spikes) as training proceeds, that pins the
instability to weight evolution, not to the settling mechanism itself.

Usage (from research/, since this imports imagenet/imagenet_pc_vgg7_ce as
top-level modules):
    cd research && python ../experiments/diagnose_pc_training.py [N_BATCHES]
(default: 200, covering where the original run's collapse appeared)
"""
import sys
import numpy as np

from imagenet import DATA_DIR, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot
from imagenet_pc_vgg7_ce import build_vgg7_network, crop_and_normalize, T, IR, W_LR

BATCH_SIZE = 128
CHECK_EVERY = 10


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    net = build_vgg7_network(BATCH_SIZE, N_CLASSES, lr=W_LR)
    net.randomize_weights(7)

    rng = np.random.default_rng(7)
    val_rng = np.random.default_rng(999)  # separate stream, fixed held-out batch each check

    print(f"*** Watching real training (ir={IR}, T={T}) for {n_batches} batches, checking "
          f"settle-trace health on a fresh held-out batch every {CHECK_EVERY} batches ***\n")

    val_idx = val_rng.choice(len(X_val_u8), size=BATCH_SIZE, replace=False)
    X_val_fixed = crop_and_normalize(X_val_u8[val_idx], train=False, rng=val_rng)
    Y_val_fixed = to_one_hot(y_val_idx[val_idx], N_CLASSES, eps=0.0)

    n_train = len(X_train_u8)
    indices = rng.permutation(n_train)

    for b in range(n_batches):
        batch_idx = indices[(b * BATCH_SIZE) % n_train:(b * BATCH_SIZE) % n_train + BATCH_SIZE]
        X_batch = crop_and_normalize(X_train_u8[batch_idx], train=True, rng=rng)
        Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES, eps=0.0)

        energy = net.train_step(X_batch, Y_batch, T, True)

        if b % CHECK_EVERY == 0 or not np.isfinite(energy):
            trace = net.debug_settle_energy_trace(X_val_fixed, Y_val_fixed, T)
            finite = np.isfinite(trace)
            if not finite.all():
                settle_status = f"NON-FINITE at settle-step {np.argmax(~finite)}"
            elif trace[-1] > 3 * trace[0] and trace[0] > 1.0:
                settle_status = f"GROWING ({trace[0]:.3g} -> {trace[-1]:.3g})"
            elif trace.max() > 10 * np.median(trace):
                settle_status = f"SPIKE (max={trace.max():.3g}, median={np.median(trace):.3g})"
            else:
                settle_status = f"bounded ({trace[0]:.3g} -> {trace[-1]:.3g})"

            print(f"batch {b:4d}: train_energy={energy:12.4f} | held-out settle-trace: {settle_status}")

        if not np.isfinite(energy):
            print(f"\nTrain energy went non-finite at batch {b} -- stopping early.")
            return

    print("\nDone. Compare where (if anywhere) the held-out settle-trace status changes from "
          "'bounded' to 'growing'/'spike' against the batch index -- that pinpoints whether "
          "instability was already present early, or developed partway through these batches.")


if __name__ == "__main__":
    main()