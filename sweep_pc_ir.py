"""
Zero-new-code test of Opus's leading hypothesis for why genuine PC
settling (imagenet_pc_vgg7_ce.py) collapses to a near-constant,
input-independent output within ~100 batches, while the BP-equivalent
control (imagenet_bp_control_vgg7_ce.py) trains fine on the identical
architecture/loss.

The claim: with forward-initialized settling, error starts at 0
everywhere except the terminal and propagates downward roughly one hop
per settling step, shrinking by about `ir` each hop. At ir=0.0223 and
~6 conv layers, error reaching the earliest layers is tiny -- but Adam
normalizes each parameter's update to ~lr regardless of gradient
magnitude, so a starved/noisy gradient still gets a full-size,
systematic step. ~100 such steps is enough to push ReLUs into a dead
regime, at which point the network's features (and therefore its
output) stop depending on the input. That matches the collapse timing
observed.

Separately, Deepity's momentum is an EMA (v=b*v+(1-b)*dz, effective
step ~= ir), not optax's SGD-with-momentum trace (v=g+b*v, effective
step ~= ir/(1-b)) -- confirmed by reading UpdateState()'s actual
source. At b=0.55, PCX's own ir=0.0223 corresponds to an effective step
of ~0.0223/0.45 ~= 0.0496 in PCX's own convention, not 0.0223. Porting
the raw number under-drives Deepity's settling by >2x, compounding
exponentially with depth.

This sweeps ir at a few candidates, all via the ALREADY-BOUND
set_inference_rate() -- no new C++ or bindings. Each candidate gets a
short run (not a full epoch) with the real architecture and real data,
checking collapse at a few checkpoints via the SAME predict()-based
diagnostic the main script already uses. If collapse disappears above
some ir threshold and reappears as ir shrinks toward 0.0223, that
supports the starvation hypothesis without needing per-layer
instrumentation. If even ir=0.5 collapses, something else is going on
and the per-layer-RMS test becomes necessary after all.

Usage:
    python sweep_pc_ir.py [N_BATCHES]
(default: 150, matching roughly where the original run's collapse appeared)
"""
import sys
import numpy as np

from imagenet import DATA_DIR, load_wnids, load_train_set_cached, load_val_set_cached, to_one_hot, pick_diagnostic_indices
from imagenet_pc_vgg7_ce import build_vgg7_network, crop_and_normalize, run_collapse_diagnostic, T, W_LR

BATCH_SIZE = 128
CHECK_AT = (0, 50, 100, 150)

# Opus's sweep, plus the momentum-convention-corrected value
# (0.0223 / (1 - 0.55)) and the original PCX-ported value for reference.
IR_CANDIDATES = [0.5, 0.2, 0.05, 0.0223 / (1.0 - 0.55), 0.0223381085942421]


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)
    diag_indices = pick_diagnostic_indices(y_val_idx)

    for ir in IR_CANDIDATES:
        print(f"\n{'='*70}\nir = {ir:.6g}\n{'='*70}")

        net = build_vgg7_network(BATCH_SIZE, N_CLASSES, lr=W_LR)
        net.randomize_weights(7)
        net.set_inference_rate(ir)

        rng = np.random.default_rng(7)
        n_train = len(X_train_u8)
        indices = rng.permutation(n_train)

        run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, BATCH_SIZE, N_CLASSES,
                                f"ir={ir:.4g}, PRE-training")

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
                print(f"  NON-FINITE at batch {b} -- stopping this ir.")
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
        print(f"\n--- ir={ir:.6g}: {status} within {n_batches} batches ---")

    print("\nDone. If collapse goes away above some ir and reappears as ir shrinks toward "
          "0.0223, that supports the starvation hypothesis. If ir=0.5 ALSO collapses, the "
          "mechanism is something else and per-layer error-RMS instrumentation is the next "
          "real step.")


if __name__ == "__main__":
    main()
