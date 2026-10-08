"""
Instrumentation, not guessing: watches genuine multi-step PC settling
unfold on the ACTUAL VGG-7 architecture (imagenet_pc_vgg7_ce.py's own
build_vgg7_network()) with REAL training batches, before committing to
another long GPU run. The run that collapsed showed two symptoms --
epoch-1 collapse to near-zero output variance, and scattered
multi-million energy spikes recovering the next batch -- consistent
with ir=0.0223 (ported directly from PCX's own JAX/optax implementation)
being poorly calibrated for Deepity's actual error-signal scale. This
script checks that directly instead of assuming it: for several
candidate ir values, it settles for T=12 steps on several REAL batches
(no weight update -- debug_settle_energy_trace() never touches weights)
and prints the energy trajectory step by step. A well-calibrated ir
should show energy settling toward a bounded value; a too-large one
should show growth or oscillation within the trace itself, visible
immediately, on the FIRST batch, without waiting for a single epoch.

Usage:
    python diagnose_pc_settling.py
"""
import numpy as np

from imagenet import DATA_DIR, load_wnids, load_train_set_cached
from imagenet_pc_vgg7_ce import build_vgg7_network, crop_and_normalize, T

BATCH_SIZE = 128
IR_CANDIDATES = [0.0223381085942421, 0.01, 0.005, 0.002, 0.001, 0.0005]
N_BATCHES_TO_CHECK = 3


def main():
    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)

    net = build_vgg7_network(BATCH_SIZE, N_CLASSES, lr=7.961890843897934e-05)
    net.randomize_weights(7)

    rng = np.random.default_rng(0)

    print(f"*** Settling diagnostic: {len(IR_CANDIDATES)} ir candidates x "
          f"{N_BATCHES_TO_CHECK} real batches, T={T} steps each, weights NEVER updated ***\n")

    for batch_i in range(N_BATCHES_TO_CHECK):
        idx = rng.choice(len(X_train_u8), size=BATCH_SIZE, replace=False)
        X_batch = crop_and_normalize(X_train_u8[idx], train=True, rng=rng)
        Y_batch = np.zeros((BATCH_SIZE, N_CLASSES), dtype=np.float32)
        Y_batch[np.arange(BATCH_SIZE), y_train_idx[idx]] = 1.0
        Y_batch = Y_batch.reshape(-1)

        print(f"=== Batch {batch_i} ===")
        for ir in IR_CANDIDATES:
            net.set_inference_rate(ir)
            trace = net.debug_settle_energy_trace(X_batch, Y_batch, T)

            finite = np.isfinite(trace)
            if not finite.all():
                status = f"NON-FINITE at step {np.argmax(~finite)}"
            elif trace[-1] > 3 * trace[0] and trace[0] > 1.0:
                status = "GROWING (energy increased >3x over settling)"
            elif trace.max() > 10 * np.median(trace):
                status = f"SPIKE (max={trace.max():.2e} vs median={np.median(trace):.2e})"
            else:
                status = "bounded"

            trace_str = " ".join(f"{e:.3g}" for e in trace)
            print(f"  ir={ir:<12.4g} [{status:40s}] trace: {trace_str}")
        print()

    print("Interpretation: an ir where EVERY batch shows 'bounded' (energy settling toward a "
          "stable value, not growing/spiking within these T=12 steps) is a reasonable candidate "
          "for an actual short training run. If ALL candidates spike on SOME batch, the "
          "instability may be data-dependent (specific batches with extreme values after "
          "augmentation) rather than purely an ir scale issue -- worth knowing before assuming "
          "a smaller ir alone fixes it.")


if __name__ == "__main__":
    main()
