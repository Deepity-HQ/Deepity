"""Cross-check for benchmark.py's finding: Deepity's hand-rolled OpenMP-
over-BLAS parallelization of DirectKPPCNetwork's per-layer-independent
phases (DirectFeedbackUpdate, the settle step, UpdateWeights) scaled
~linearly with depth (25x slower from depth=2->96, same as plain BP),
not the paper's claimed O(1). That C++ implementation dispatches each
layer's phase as a SEPARATE OpenMP-scheduled task, each calling its own
tiny (64x64) BLAS GEMM -- thread fork/join and nested-BLAS-threading
overhead could plausibly dominate at these tiny per-layer sizes and be
the reason the ratio stayed flat, in which case that result would say
more about our dispatch mechanism than about the algorithm itself.

This reimplements the SAME per-phase structure -- NOT bit-exact DKP-PC
math, synthetic/representative timing only, same convention as
benchmark.py's own "no accuracy claim" scoping -- but dispatches each
cross-layer-independent phase as ONE jax.numpy batched einsum over a
stacked [L, ...] tensor: a single XLA-compiled kernel launch per phase,
zero per-layer Python-level or thread-level dispatch at all. This is
the "genuinely P=L, data-parallel, no per-task dispatch overhead"
execution model the paper's O(1)-per-phase claim actually depends on.

Two separate numbers are reported, matching Gemini's two-part critique:
  1. "independent phases only" time vs depth -- if THIS stays flat
     even once batched-dispatch overhead is removed, that's
     implementation-independent confirmation of the Brent's-theorem
     argument (total work is O(L) regardless of how cheaply you
     dispatch it) -- i.e. our C++ result wasn't a dispatch-overhead
     artifact. If it flattens out here where it didn't in C++, that
     means dispatch overhead WAS the dominant cost there.
  2. "full step including forward init" time vs depth -- forward
     propagation is a strict sequential chain (layer i+1 needs layer
     i's output) that no batching scheme can remove; included here to
     check Gemini's separate point that this alone forces O(L) total
     step time regardless of what happens to the backward phases.

A third number, bp_ms, closes a gap the write-up of the first two
numbers missed: this script's only DKP-PC-vs-BP comparison up to that
point was indirect, borrowed from the earlier C++/PyTorch benchmark
where DKP-PC used the flawed thread-per-layer dispatch and BP used
PyTorch's own autodiff -- not an apples-to-apples comparison under
equally good dispatch. bp_ms is BP's actual backward pass (jax.grad
through the same forward_sequential(), fully jit-compiled, same "one
fused XLA graph" standard as independent_phases_batched()), so dkp/bp
here is the first fair comparison between the two under identical
dispatch quality.

CPU-only (no GPU on this machine) via JAX's XLA CPU backend -- same
physical core count and SIMD width as the C++/OpenMP version, so any
difference in scaling behavior is about dispatch mechanism, not access
to more raw parallel hardware. JAX dispatches asynchronously -- every
timed call below is wrapped in block_until_ready() so wall-clock timing
isn't silently undercounting real compute time.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

IN_DIM = 64
HIDDEN = 64
OUT_DIM = 10
BATCH = 128
WARMUP = 3
TIMED = 10
DEPTHS = [2, 4, 8, 16, 32, 64, 96]


def make_params(key: jax.Array, depth: int):
    """depth hidden Linear(HIDDEN,HIDDEN) layers, matching benchmark.py's
    build_dkppcn()/BPMlp() architecture (depth-1 hidden layers plus one
    final HIDDEN->OUT_DIM layer -- here folded into `depth` same-shaped
    [HIDDEN,HIDDEN] layers stackable into one [depth,HIDDEN,HIDDEN]
    tensor, since same-shape stacking is what makes the single-batched-
    einsum dispatch possible in the first place)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W_in = jax.random.normal(k1, (IN_DIM, HIDDEN)) * 0.05
    W = jax.random.normal(k2, (depth, HIDDEN, HIDDEN)) * 0.05
    Psi = jax.random.normal(k3, (depth, OUT_DIM, HIDDEN)) * 0.05
    W_out = jax.random.normal(k4, (HIDDEN, OUT_DIM)) * 0.05
    return W_in, W, Psi, W_out


@jax.jit
def forward_sequential(W_in, W, W_out, x):
    """Strictly sequential across layers -- layer i+1's input is layer
    i's output -- via lax.scan, the honest JAX primitive for a real
    carry dependency (not vmap/einsum, which only fit the OTHER phases
    below precisely because those have no such dependency)."""
    z0 = jnp.tanh(x @ W_in)

    def step(z, Wl):
        return jnp.tanh(z @ Wl), None

    zL, _ = jax.lax.scan(step, z0, W)
    return zL @ W_out


@jax.jit
def independent_phases_batched(W, Psi, z, err):
    """Stand-in for DirectFeedbackUpdate + settle-step + UpdateWeights,
    each cross-layer-independent (every layer reads only its own z/W/Psi
    plus the shared, already-frozen terminal error -- never another
    layer's freshly-computed output), so all `depth` layers' versions of
    each op are expressed as ONE einsum with an explicit leading `l`
    (layer) axis -- one XLA kernel launch total, not `depth` separate
    calls.
    """
    # DirectFeedbackUpdate stand-in: each layer projects the shared
    # terminal error through its own feedback weights.
    proj = jnp.einsum("bt,lth->lbh", err, Psi)
    feedback_grad = jnp.einsum("lbh,bt->lth", proj, err)
    Psi = Psi - 1e-3 * feedback_grad

    # Settle-step stand-in: each layer computes its own prediction from
    # its own (unchanged this phase) z, compares to a fixed target.
    mu = jnp.einsum("lbh,lho->lbo", z, W)
    e = proj - mu
    z = z + 0.1 * jnp.einsum("lbo,lho->lbh", e, W)

    # UpdateWeights stand-in: each layer's own outer-product gradient.
    weight_grad = jnp.einsum("lbh,lbo->lho", z, e)
    W = W - 1e-3 * weight_grad

    return W, Psi, z


def mse_loss(W_in, W, W_out, x, target):
    pred = forward_sequential(W_in, W, W_out, x)
    return jnp.mean((pred - target) ** 2)


# BP's actual backward pass, under the SAME "good dispatch" standard as
# independent_phases_batched() above -- one jit-compiled XLA graph, no
# per-layer Python loop. This is the comparison Gemini's synthesis
# skipped: our only DKP-PC-vs-BP numbers so far came from the earlier
# C++/PyTorch benchmark, where DKP-PC used the flawed thread-per-layer
# dispatch and BP used PyTorch's own autodiff -- not apples-to-apples.
bp_grad_fn = jax.jit(jax.grad(mse_loss, argnums=(0, 1, 2)))


def time_fn(fn, *args) -> float:
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(TIMED):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / TIMED


def main() -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (BATCH, IN_DIM))
    err = jax.random.normal(key, (BATCH, OUT_DIM))
    target = jax.random.normal(key, (BATCH, OUT_DIM))

    print(f"IN_DIM={IN_DIM} HIDDEN={HIDDEN} OUT_DIM={OUT_DIM} BATCH={BATCH}  (JAX CPU backend)\n")
    print(f"{'depth':>6}{'phases_ms':>12}{'full_step_ms':>14}{'bp_ms':>10}{'dkp/bp':>10}")

    phase_results = []
    full_results = []
    bp_results = []
    for depth in DEPTHS:
        W_in, W, Psi, W_out = make_params(key, depth)
        z = jax.random.normal(key, (depth, BATCH, HIDDEN))

        phases_s = time_fn(independent_phases_batched, W, Psi, z, err)

        def full_step(W_in, W, W_out, x, Psi, z, err):
            out = forward_sequential(W_in, W, W_out, x)
            Wn, Psin, zn = independent_phases_batched(W, Psi, z, err)
            return out, Wn, Psin, zn

        full_s = time_fn(jax.jit(full_step), W_in, W, W_out, x, Psi, z, err)
        bp_s = time_fn(bp_grad_fn, W_in, W, W_out, x, target)

        phase_results.append((depth, phases_s))
        full_results.append((depth, full_s))
        bp_results.append((depth, bp_s))
        print(
            f"{depth:>6}{phases_s * 1e3:>12.4f}{full_s * 1e3:>14.4f}"
            f"{bp_s * 1e3:>10.4f}{full_s / bp_s:>10.3f}"
        )

    d0, p0 = phase_results[0]
    dN, pN = phase_results[-1]
    f0 = full_results[0][1]
    fN = full_results[-1][1]
    b0 = bp_results[0][1]
    bN = bp_results[-1][1]
    print(f"\nGrowth from depth={d0} to depth={dN} ({dN / d0:.1f}x more layers):")
    print(f"  independent phases only (batched, zero per-layer dispatch): {pN / p0:.2f}x slower")
    print(f"  full step (forward + phases):                               {fN / f0:.2f}x slower")
    print(f"  BP (jax.grad, same forward pass, equally well-dispatched):  {bN / b0:.2f}x slower")


if __name__ == "__main__":
    main()
