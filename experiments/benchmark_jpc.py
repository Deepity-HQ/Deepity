"""
Benchmarks jpc (JAX predictive-coding library, github.com/thebuckleylab/jpc)
on the SAME architecture/batch size as Deepity's own tGflopsBenchmark.cpp
(784-512-256-64-10, batch=256), as a CPU-vs-CPU comparison point for the
"JAX's lax.scan-fused settling loop should beat a hand-written CPU engine"
claim.

Deliberately uses jpc's OWN idiomatic layer construction (the
activate-then-linear convention from jpc.make_mlp, see its docstring) and
its OWN default settling mechanism (Heun ODE solver via diffrax,
max_t1=20, default PID step-size controller) rather than forcing it to
replicate Deepity's exact mu=phi(W@z+b) formula -- this is "jpc's own best
effort" on a matched architecture/batch, not a matched-equations
comparison. jpc's make_mlp() only supports a single uniform width repeated
`depth` times, so the varying-width layer list is built by hand, copying
make_mlp's own construction pattern (see jpc/_utils.py).

Note on why this isn't reported as GFLOPS: jpc's default solver takes an
ADAPTIVE number of Heun steps within the [0, max_t1] integration window
(PID step-size control) -- confirmed via jpc's own `t_max` diagnostic to
be just 4 actual steps for this architecture/tolerance, not 20. That's
not a comparable amount of settling work to Deepity's fixed 20 Euler
steps, so a raw wall-clock comparison against jpc's defaults alone would
be comparing different amounts of work, not engine speed. This script
therefore runs TWO jpc configurations: jpc's own defaults (adaptive,
whatever work it decides to do) AND a forced fixed-step Euler solver
with exactly 20 steps (diffrax.Euler + ConstantStepSize(dt=1.0)),
matching Deepity's own integration method and step count exactly, as the
genuinely apples-to-apples number.
"""

import time

import diffrax
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import jpc

LAYER_SIZES = [784, 512, 256, 64, 10]
BATCH_SIZE = 256
WARMUP_ITERS = 2
TIMED_ITERS = 30


def build_model(key):
    # Matches jpc.make_mlp()'s own construction (jpc/_utils.py):
    # nn.Sequential([nn.Lambda(act_fn_l), linear]) per layer, Identity()
    # on the very first layer, sigmoid on the rest. The only difference
    # from calling make_mlp() directly is per-layer widths instead of a
    # single repeated `width`.
    depth = len(LAYER_SIZES) - 1
    subkeys = jr.split(key, depth)
    layers = []
    for i in range(depth):
        act_fn_l = nn.Identity() if i == 0 else jax.nn.sigmoid
        linear = nn.Linear(
            LAYER_SIZES[i], LAYER_SIZES[i + 1], use_bias=True, key=subkeys[i]
        )
        layers.append(nn.Sequential([nn.Lambda(act_fn_l), linear]))
    return layers


def main():
    key = jr.PRNGKey(42)
    model_key, data_key = jr.split(key)

    model = build_model(model_key)
    optim = optax.sgd(learning_rate=0.001)
    opt_state = optim.init((eqx.filter(model, eqx.is_array), None))

    x = jr.uniform(data_key, (BATCH_SIZE, LAYER_SIZES[0]))
    # Soft one-hot targets, matching tGflopsBenchmark.cpp's own Y
    # construction (0.001 background, 0.999 at one active index per row).
    y = jnp.full((BATCH_SIZE, LAYER_SIZES[-1]), 0.001)
    active_idx = jnp.arange(BATCH_SIZE) % LAYER_SIZES[-1]
    y = y.at[jnp.arange(BATCH_SIZE), active_idx].set(0.999)

    def make_train_step(**pc_step_kwargs):
        def train_step(model, opt_state, x, y):
            result = jpc.make_pc_step(
                model=model,
                optim=optim,
                opt_state=opt_state,
                output=y,
                input=x,
                loss_id="mse",
                **pc_step_kwargs,
            )
            return result["model"], result["opt_state"]

        return train_step

    def report_t_max(label, model, opt_state, **pc_step_kwargs):
        # Separate, UNTIMED call with record_activities=True: recording
        # the full trajectory adds real overhead of its own, so this is
        # never run inside the timed loop, only as a diagnostic.
        result = jpc.make_pc_step(
            model=model, optim=optim, opt_state=opt_state, output=y, input=x,
            loss_id="mse", record_activities=True, **pc_step_kwargs,
        )
        print(f"{label}: actual solver steps taken (t_max) = {result['t_max']}")

    def run_config(name, train_step, model, opt_state):
        print(f"\n=== {name} ===")
        for _ in range(WARMUP_ITERS):
            model, opt_state = train_step(model, opt_state, x, y)
            jax.block_until_ready(model)

        times = []
        for _ in range(TIMED_ITERS):
            t0 = time.perf_counter()
            model, opt_state = train_step(model, opt_state, x, y)
            jax.block_until_ready(model)
            times.append(time.perf_counter() - t0)

        times.sort()
        median = times[len(times) // 2]
        mean = sum(times) / len(times)
        best = times[0]

        print(f"Timed iterations: {TIMED_ITERS}")
        print(f"Median time/step: {median * 1000:.3f} ms")
        print(f"Mean time/step:   {mean * 1000:.3f} ms")
        print(f"Best time/step:   {best * 1000:.3f} ms")
        print(f"Median throughput: {BATCH_SIZE / median:.1f} samples/sec")
        return model, opt_state

    print(f"Architecture: {LAYER_SIZES}, batch={BATCH_SIZE}")

    # Config 1: jpc's own defaults -- Heun ODE solver, adaptive PID
    # step-size control, max_t1=20. t_max reported separately (untimed)
    # since recording activities adds its own overhead.
    report_t_max("jpc defaults (adaptive Heun)", model, opt_state)
    default_step = make_train_step()
    model, opt_state = run_config("jpc defaults (adaptive Heun)", default_step, model, opt_state)

    # Config 2: forced fixed-step Euler, exactly 20 steps -- matches
    # Deepity's tGflopsBenchmark.cpp's own integration method (Euler)
    # and step count exactly (dt=1.0, max_t1=20, ConstantStepSize =
    # always all 20 steps, no diagnostic needed). This is the
    # apples-to-apples number.
    euler_kwargs = dict(
        ode_solver=diffrax.Euler(),
        stepsize_controller=diffrax.ConstantStepSize(),
        dt=1.0,
        max_t1=20,
    )
    euler_step = make_train_step(**euler_kwargs)
    model, opt_state = run_config("jpc forced fixed Euler, T=20 (matched)", euler_step, model, opt_state)


if __name__ == "__main__":
    main()
