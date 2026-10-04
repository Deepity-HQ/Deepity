import numpy as np
import os
import sys
from time import perf_counter
from pydeepity import dy  # pyright: ignore[reportAttributeAccessIssue] -- dy is the compiled native extension; see pydeepity/_backend.py and pyrightconfig.json's pydeepity/bindings suppressions for the same reason
from PIL import Image

IMG_SIZE = 64
N_CHANNELS = 3
DATA_DIR = "tiny-imagenet-200"


# =============================================================================
# Data loading (unchanged from the previous version of this script -- this
# part was already solid: JPEG decoding, CHW conversion, and an on-disk
# .npz cache so repeat runs skip straight to loading an already-converted
# array instead of re-decoding 110,000 JPEGs every time).
# =============================================================================

def load_wnids(data_dir):
    with open(os.path.join(data_dir, "wnids.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def load_image_as_chw(path):
    # PIL gives (H,W,3); transposed to (3,H,W) -- channels-first, matching
    # this codebase's conv layer convention (ConvPCLayer/FullConvPCLayer
    # beliefs are shaped {batch, in_channels, in_height, in_width}).
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # (H,W,3)
    arr = arr.transpose(2, 0, 1)  # (3,H,W)
    return arr.reshape(-1)


def load_train_set(data_dir, wnids):
    print("Loading Tiny ImageNet training set (100,000 images)...")
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    img_dim = N_CHANNELS * IMG_SIZE * IMG_SIZE

    X = np.zeros((len(wnids) * 500, img_dim), dtype=np.uint8)
    y_idx = np.zeros(len(wnids) * 500, dtype=np.int64)

    pos = 0
    for wnid in wnids:
        img_dir = os.path.join(data_dir, "train", wnid, "images")
        filenames = sorted(os.listdir(img_dir))
        for fname in filenames:
            X[pos] = load_image_as_chw(os.path.join(img_dir, fname))
            y_idx[pos] = wnid_to_idx[wnid]
            pos += 1
        print(f"  loaded class {wnid} ({pos}/{len(wnids) * 500})", end="\r")
    print()

    return X[:pos], y_idx[:pos], len(wnids)


def load_val_set(data_dir, wnids):
    print("Loading Tiny ImageNet validation set (10,000 images)...")
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    img_dim = N_CHANNELS * IMG_SIZE * IMG_SIZE

    annotations = {}
    with open(os.path.join(data_dir, "val", "val_annotations.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            annotations[parts[0]] = parts[1]

    img_dir = os.path.join(data_dir, "val", "images")
    filenames = sorted(annotations.keys())

    X = np.zeros((len(filenames), img_dim), dtype=np.uint8)
    y_idx = np.zeros(len(filenames), dtype=np.int64)

    for i, fname in enumerate(filenames):
        X[i] = load_image_as_chw(os.path.join(img_dir, fname))
        y_idx[i] = wnid_to_idx[annotations[fname]]
        if i % 1000 == 0:
            print(f"  loaded {i}/{len(filenames)}", end="\r")
    print()

    return X, y_idx


def hwc_flat_to_chw_flat(X_hwc_flat):
    """Fast, in-memory conversion from the old (HWC, pre-transpose)
    flat cache layout to CHW -- reshape+transpose, no JPEG re-decode."""
    n = X_hwc_flat.shape[0]
    X_hwc = X_hwc_flat.reshape(n, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    X_chw = X_hwc.transpose(0, 3, 1, 2)
    return X_chw.reshape(n, -1)


def load_train_set_cached(data_dir, wnids):
    chw_cache = os.path.join(data_dir, "_train_cache_chw.npz")
    if os.path.exists(chw_cache):
        print(f"Loading cached (CHW) training set from {chw_cache}...")
        data = np.load(chw_cache)
        return data["X"], data["y"], len(wnids)

    old_cache = os.path.join(data_dir, "_train_cache.npz")
    if os.path.exists(old_cache):
        print(f"Found existing HWC cache at {old_cache} -- converting to CHW "
              f"in memory (fast) instead of re-decoding {len(wnids) * 500} JPEGs...")
        data = np.load(old_cache)
        X = hwc_flat_to_chw_flat(data["X"])
        y_idx = data["y"]
        print(f"Caching converted result to {chw_cache} for future runs...")
        np.savez_compressed(chw_cache, X=X, y=y_idx)
        return X, y_idx, len(wnids)

    X, y_idx, n_classes = load_train_set(data_dir, wnids)
    print(f"Caching to {chw_cache} for future runs...")
    np.savez_compressed(chw_cache, X=X, y=y_idx)
    return X, y_idx, n_classes


def load_val_set_cached(data_dir, wnids):
    chw_cache = os.path.join(data_dir, "_val_cache_chw.npz")
    if os.path.exists(chw_cache):
        print(f"Loading cached (CHW) validation set from {chw_cache}...")
        data = np.load(chw_cache)
        return data["X"], data["y"]

    old_cache = os.path.join(data_dir, "_val_cache.npz")
    if os.path.exists(old_cache):
        print(f"Found existing HWC cache at {old_cache} -- converting to CHW "
              f"in memory (fast) instead of re-decoding 10,000 JPEGs...")
        data = np.load(old_cache)
        X = hwc_flat_to_chw_flat(data["X"])
        y_idx = data["y"]
        print(f"Caching converted result to {chw_cache} for future runs...")
        np.savez_compressed(chw_cache, X=X, y=y_idx)
        return X, y_idx

    X, y_idx = load_val_set(data_dir, wnids)
    print(f"Caching to {chw_cache} for future runs...")
    np.savez_compressed(chw_cache, X=X, y=y_idx)
    return X, y_idx


# =============================================================================
# Normalization: FIXED, dataset-level statistics computed ONCE, not
# recomputed per batch.
#
# The previous version of this script called (X - X.mean()) / X.std() on
# every single batch -- every batch got a slightly different normalization
# depending on which 250 images happened to land in it, train and
# validation batches were normalized inconsistently with each other, and
# the tiny 5-image diagnostic batch (padded out with filler images, see
# run_collapse_diagnostic below) got yet another, even noisier, statistic.
# Standard practice is to compute mean/std once from training data and
# apply those same fixed numbers everywhere. Done here per-channel (R/G/B
# separately), which is both more standard and more correct than a single
# global scalar for natural images, whose three channels don't share the
# same statistics.
# =============================================================================

def compute_normalization_stats(X_train_u8, n_sample=2000, seed=0):
    # 2000 is already far more than needed for a stable per-channel
    # mean/std estimate (law of large numbers saturates well before
    # this for a reasonably homogeneous image dataset) -- kept modest
    # specifically to bound this function's transient memory: the
    # uint8->float32 cast below is the single largest temporary
    # allocation in this script's data pipeline (4x the sample's raw
    # byte size), and host RAM on a shared HPC node is often the
    # tightest resource, not compute.
    rng = np.random.default_rng(seed)
    n = min(n_sample, len(X_train_u8))
    idx = rng.choice(len(X_train_u8), size=n, replace=False)
    sample = X_train_u8[idx].astype(np.float32).reshape(n, N_CHANNELS, IMG_SIZE * IMG_SIZE)
    mean = sample.mean(axis=(0, 2))  # (N_CHANNELS,)
    std = sample.std(axis=(0, 2))    # (N_CHANNELS,)
    return mean.astype(np.float32), std.astype(np.float32)


def to_float_batch(X_uint8_batch, mean, std):
    n = len(X_uint8_batch)
    X = X_uint8_batch.astype(np.float32).reshape(n, N_CHANNELS, IMG_SIZE * IMG_SIZE)
    X = (X - mean[None, :, None]) / (std[None, :, None] + 1e-8)
    return X.reshape(n, -1)


def to_one_hot(y_idx_batch, n_classes, eps=0.001):
    Y = np.full((len(y_idx_batch), n_classes), eps, dtype=np.float32)
    Y[np.arange(len(y_idx_batch)), y_idx_batch] = 1.0 - eps
    return Y


def pick_diagnostic_indices(y_val_idx, n=5):
    indices, seen = [], set()
    for i in range(len(y_val_idx)):
        if y_val_idx[i] not in seen:
            indices.append(i)
            seen.add(y_val_idx[i])
        if len(indices) >= n:
            break
    return indices


def run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, mean, std,
                            batch_size, inference_steps, n_classes, label):
    diag_X = to_float_batch(X_val_u8[diag_indices], mean, std)
    diag_true = y_val_idx[diag_indices]

    pad_n = batch_size - len(diag_X)
    if pad_n > 0:
        filler = to_float_batch(X_val_u8[len(diag_indices):len(diag_indices) + pad_n], mean, std)
        diag_X_padded = np.vstack([diag_X, filler])
    else:
        diag_X_padded = diag_X[:batch_size]

    preds = net.predict(diag_X_padded, inference_steps).reshape(batch_size, n_classes)
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


def build_network(batch_size, n_classes, lr, ir, fl, lmbda, device,
                  use_momentum, use_ipc):
    """Deep (10 weight-bearing layer) conv PC network, built around the
    three techniques that this codebase has ACTUAL verified evidence for,
    not just plausible-sounding defaults:

      - ePC: a synthetic 8-hidden-layer convergence check in this repo
        (tests/tFullConvPCEPCConvergenceVerify.cpp) shows ePC reaching
        ~106x lower energy than plain one-hop settling at the SAME small
        step count. That's the entire point of using it here: at 10
        weight-bearing layers deep, plain settling would need far more
        inference_steps per call to propagate error across depth, which
        is both slow and the known failure mode ePC specifically targets.
      - Cross-entropy terminal: MSE-style regression against a one-hot
        target (what the old version of this script did, implicitly, by
        training without use_cross_entropy) is a poor loss for 200-way
        classification. This mirrors temp.py's own hard-won FullPCN
        config (see that file's extensive LR-sweep comments), which
        found use_cross_entropy=True essential for stable multi-epoch
        training on a one-hot classification target.
      - ADAMW for both the main and Psi (direct-feedback) optimizers,
        again mirroring temp.py's validated FullPCN recipe.
      - muPC scaling: NOT optional for this architecture, unlike the
        other toggles below. temp.py's own validated FullPCN recipe
        skipped it (that network was only 6 dense layers, shrinking
        width); this one is 10 conv layers with channels GROWING
        64->128->256. Tested directly (CPU, synthetic data, identical
        everything-else): without muPC scaling this architecture's
        energy explodes from O(10) to O(1e28) within 2 TrainStep() calls
        regardless of learning rate (tried 1e-5 through 1e-3, all
        exploded at the same step count) -- a signal-propagation-across-
        depth problem, not a learning-rate problem, which is exactly
        what muPC scaling exists to fix. With it on, identical setup
        trains stably for 60+ steps. Leave this on.

    Residual connections are the one technique deliberately NOT part of
    the default recipe: FullConvPCNetwork::Compile() REQUIRES every
    hidden layer from index 2 to H-1 (in the maintainer's 1-indexed
    terminology) to have IDENTICAL in/out channels and spatial size, or
    it throws -- there's no per-layer opt-out, it's all-or-nothing across
    that whole middle stretch. A real multi-stage pyramid (channels
    growing, spatial shrinking across several stride-2 transitions, which
    is what actually works well for image classification) is
    fundamentally incompatible with that constraint. temp.py hit the
    exact same wall for its own (dense, shrinking-width) architecture and
    made the same call: residual off, documented why. See the per-stage
    comments in the architecture below for exactly where this would
    throw if enabled.

    use_momentum/use_ipc are exposed as parameters specifically so you
    can flip them on for your own experiments without editing the
    architecture -- both individually verified not to crash in
    combination with the rest of this recipe
    (tests/tFullConvPCOtherTogglesSanity.cpp), just not part of the
    default since temp.py's one validated reference didn't use them
    either.
    """
    net = dy.FullConvPCNetwork(batch_size=batch_size, device=device)

    def conv(in_c, out_c, hw, k, stride, pad, activation="relu", activation_deriv="drelu"):
        net.add_layer(in_c, out_c, hw, hw, k, k, stride_h=stride, stride_w=stride,
                     pad_h=pad, pad_w=pad, terminal_size=n_classes,
                     lr=lr, ir=ir, fl=fl, lmbda=lmbda,
                     activation=activation, activation_deriv=activation_deriv)

    # Stem: a single, larger-kernel, stride-2 layer handling the raw
    # 64x64x3 input (standard practice: use a bigger receptive field only
    # where the input is still raw pixels, 3x3 everywhere deeper in).
    # 64x64 -> 32x32x64. This is hidden layer l=1 in the residual
    # terminology above -- the one index that IS exempt from the
    # matching-shape requirement, were residual ever turned on.
    conv(N_CHANNELS, 64, IMG_SIZE, k=5, stride=2, pad=2)

    # Stage 1 @ 32x32, 64 channels: two same-shape layers. (These two ARE
    # each other's shape-match if residual were enabled -- it's the
    # FOLLOWING stride-2 transition that would immediately throw, since
    # it changes both channel count and spatial size mid-stretch.)
    conv(64, 64, 32, k=3, stride=1, pad=1)
    conv(64, 64, 32, k=3, stride=1, pad=1)

    # Downsample 1: 32x32 -> 16x16, 64 -> 128 channels.
    conv(64, 128, 32, k=3, stride=2, pad=1)

    # Stage 2 @ 16x16, 128 channels.
    conv(128, 128, 16, k=3, stride=1, pad=1)
    conv(128, 128, 16, k=3, stride=1, pad=1)

    # Downsample 2: 16x16 -> 8x8, 128 -> 256 channels.
    conv(128, 256, 16, k=3, stride=2, pad=1)

    # Stage 3 @ 8x8, 256 channels.
    conv(256, 256, 8, k=3, stride=1, pad=1)
    conv(256, 256, 8, k=3, stride=1, pad=1)

    # Classifier: a full-size (8x8) kernel collapses the whole spatial
    # extent to 1x1 in one GEMM (same trick the previous version of this
    # script used for its own last layer), producing one score per class.
    # TanH here, not Linear, feeding into cross-entropy: this exactly
    # mirrors temp.py's own validated choice (its last Linear() was
    # followed by TanH() before the auto-appended terminal, under the
    # same use_cross_entropy=True), rather than assuming the conventional
    # "raw linear logits into softmax" convention transfers unchanged to
    # this framework's internals.
    conv(256, n_classes, 8, k=8, stride=1, pad=0, activation="tanh", activation_deriv="dtanh")

    # Terminal sink: outChannels=0, matching every other *PCLayer family
    # in this codebase's "pure sink, no further projection" convention.
    net.add_layer(n_classes, 0, 1, 1, 1, 1, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                 terminal_size=n_classes, lr=lr, ir=ir, fl=fl, lmbda=lmbda,
                 activation="linear", activation_deriv="dlinear")

    # Must be set before compile(): both are applied structurally, inside
    # Compile() itself (see FullConvPCNetwork::Compile()'s own source).
    # muPC scaling is NOT a toggle (see docstring: required for stability
    # at this depth, not just an available option).
    net.set_use_mu_pc_scaling(True)
    net.set_use_residual_connections(False)  # see build_network's docstring
    net.compile()

    # Everything else is a runtime flag, read fresh at TrainStep() time --
    # set after compile(), matching tFullConvPCOtherTogglesSanity.cpp's
    # own verified ordering exactly.
    net.set_use_ipc(use_ipc)
    net.set_use_epc(True)
    net.set_use_momentum(use_momentum, 0.9)
    net.set_use_cross_entropy(True)
    net.set_optimizer("ADAMW")
    net.set_psi_optimizer("ADAMW")

    return net


def main() -> None:
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    INFERENCE_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    BATCH_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 250
    SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    # Default raised from temp.py's inherited 1e-4 to 1.0 after the first
    # two real GPU runs: epoch 1 trained cleanly (energy steady in
    # 2300-3600), but epoch 2 blew up exponentially over ~50 batches
    # (8596 -> 8.7e25 -> non-finite) at BOTH lr=1e-5 and lr=1e-6 -- same
    # failure, just slower at the smaller lr, not avoided by it. Why lr
    # alone can't fix this: AdamWStep's decay term is `lr * lmbda *
    # param[i]`, and its gradient term is `(lr * sqrt(beta2_t)/beta1_t) *
    # m[i]/(sqrt(v[i])+eps)` -- both scale with lr identically, so their
    # RATIO (what actually determines whether weights grow or shrink)
    # depends only on lmbda, never on lr. Adam's normalization keeps
    # m[i]/sqrt(v[i]) around order-1 magnitude, so the gradient term's
    # own magnitude is roughly `lr` regardless of current weight size,
    # while the decay term is `lr * lmbda * |param|` -- for decay to
    # meaningfully compete with an order-1 gradient push, you need
    # roughly lmbda * |param| ~ 1, i.e. lmbda ~ 1/|param|. This
    # architecture's He/Kaiming-style init (limit = sqrt(2/fan_in))
    # produces weights around 0.01-0.1, which needs lmbda roughly in the
    # 10-100 range, not 1e-4 (off by 5-6 orders of magnitude -- decay was
    # providing essentially zero counter-pressure against growth at any
    # lr). 1.0 is a deliberately moderate first step in that direction,
    # not the final answer -- CLI-overridable (6th arg) specifically so
    # finding the right value doesn't need another script edit.
    LMBDA = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

    # --- Experimental toggles -----------------------------------------
    # muPC scaling is NOT here -- it's required for stability at this
    # depth and always on inside build_network(), see its docstring.
    # These two are both individually verified not to crash in
    # combination with the rest of this recipe
    # (tests/tFullConvPCOtherTogglesSanity.cpp), but aren't part of the
    # one combination this repo has ACTUAL tuned evidence for (temp.py's
    # FullPCN result). Flip these on if you want to explore beyond that
    # known-reasonable starting point.
    USE_MOMENTUM = False
    USE_IPC = False

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' not found in the current directory -- "
            f"run this script from wherever you unzipped tiny-imagenet-200.zip."
        )

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    mean, std = compute_normalization_stats(X_train_u8, seed=SEED)
    print(f"Per-channel normalization (computed once, from a 2,000-image training "
          f"sample): mean={mean}, std={std}")

    IR = 0.15
    FL = 1e-3
    DECAY_RATE = 0.94

    print(f"\nBuilding FullConvPCNetwork (10 weight-bearing conv layers, cross-entropy + "
          f"ePC + ADAMW): 3x64x64 -> 64x32x32 -> 64x32x32 -> 128x16x16 -> 128x16x16 -> "
          f"256x8x8 -> 256x8x8 -> {N_CLASSES}x1x1")
    net = build_network(BATCH_SIZE, N_CLASSES, LR, IR, FL, LMBDA, device="gpu",
                        use_momentum=USE_MOMENTUM, use_ipc=USE_IPC)
    net.randomize_weights(SEED)

    print(f"\n*** TINY IMAGENET, FullConvPCNetwork (cross-entropy, ePC) ***")
    print(f"Training: {EPOCHS} epochs, inference_steps={INFERENCE_STEPS}, lr={LR}, "
          f"lmbda={LMBDA}, batch_size={BATCH_SIZE}, seed={SEED}")
    print(f"NOTE: this LR is a reasoned-but-unvalidated starting point (see "
          f"build_network()'s docstring) -- no GPU was available to tune it while "
          f"writing this. Watch the first several epochs' energy the same way temp.py's "
          f"own comments describe: monotonically decreasing is healthy, bottoming out "
          f"then rising again is the early warning sign of the LR being too high.\n")

    # Isolating whether predict() itself is what destabilizes training at
    # the epoch boundary (two straight runs blew up exponentially a few
    # dozen batches into epoch 2, right after this diagnostic and the
    # per-epoch accuracy eval below -- both are predict() calls),
    # independent of lr/lmbda (neither changed WHETHER it happens, just
    # how fast once triggered). SKIP_PREDICT=1 removes every predict()
    # call so training runs continuously with none of them in the way;
    # if it blows up at roughly the same batch count anyway, predict()
    # was never the trigger, just a correlated bystander.
    skip_predict = os.environ.get("SKIP_PREDICT", "0") == "1"
    if skip_predict:
        print("SKIP_PREDICT=1: skipping the PRE-training diagnostic and all per-epoch "
              "accuracy evals -- training runs continuously with zero predict() calls.")

    diag_indices = pick_diagnostic_indices(y_val_idx)
    if not skip_predict:
        run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, mean, std,
                                BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "PRE-training")

    rng = np.random.default_rng(SEED)
    n_train = len(X_train_u8)
    n_batches = n_train // BATCH_SIZE
    start_time = perf_counter()

    for epoch in range(EPOCHS):
        # Decaying lr/ir/fl together, not just lr: temp.py's own
        # comments document exactly this failure mode for a different
        # (dense) deep cross-entropy+ePC network -- ir undecayed while
        # lr/fl shrink creates a growing settling/learning speed mismatch
        # that eventually destabilizes the energy landscape late in
        # training. Decaying all three together is the fix that was
        # found to work there; applying it from the start here rather
        # than discovering the same instability again.
        current_lr = LR * (DECAY_RATE ** epoch)
        current_ir = IR * (DECAY_RATE ** epoch)
        current_fl = FL * (DECAY_RATE ** epoch)
        net.set_learning_rate(current_lr)
        net.set_feedback_rate(current_fl)
        # No network-level set_inference_rate() exists for
        # FullConvPCNetwork (unlike set_learning_rate/set_feedback_rate,
        # which loop over every layer internally -- see
        # FullConvPCNetwork.h), so this loop does it by hand.
        for layer in net.layers:
            layer.set_inference_rate(current_ir)

        indices = rng.permutation(n_train)
        epoch_energy = 0.0

        for b in range(n_batches):
            batch_idx = indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            X_batch = to_float_batch(X_train_u8[batch_idx], mean, std)
            Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES)

            energy = net.train_step(X_batch, Y_batch, INFERENCE_STEPS)
            # Checked per-batch, not just once at the epoch's end: a
            # single non-finite batch poisons every later += (NaN
            # propagates through addition), so checking only the
            # epoch-end average can't tell you WHICH batch first broke --
            # it could be batch 0 or batch 399. Catching it here instead
            # pinpoints the exact batch, which matters for diagnosing why.
            if not np.isfinite(energy):
                print(f"Epoch {epoch+1}, batch {b}/{n_batches}: NON-FINITE ENERGY -- STOPPING "
                      f"(previous batch's energy was {epoch_energy / b if b > 0 else 'N/A (first batch of epoch)'})")
                return
            epoch_energy += energy
            if b % 50 == 0:
                print(f"  epoch {epoch+1}, batch {b}/{n_batches}: energy={energy:.4f}")

        avg_energy = epoch_energy / n_batches

        epoch_acc = float("nan")
        if not skip_predict:
            N_ACC_BATCHES = 20
            correct, total = 0, 0
            for b in range(min(N_ACC_BATCHES, len(X_val_u8) // BATCH_SIZE)):
                X_batch = to_float_batch(X_val_u8[b * BATCH_SIZE:(b + 1) * BATCH_SIZE], mean, std)
                y_batch = y_val_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

                preds = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, N_CLASSES)
                pred_classes = np.argmax(preds, axis=1)
                correct += np.sum(pred_classes == y_batch)
                total += BATCH_SIZE
            epoch_acc = 100.0 * correct / total

        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | "
              f"Avg energy: {avg_energy:.4f} | lr={current_lr:.2e} ir={current_ir:.4f} "
              f"fl={current_fl:.2e}")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    if skip_predict:
        print("SKIP_PREDICT=1: skipping final validation accuracy and the POST-training "
              "diagnostic (both are predict() calls).")
        return

    correct, total = 0, 0
    for i in range(0, len(X_val_u8), BATCH_SIZE):
        X_batch_u8 = X_val_u8[i:i + BATCH_SIZE]
        y_batch = y_val_idx[i:i + BATCH_SIZE]
        if len(X_batch_u8) != BATCH_SIZE:
            continue
        X_batch = to_float_batch(X_batch_u8, mean, std)
        preds = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, N_CLASSES)
        pred_classes = np.argmax(preds, axis=1)
        correct += np.sum(pred_classes == y_batch)
        total += BATCH_SIZE

    val_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"FullConvPCNetwork Tiny ImageNet validation accuracy: {val_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")

    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, mean, std,
                            BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "POST-training")


if __name__ == "__main__":
    main()
