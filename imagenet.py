import numpy as np
import os
import sys
from time import perf_counter
from pydeepity import dy
from PIL import Image

IMG_SIZE = 64
N_CHANNELS = 3
DATA_DIR = "tiny-imagenet-200"


def load_wnids(data_dir):
    with open(os.path.join(data_dir, "wnids.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def load_image_as_chw(path):
    # PIL gives (H,W,3); transposed to (3,H,W) -- channels-first,
    # matching this codebase's conv layer convention (see ConvPCLayer's
    # own beliefs shape: {batch, in_channels, in_height, in_width}).
    # Not directly confirmed against SimpleConvPCLayer's own forward
    # pass tonight -- worth checking directly if results look wrong.
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


def to_float_batch(X_uint8_batch):
    X = X_uint8_batch.astype(np.float32) / 255.0
    return (X - X.mean()) / (X.std() + 1e-8)


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


def run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices, batch_size,
                            inference_steps, n_classes, label):
    diag_X = to_float_batch(X_val_u8[diag_indices])
    diag_true = y_val_idx[diag_indices]

    pad_n = batch_size - len(diag_X)
    filler = to_float_batch(X_val_u8[len(diag_indices):len(diag_indices) + pad_n])
    diag_X_padded = np.vstack([diag_X, filler])

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


def main() -> None:
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    INFERENCE_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    LR = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-4

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' not found in the current directory -- "
            f"run this script from wherever you unzipped tiny-imagenet-200.zip."
        )

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set_cached(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set_cached(DATA_DIR, wnids)

    BATCH_SIZE = 250
    IR = 0.15
    LMBDA = 1e-4
    DECAY_RATE = 0.94

    print(f"\nBuilding conv network: 3x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8 -> {N_CLASSES}x1x1")

    net = dy.SimpleConvPCNetwork(batch_size=BATCH_SIZE, device="gpu")
    net.add_layer(3, 32, IMG_SIZE, IMG_SIZE, 5, 5, stride_h=2, stride_w=2, pad_h=2, pad_w=2,
                  lr=LR, ir=IR, lmbda=LMBDA, activation="relu", activation_deriv="drelu")
    net.add_layer(32, 64, 32, 32, 5, 5, stride_h=2, stride_w=2, pad_h=2, pad_w=2,
                  lr=LR, ir=IR, lmbda=LMBDA, activation="relu", activation_deriv="drelu")
    net.add_layer(64, 128, 16, 16, 5, 5, stride_h=2, stride_w=2, pad_h=2, pad_w=2,
                  lr=LR, ir=IR, lmbda=LMBDA, activation="relu", activation_deriv="drelu")
    net.add_layer(128, N_CLASSES, 8, 8, 8, 8, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                  lr=LR, ir=IR, lmbda=LMBDA, activation="linear", activation_deriv="dlinear")

    net.set_optimizer("ADAM")
    net.compile()
    net.randomize_weights()  # no seed param on this class -- not reproducible run-to-run

    print(f"\n*** TINY IMAGENET, SimpleConvPCNetwork ***")
    print(f"Training: {EPOCHS} epochs, inference_steps={INFERENCE_STEPS}, lr={LR}\n")

    diag_indices = pick_diagnostic_indices(y_val_idx)
    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices,
                            BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "PRE-training")

    rng = np.random.default_rng(7)
    n_train = len(X_train_u8)
    n_batches = n_train // BATCH_SIZE
    start_time = perf_counter()

    for epoch in range(EPOCHS):
        indices = rng.permutation(n_train)
        epoch_energy = 0.0

        for b in range(n_batches):
            batch_idx = indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            X_batch = to_float_batch(X_train_u8[batch_idx])
            Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES)

            energy = net.train_step(X_batch, Y_batch, INFERENCE_STEPS)
            epoch_energy += energy

        avg_energy = epoch_energy / n_batches
        if not np.isfinite(avg_energy):
            print(f"Epoch {epoch+1}: NON-FINITE ENERGY -- STOPPING")
            return

        N_ACC_BATCHES = 10
        correct, total = 0, 0
        for b in range(min(N_ACC_BATCHES, len(X_val_u8) // BATCH_SIZE)):
            X_batch = to_float_batch(X_val_u8[b * BATCH_SIZE:(b + 1) * BATCH_SIZE])
            y_batch = y_val_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            preds = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, N_CLASSES)
            pred_classes = np.argmax(preds, axis=1)
            correct += np.sum(pred_classes == y_batch)
            total += BATCH_SIZE

        epoch_acc = 100.0 * correct / total
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | Avg energy: {avg_energy:.4f}")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    correct, total = 0, 0
    for i in range(0, len(X_val_u8), BATCH_SIZE):
        X_batch_u8 = X_val_u8[i:i + BATCH_SIZE]
        y_batch = y_val_idx[i:i + BATCH_SIZE]
        if len(X_batch_u8) != BATCH_SIZE:
            continue
        X_batch = to_float_batch(X_batch_u8)
        preds = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, N_CLASSES)
        pred_classes = np.argmax(preds, axis=1)
        correct += np.sum(pred_classes == y_batch)
        total += BATCH_SIZE

    val_acc = 100.0 * correct / total
    print(f"\n=== Result ===")
    print(f"SimpleConvPCNetwork Tiny ImageNet validation accuracy: {val_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")

    run_collapse_diagnostic(net, X_val_u8, y_val_idx, diag_indices,
                            BATCH_SIZE, INFERENCE_STEPS, N_CLASSES, "POST-training")


if __name__ == "__main__":
    main()