"""
Tiny ImageNet-200 training for DKPPCN.

v2: deeper/wider network (~63M params, up from ~15M), matching the
literature's suggestion that published, working PCN results on Tiny
ImageNet use substantially larger networks (PCX reports up to
AlexNet-scale, ~160M params) than a naive MNIST-scale architecture.
LR/FL deliberately held at the confirmed-stable 5e-5 from the previous
run, so this test isolates architecture size as the one changed
variable -- if energy stays stable and accuracy improves, that confirms
size was the real gap; if energy destabilizes again at the same LR,
that's equally informative (suggests LR needs to scale down further as
network size grows, a documented phenomenon in this literature).
"""
import numpy as np
import os
import sys
from time import perf_counter
from pydeepity import DKPPCN
from PIL import Image

IMG_SIZE = 64
IMG_DIM = IMG_SIZE * IMG_SIZE * 3
DATA_DIR = "tiny-imagenet-200"


def load_wnids(data_dir):
    with open(os.path.join(data_dir, "wnids.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def load_image_as_vector(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8).reshape(-1)


def load_train_set(data_dir, wnids):
    print("Loading Tiny ImageNet training set (100,000 images)...")
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    X = np.zeros((len(wnids) * 500, IMG_DIM), dtype=np.uint8)
    y_idx = np.zeros(len(wnids) * 500, dtype=np.int64)

    pos = 0
    for wnid in wnids:
        img_dir = os.path.join(data_dir, "train", wnid, "images")
        filenames = sorted(os.listdir(img_dir))
        for fname in filenames:
            X[pos] = load_image_as_vector(os.path.join(img_dir, fname))
            y_idx[pos] = wnid_to_idx[wnid]
            pos += 1
        print(f"  loaded class {wnid} ({pos}/{len(wnids) * 500})", end="\r")
    print()

    return X[:pos], y_idx[:pos], len(wnids)


def load_val_set(data_dir, wnids):
    print("Loading Tiny ImageNet validation set (10,000 images)...")
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    annotations = {}
    with open(os.path.join(data_dir, "val", "val_annotations.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            annotations[parts[0]] = parts[1]

    img_dir = os.path.join(data_dir, "val", "images")
    filenames = sorted(annotations.keys())

    X = np.zeros((len(filenames), IMG_DIM), dtype=np.uint8)
    y_idx = np.zeros(len(filenames), dtype=np.int64)

    for i, fname in enumerate(filenames):
        X[i] = load_image_as_vector(os.path.join(img_dir, fname))
        y_idx[i] = wnid_to_idx[annotations[fname]]
        if i % 1000 == 0:
            print(f"  loaded {i}/{len(filenames)}", end="\r")
    print()

    return X, y_idx


def to_float_batch(X_uint8_batch):
    return X_uint8_batch.astype(np.float32) / 255.0


def to_one_hot(y_idx_batch, n_classes, eps=0.001):
    Y = np.full((len(y_idx_batch), n_classes), eps, dtype=np.float32)
    Y[np.arange(len(y_idx_batch)), y_idx_batch] = 1.0 - eps
    return Y


def main() -> None:
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    INFERENCE_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' not found in the current directory -- "
            f"run this script from wherever you unzipped tiny-imagenet-200.zip."
        )

    wnids = load_wnids(DATA_DIR)
    X_train_u8, y_train_idx, N_CLASSES = load_train_set(DATA_DIR, wnids)
    X_val_u8, y_val_idx = load_val_set(DATA_DIR, wnids)

    BATCH_SIZE = 250
    TERMINAL_SIZE = N_CLASSES

    HIDDEN_1 = 4096
    HIDDEN_2 = 2048

    LR = 5e-5
    IR = 0.15
    FL = 5e-5
    LMBDA = 1e-4
    DECAY_RATE = 0.94

    print(f"\nBuilding network ({IMG_DIM}->{HIDDEN_1}->{HIDDEN_2}->{TERMINAL_SIZE}), seed={SEED}...")
    net = DKPPCN(batch_size=BATCH_SIZE, device="gpu")
    net.add_layer(IMG_DIM, HIDDEN_1, TERMINAL_SIZE, lr=LR, ir=IR, fl=FL, lmbda=LMBDA, act="linear")
    net.add_layer(HIDDEN_1, HIDDEN_2, TERMINAL_SIZE, lr=LR, ir=IR, fl=FL, lmbda=LMBDA, act="sigmoid")
    net.add_layer(HIDDEN_2, TERMINAL_SIZE, TERMINAL_SIZE, lr=LR, ir=IR, fl=FL, lmbda=LMBDA, act="sigmoid")
    net.add_layer(TERMINAL_SIZE, 0, TERMINAL_SIZE, lr=LR, ir=IR, fl=FL, lmbda=LMBDA, act="linear")
    net.set_optimizer("ADAM")
    net.set_psi_optimizer("ADAM")
    net.compile()
    net.randomize_weights()

    total_params = IMG_DIM * HIDDEN_1 + HIDDEN_1 * HIDDEN_2 + HIDDEN_2 * TERMINAL_SIZE
    total_params += (IMG_DIM + HIDDEN_1 + HIDDEN_2) * TERMINAL_SIZE
    print(f"Approx. total parameters (W + Psi): {total_params:,}")

    print(f"\n*** TINY IMAGENET DKP-PC RUN (v2: bigger network) ***")
    print(f"Training DKPPCN: {EPOCHS} epochs, inference_steps={INFERENCE_STEPS}, ")
    print(f"lr={LR}, ir={IR}, fl={FL}, lmbda={LMBDA}, decay_rate={DECAY_RATE}")
    print(f"NOTE: LR/FL held fixed from the previous, confirmed-stable run --")
    print(f"      this test isolates architecture size as the only changed variable.\n")

    rng = np.random.default_rng(SEED)
    n_train = len(X_train_u8)
    n_batches = n_train // BATCH_SIZE
    start_time = perf_counter()
    epoch_accs = []

    for epoch in range(EPOCHS):
        current_lr = LR * (DECAY_RATE ** epoch)
        net.set_learning_rate(current_lr)
        current_fl = FL * (DECAY_RATE ** epoch)
        net.set_feedback_rate(current_fl)

        indices = rng.permutation(n_train)
        epoch_energy = 0.0

        for b in range(n_batches):
            batch_idx = indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            X_batch = to_float_batch(X_train_u8[batch_idx])
            Y_batch = to_one_hot(y_train_idx[batch_idx], N_CLASSES)

            energy = net.train_step(X_batch, Y_batch, INFERENCE_STEPS)
            epoch_energy += energy

        N_ACC_BATCHES = 10
        correct = 0
        total = 0
        for b in range(min(N_ACC_BATCHES, len(X_val_u8) // BATCH_SIZE)):
            X_batch = to_float_batch(X_val_u8[b * BATCH_SIZE:(b + 1) * BATCH_SIZE])
            y_batch = y_val_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            preds = net.predict(X_batch, INFERENCE_STEPS).reshape(BATCH_SIZE, N_CLASSES)
            pred_classes = np.argmax(preds, axis=1)
            correct += np.sum(pred_classes == y_batch)
            total += BATCH_SIZE

        epoch_acc = 100.0 * correct / total
        epoch_accs.append(epoch_acc)
        avg_energy = epoch_energy / n_batches
        elapsed = perf_counter() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}s | Acc: {epoch_acc:.2f}% | Avg energy: {avg_energy:.4f}")

    train_time = perf_counter() - start_time
    print(f"\nTraining complete in {train_time:.1f}s.")

    print("\nRunning final validation evaluation (full 10,000 images)...")
    correct = 0
    total = 0
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
    print(f"DKPPCN Tiny ImageNet validation accuracy: {val_acc:.2f}%")
    print(f"Train time: {train_time:.1f}s")


if __name__ == "__main__":
    main()