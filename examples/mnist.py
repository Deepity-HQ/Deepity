from pydeepity import SequentialPCN
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# Hyperparameters
# ----------------------------

BATCH = 256
LR = 5e-4
IR = 0.05
PR = 0.0         # Disabled adaptive precision to prevent energy collapse
LAMBDA = 1e-4

EPOCHS = 50
INFERENCE_STEPS = 30

# ----------------------------
# Dataset
# ----------------------------

# Zero-center the data to [-1.0, 1.0] to prevent tanh saturation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transform,
)

test = datasets.MNIST(
    "./data",
    train=False,
    transform=transform,
)

trainloader = DataLoader(
    train,
    batch_size=BATCH,
    shuffle=True,
    drop_last=True,
)

testloader = DataLoader(
    test,
    batch_size=BATCH,
    shuffle=False,
    drop_last=True,
)

# ----------------------------
# Network
# ----------------------------

# SequentialPCN wraps the raw DiscriminativePCNetwork backend and exposes
# the friendlier act= API (auto-derives the activation_deriv string), plus
# train_step()/predict() convenience methods -- it's the public class this
# architecture is meant to be built with, not the raw backend directly.
net = SequentialPCN(batch_size=BATCH)

net.add_layer(784, 512, lr=LR, ir=IR, pr=PR, act="tanh", lmbda=LAMBDA)
net.add_layer(512, 256, lr=LR, ir=IR, pr=PR, act="tanh", lmbda=LAMBDA)
net.add_layer(256, 10, lr=LR, ir=IR, pr=PR, act="tanh", lmbda=LAMBDA)

# Terminal layer: next_size=0 marks this as the network's output/terminal layer
net.add_layer(10, 0, lr=LR, ir=IR, pr=PR, act="linear", lmbda=LAMBDA)

# Compile BEFORE randomize_weights()/training -- this allocates the
# contiguous weight arena every layer's beliefs/errors/weights views are
# backed by. Skipping it (as the original script did) touches uninitialized
# memory.
net.compile()
net.randomize_weights()

# ----------------------------
# Utilities
# ----------------------------

def one_hot(labels):
    # Scale targets to [-0.9, 0.9] to match the tanh output bounds
    y = np.full((len(labels), 10), -0.9, dtype=np.float32)
    y[np.arange(len(labels)), labels] = 0.9
    return y

# ----------------------------
# Training
# ----------------------------

for epoch in range(EPOCHS):

    energy_sum = 0.0

    for images, labels in trainloader:

        x = images.numpy().reshape(BATCH, -1).astype(np.float32)
        y = one_hot(labels.numpy())

        # SequentialPCN.train_step() handles reset_state/clamp_input/
        # clamp_state/settle/update_weights/update_precision/unclamp_state
        # internally in one call.
        energy = net.train_step(x, y, steps=INFERENCE_STEPS)

        energy_sum += energy

    print(
        epoch,
        energy_sum / len(trainloader)
    )
