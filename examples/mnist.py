from pydeepity import deepity  # pyright: ignore[reportAttributeAccessIssue] - raw C++ extension, no static type info available
import numpy as np
import torch
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
)

# ----------------------------
# Network
# ----------------------------

net = deepity.DiscriminativePCNetwork(BATCH)

net.add_layer(
    784, 512,
    lr=LR,
    ir=IR,
    pr=PR,
    lmbda=LAMBDA,
    activation="tanh",
    activation_deriv="dtanh"
)

net.add_layer(
    512, 256,
    lr=LR,
    ir=IR,
    pr=PR,
    lmbda=LAMBDA,
    activation="tanh",
    activation_deriv="dtanh"
)

# Changed to tanh to bound predictions before the terminal layer
net.add_layer(
    256, 10,
    lr=LR,
    ir=IR,
    pr=PR,
    lmbda=LAMBDA,
    activation="tanh",
    activation_deriv="dtanh"
)

net.add_layer(
    10, 0,
    lr=LR,
    ir=IR,
    pr=PR,
    lmbda=LAMBDA,
    activation="linear",
    activation_deriv="dlinear"
)

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

        net.reset_state()

        net.clamp_input(x)

        terminal = net.get_terminal_layer()
        terminal.clamp_state(y)

        energy = 0.0
        for _ in range(INFERENCE_STEPS):
            energy = net.calculate_state()
            net.update_state()

        terminal.unclamp_state()

        net.update_weights()
        net.update_precision()

        energy_sum += energy

    print(
        epoch,
        energy_sum / len(trainloader)
    )