import deepity as deep
import numpy as np

net = deep.PCNetwork()
net.add_layer(784, 512, 1e-6, "relu", "drelu")
net.add_layer(512, 256, 1e-6, "relu", "drelu")
net.add_layer(256, 64, 1e-6, "relu", "drelu")
net.add_layer(64, 10, 1e-6, "relu", "drelu")
net.randomize_weights()