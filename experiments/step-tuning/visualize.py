import numpy as np
import deepity as deep
import matplotlib.pyplot as plt

# Load the network
net = deep.PCNetwork()
# Architecture must match the training script exactly
net.add_layer(784, 256, lr=1e-4, ir=1e-3, step_size=20, act="relu")
net.add_layer(256, 64,  lr=1e-4, ir=1e-3, step_size=20, act="relu")
net.compile()

# Load the weights
net.load("mnist_pc_model.bin")

# To generate, we usually pass in a 'target' to the top layer
# or use generation_step to propagate top-down.
# Here we simulate starting from the top layer (64 neurons)
# and generating back down to the 784 pixels.

# Get the last layer to generate
# NOTE: This assumes your binding for getting layers works or 
# you have a mechanism to trigger generation from the top.
# For demo purposes, we will feed a random latent vector to generation_step.

print("Generating digit from latent space...")
latent_sample = np.random.normal(0, 1, 64).astype(np.float32)

# Use the library's generation_step
net.generation_step(latent_sample)
net.flush_generation()

# How to retrieve the generated image depends on your C++ binding
# Typically you would grab the output of the final layer 
# or the reconstruction from the first layer.
# Assuming you can get the reconstruction:
# reconstruction = net.get_reconstruction() 

print("Generation process triggered.")
print("Note: Ensure your C++ implementation supports generation_step data extraction.")