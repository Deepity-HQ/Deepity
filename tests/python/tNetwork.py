import deepity as deep
import numpy as np

net = deep.PCNetwork()
net.add_layer(784, 256, lr=1e-4, ir=1e-4, step_size=30, act="tanh")
net.add_layer(256, 64)
net.add_layer(64, 10)
net.compile()

# Feed an input bottom-up
x = np.random.uniform(0.0, 1.0, 784).astype(np.float32)
net.inference_step(x)
net.flush_inference()

print("Energy:", net.total_energy())
net.save("model.bin")