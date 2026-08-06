import numpy as np
import numpy.typing as npt
from typing import Optional
from . import deepity as dy

class SequentialPCN(dy.DiscriminativePCNetwork):
    """
    A Sequential Predictive Coding Network (PCN) wrapper for the Deepity C++ backend.
    
    This class orchestrates the continuous-time inference and local Hebbian learning 
    phases of the network, delegating the heavy computational lifting to optimized 
    C++ SIMD micro-kernels.
    """

    def __init__(self, batch_size: Optional[int] = None) -> None:
        """
        Initializes the Predictive Coding Network.

        Args:
            batch_size (int, optional): The fixed batch size for training/inference. 
                If None, the backend will calculate the optimal batch size based on 
                the host OS and L2 cache availability to maximize OpenMP thread utilization.
        """
        bsz = dy.auto_batch_size() if batch_size is None else batch_size
        super().__init__(bsz)
    
    def add_layer(
        self, 
        in_features: int, 
        out_features: int, 
        lr: float, 
        ir: float, 
        pr: float, 
        act: str, 
        lmbda: float = 0.0001
    ) -> None:
        """
        Appends a new DiscriminativePCLayer to the top of the network hierarchy.

        Args:
            in_features (int): Dimensionality of the input (bottom-up) states.
            out_features (int): Dimensionality of the output (top-down) predictions.
            lr (float): The base learning rate for synaptic weight updates.
            ir (float): The inference rate (Euler integration step size) for state relaxation.
            pr (float): The precision rate for tracking inverse variance (uncertainty).
            act (str): The activation function string identifier (e.g., 'tanh', 'relu').
            lmbda (float, optional): L2 regularization/weight decay penalty. Defaults to 0.0001.
        """
        super().add_layer(
            in_features, 
            out_features, 
            lr=lr, 
            ir=ir, 
            pr=pr,
            lmbda=lmbda, 
            activation=act, 
            activation_deriv='d' + act
        )

    def randomize_weights(self) -> None:
        """
        Populates all synaptic weight matrices across the network with 
        normally distributed random values scaled by the layer dimensions.
        """
        super().randomize_weights()

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        """
        Executes a full forward initialization, state relaxation, and weight update cycle.

        During state relaxation, both boundary layers (input and target) are clamped, 
        allowing the hidden layers to perform gradient descent on free energy.

        Args:
            X (npt.NDArray[np.float32]): The flattened batched input data. 
                Example: For MNIST, this should be sized (batch_size * 784).
            Y (npt.NDArray[np.float32]): The flattened batched target labels.
                Example: For MNIST, this should be sized (batch_size * 10).
            steps (int): The number of continuous-time integration steps to run 
                before allowing the weights to update.

        Returns:
            float: The total network free energy accumulated over the inference steps.
        """
        self.reset_state()

        self.clamp_input(X)
        self[-1].clamp_state(Y)

        total_energy: float = 0.0

        for _ in range(steps):
            total_energy += self.calculate_state()
            self.update_state()

        self.update_weights()
        self.update_precision()
        self[-1].unclamp_state()
        
        return total_energy
    
    def set_learning_rate(self, lr: float) -> None:
        """
        Dynamically adjusts the base learning rate across all layers in the network.

        Args:
            lr (float): The new global learning rate to apply.
        """
        super().set_learning_rate(lr)

    def compile(self) -> None:
        """
        Calculates the exact footprint of the network and binds all layer states, 
        weights, and biases to a single, contiguous C++ MemoryArena block.
        
        Warning: 
            This must be called BEFORE randomize_weights() to ensure memory 
            pointers are correctly locked in prior to initialization.
        """
        super().compile()
    
    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        """
        Runs generative/discriminative inference on the provided input data.

        Unlike train_step, the terminal layer is left unclamped, allowing the 
        network's internal dynamics to push bottom-up signals until the top 
        layer settles into a predicted state.

        Args:
            X (npt.NDArray[np.float32]): The flattened input batch.
            steps (int): The number of Euler integration steps to allow the 
                network to settle.

        Returns:
            npt.NDArray[np.float32]: The predicted states (beliefs) of the terminal 
                layer, reshaped to (batch_size, num_classes).
        """
        self.reset_state()
        self.clamp_input(X.flatten())

        for _ in range(steps):
            self.calculate_state()
            self.update_state()

        return np.array(self[-1].beliefs)

    def save(self, dir_path: str) -> bool:
        """
        Serializes the network into a highly-optimized directory structure.
        
        Creates 'manifest.json' for architecture metadata, 'README.md' for 
        human-readable summaries, and 'weights.bin' for the raw memory payload.

        Args:
            dir_path (str): The destination directory (will be created if it doesn't exist).

        Returns:
            bool: True if serialization was successful, False otherwise.
        """
        return super().save(dir_path)

    def load(self, dir_path: str) -> bool:
        """
        Loads the network parameters directly into the contiguous MemoryArena.
        
        Warning: 
            The network must be instantiated with the identical layer topology 
            and must be compiled via `net.compile()` BEFORE calling load().

        Args:
            dir_path (str): The source directory containing the model files.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        return super().load(dir_path)