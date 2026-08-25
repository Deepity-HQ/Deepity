from ._backend import dy
from typing import Optional
import numpy as np
import numpy.typing as npt
from .utils import _PCNMixin

class SequentialPCN(dy.DiscriminativePCNetwork, _PCNMixin):
    """
    A Sequential Predictive Coding Network wrapper for the Deepity C++ backend.
    """
    def __init__(self, batch_size: Optional[int] = None) -> None:
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
        super().add_layer(
            in_features, out_features, lr=lr, ir=ir, pr=pr,
            lmbda=lmbda, activation=act, activation_deriv='d' + act
        )

    def set_learning_rate(self, lr: float) -> None:
        super().set_learning_rate(lr)

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        """Executes a full forward clamp, state relaxation, and weight update cycle."""
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

    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        """Runs generative/discriminative inference on the provided input data."""
        self.reset_state()
        self.clamp_input(X.flatten())

        for _ in range(steps):
            self.calculate_state()
            self.update_state()

        return np.array(self[-1].beliefs)

    def save(self, dir_path: str) -> bool:
        return super().save(dir_path)

    def load(self, dir_path: str) -> bool:
        return super().load(dir_path)

