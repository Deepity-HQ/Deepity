"""Architecture building blocks: pass these positionally to a network's
constructor to describe its layer stack (see each network class's own
docs for exact ordering rules)."""

from abc import ABC, abstractmethod


class Layer(ABC):
    """Base class for network layers."""

    @abstractmethod
    def to_string(self) -> str:
        ...


class Linear(Layer):
    """A weighted layer of shape (in_n, out_n)."""

    def __init__(self, in_n: int, out_n: int):
        self.in_n = in_n
        self.out_n = out_n

    def to_string(self) -> str:
        return "linear"

class Convolution(Layer):
    """A 2D convolutional layer."""

    def __init__(
        self,
        out_channels: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0,
    ):
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w

    def to_string(self) -> str:
        return "convolution"

class Activation(ABC):
    """Base class for activation functions. Placed after a Linear/
    Convolution layer to describe which activation it uses."""

    @abstractmethod
    def to_string(self) -> str:
        ...


class Sigmoid(Activation):
    """Logistic sigmoid activation."""

    def to_string(self) -> str:
        return "sigmoid"


class TanH(Activation):
    """Hyperbolic tangent activation."""

    def to_string(self) -> str:
        return "tanh"


class ReLU(Activation):
    """Rectified linear unit activation."""

    def to_string(self) -> str:
        return "relu"


class GeLU(Activation):
    """Gaussian error linear unit activation (tanh approximation)."""

    def to_string(self) -> str:
        return "gelu"
