from abc import ABC, abstractmethod


class Layer(ABC):
    """Base class for network layers."""

    @abstractmethod
    def to_string(self) -> str:
        ...


class Linear(Layer):
    def __init__(self, in_n: int, out_n: int):
        self.in_n = in_n
        self.out_n = out_n

    def to_string(self) -> str:
        return "linear"

class Convolution(Layer):
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
    """Base class for activation functions."""

    @abstractmethod
    def to_string(self) -> str:
        ...


class Sigmoid(Activation):
    def to_string(self) -> str:
        return "sigmoid"


class TanH(Activation):
    def to_string(self) -> str:
        return "tanh"


class ReLU(Activation):
    def to_string(self) -> str:
        return "relu"


class GeLU(Activation):
    def to_string(self) -> str:
        return "gelu"
