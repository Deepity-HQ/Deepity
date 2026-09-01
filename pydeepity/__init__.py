__version__ = "1.0.0"

from .SequentialPCN import SequentialPCN
from .SimplePCN import SimplePCN
from .ConvolutionalPCN import ConvolutionalPCN
from .SimpleConvolutionalPCN import SimpleConvolutionalPCN
from .GaussSeidelPCN import GaussSeidelPCN
from .DKPPCN import DKPPCN
from ._backend import dy

StreamAlignedBatcher = dy.StreamAlignedBatcher

# Hardware & Threading Utilities
get_l2_cache_bytes = dy.get_l2_cache_bytes
auto_batch_size = dy.auto_batch_size
dynamic_thread = dy.dynamic_thread
omp_max_threads = dy.omp_max_threads
omp_num_procs = dy.omp_num_procs

# In-place SIMD Activation Functions
relu = dy.relu
drelu = dy.drelu
tanh = dy.tanh
dtanh = dy.dtanh
sigmoid = dy.sigmoid
dsigmoid = dy.dsigmoid

__all__ = [
    # Networks
    "SequentialPCN",
    "SimplePCN",
    "ConvolutionalPCN",
    "GaussSeidelPCN",
    "DKPPCN",
    # "SimpleConvolutionalPCN",
    
    # Utilities
    "StreamAlignedBatcher",
    "get_l2_cache_bytes",
    "auto_batch_size",
    "dynamic_thread",
    "omp_max_threads",
    "omp_num_procs",
    
    # Activations
    "relu",
    "drelu",
    "tanh",
    "dtanh",
    "sigmoid",
    "dsigmoid",
]
