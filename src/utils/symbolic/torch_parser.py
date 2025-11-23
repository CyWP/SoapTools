import math
import torch
import torch.nn.functional as F

from .parser import Parser


class TorchParser(Parser):

    _torch_functions = {
        "abs": lambda x: x.abs(),
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "log": torch.log,
        "softmax": lambda x: F.softmax(x, dim=-1),
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
        "relu": F.relu,
        "normalize": lambda x: F.normalize(x, dim=-1),
        "norm": lambda x: x.norm(),
        "max": lambda x: x.max(),
        "min": lambda x: x.min(),
        "mean": lambda x: x.mean(),
    }

    _torch_constants = {"pi": torch.tensor(math.pi), "e": torch.tensor(math.e)}

    def __init__(self):
        super().__init__(
            functions=TorchParser._torch_functions,
            constants=TorchParser._torch_constants,
        )
