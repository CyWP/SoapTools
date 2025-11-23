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
        "soft": lambda x: F.softmax(x, dim=-1),
        "sig": F.sigmoid,
        "tanh": F.tanh,
        "relu": F.relu,
        "normz": lambda x: F.normalize(x, dim=-1),
        "norm": lambda x: x.norm(),
        "max": lambda x: x.max(),
        "emax": lambda x, y: torch.maximum(x, y),
        "min": lambda x: x.min(),
        "emin": lambda x, y: torch.minimum(x, y),
        "mean": lambda x: x.mean(),
        "clamp": lambda x, a, b: torch.clamp(x, min=a, max=b),
    }

    def __init__(self):
        super().__init__(
            functions=TorchParser._torch_functions,
        )
