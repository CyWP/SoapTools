import torch

from typing import Dict

from ...easydict import EasyDict


class SolverConfig(EasyDict):

    _default = EasyDict(
        solver="AUTO",
        precond="NONE",
        iters=100,
        tolerance=0.0,
        block_size=3,
        device=torch.device("cpu"),
    )

    def __init__(self, data: Dict):
        config = EasyDict(SolverConfig._default)
        config.update(data)
        super().__init__(config)
