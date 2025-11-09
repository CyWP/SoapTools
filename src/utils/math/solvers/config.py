import torch


class SolverConfig:

    def __init__(
        self,
        solver: str = "AUTO",
        precond: str = "NONE",
        iters: int = 100,
        tolerance: float = 0.0,
        block_size: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        self.solver = solver
        self.precond = precond
        self.iters = iters
        self.tolerance = tolerance
        self.block_size = block_size
        self.device = device
