import torch


class SolverConfig:

    def __init__(
        self,
        solver: str = "AUTO",
        precond: str = "NONE",
        iters: int = 100,
        tolerance: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.solver = solver
        self.precond = precond
        self.iters = iters
        self.tolerance = tolerance
        self.device = device
