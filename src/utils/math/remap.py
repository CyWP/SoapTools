import torch


class Remap:
    """
    Bunch of functions for remapping in 0-1 1D space.
    """

    @staticmethod
    def map_val(x: torch.Tensor, source: float, dest: float, eps=1e-5) -> torch.Tensor:
        if source < eps or 1 - dest < eps:
            return torch.zeros_like(x)
        if dest < eps or 1 - source < eps:
            return torch.ones_like(x)
        return x / (x + ((1 - x) * dest * (1 - source) / (source * (1 - dest))))

    @staticmethod
    def linear(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def invert(x: torch.Tensor) -> torch.Tensor:
        return 1 - x

    @staticmethod
    def fill(x: torch.Tensor) -> torch.Tensor:
        x_max = x.max()
        x_min = x.min()
        den = (x_max - x_min).clamp(min=1e-8)
        return (x - x_min) / den

    @staticmethod
    def smooth(x: torch.Tensor) -> torch.Tensor:
        return 3 * x**2 - 2 * x**3

    @staticmethod
    def threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        return (x >= threshold).float()

    @staticmethod
    def gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
        return torch.exp(-((x - mu) ** 2) / (2 * sigma**2))

    @staticmethod
    def sine(x: torch.Tensor, period: float, phase: float) -> torch.tensor:
        return 0.5 * (torch.sin(2 * torch.pi * (x / period + phase)) + 1)

    @staticmethod
    def saw(x: torch.Tensor, period: float, phase: float) -> torch.Tensor:
        return (x / period + phase) % 1.0

    @staticmethod
    def pulse(x: torch.Tensor, period: float, phase: float) -> torch.Tensor:
        t = (x / period + phase) % 1.0
        return (t < 0.5).float()

    @staticmethod
    def step(x: torch.Tensor, steps: int) -> torch.Tensor:
        return torch.floor(x * steps) / steps
