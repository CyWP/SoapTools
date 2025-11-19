import torch


class Remap:
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
