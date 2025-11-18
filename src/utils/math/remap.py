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
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def smooth(x: torch.Tensor) -> torch.Tensor:
        return 3 * x**2 - 2 * x**3

    @staticmethod
    def threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        result = torch.ones_like(x)
        result[x < threshold] = 0.0
        return result

    @staticmethod
    def gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
        return torch.exp((-0.25 * (x - mu) / sigma) ** 2)

    @staticmethod
    def sine(x: torch.Tensor, period: float, phase: float) -> torch.tensor:
        return 0.5 * (torch.sin(x / period + phase) + 1)

    @staticmethod
    def saw(x: torch.Tensor, period: float, phase: float) -> torch.Tensor:
        return (x + phase) / period - torch.floor((x + 0.5 * phase) / period)

    @staticmethod
    def pulse(x: torch.Tensor, period: float, phase: float) -> torch.Tensor:
        return torch.ceil(torch.sin(x / period + phase))

    @staticmethod
    def step(x: torch.Tensor, steps: int) -> torch.Tensor:
        return torch.floor(x + x / steps)
