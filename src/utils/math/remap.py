import torch


def remap_linear(x: torch.Tensor) -> torch.Tensor:
    return x


def remap_fill(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max()
    x_min = x.min()
    return x / (x_max - x_min) - x_min


def remap_smooth(x: torch.Tensor) -> torch.Tensor:
    return 3 * x**2 - 2 * x**3


def remap_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    result = torch.ones_like(x)
    result[x < threshold] = 0.0
    return result


def remap_gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return torch.exp((-0.25(x - mu) / sigma) ** 2)


def remap_sine(x: torch.Tensor, period: float, phase: float) -> torch.tensor:
    return 0.5 * (torch.sin(x / period + phase) + 1)


def remap_saw(x: torch.Tensor, period: float, phase: float) -> torch.Tensor:
    return (x + phase) / period - torch.floor((x + 0.5 * phase) / period)


def remap_pulse(x: torch.Tensor, period: float, phase: float) -> torch.Tensor:
    return torch.ceil(torch.sin(x / period + phase))


def remap_step(x: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.floor(x + x / steps)
