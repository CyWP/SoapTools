import torch


def batched_dot(a, b):
    return torch.sum(a * b, dim=-1)
