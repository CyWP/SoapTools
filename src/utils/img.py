from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from PIL import Image
from typing import Optional, Union, List


class ImageTensor(Tensor):
    """
    A subclass of torch.Tensor specialized for images.
    - Always has shape [B, C, H, W]
    - Supports construction from files, PIL images, NumPy arrays, or tensors
    - Keeps metadata (source, size, etc.)
    """

    def __new__(
        cls, data, device: torch.device = None, source: Union[str, Path] = None
    ):
        if not isinstance(data, Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)

        if data.dtype != torch.float32:
            data = data.float()
        if data.max() > 1.0:
            data = data / 255.0
        if device:
            data = data.to(device)
        # Enforce [B, C, H, W]
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            if data.shape[-1] in (1, 3, 4):
                data = data.permute(2, 0, 1)
            assert data.shape[0] in (1, 3, 4), "3D tensor must have C=1,3,4"
            data = data.unsqueeze(0)
        elif data.ndim == 4:
            if data.shape[-1] in (1, 3, 4):
                data = data.permute(0, 3, 1, 2)
            assert data.shape[1] in (1, 3, 4), "4D tensor must have C=1,3,4"
        else:
            raise ValueError("ImageTensor must be 2D, 3D, or 4D input")

        assert data.ndim == 4, f"ImageTensor must be 4D [B, C, H, W], got {data.shape}"
        B, C, H, W = data.shape
        assert C in (1, 3, 4), f"Invalid channel count: {C}"
        obj = torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)
        obj.source = str(source)
        obj.batch = B
        obj.channels = C
        obj.height = H
        obj.width = W
        return obj

    @classmethod
    def from_file(cls, path: Path, device: torch.device) -> ImageTensor:
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.float32)
        return cls(arr, device=device, source=path)

    @classmethod
    def from_files(cls, device: torch.device, paths: List[str]) -> ImageTensor:
        imgs = [cls.from_file(p) for p in paths]
        return cls(
            torch.cat(imgs, dim=0), device=device, source="batch:" + ",".join(paths)
        )

    @classmethod
    def from_pil(
        cls, img: Image.Image, device: torch.device, source: Optional[str] = None
    ) -> ImageTensor:
        return cls(
            np.array(img.convert("RGBA"), dtype=np.float32),
            device=device,
            source=source,
        )

    @classmethod
    def from_numpy(
        cls, arr: np.ndarray, device: torch.device, source: Optional[str] = None
    ) -> ImageTensor:
        return cls(arr, device=device, source=source)

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        device: Optional[torch.device] = None,
        source: Optional[str] = None,
    ) -> ImageTensor:
        return cls(tensor, device=device, source=source)

    @classmethod
    def solid_image(H: int, W: int, color: torch.Tensor) -> ImageTensor:
        color = color.view(-1, 1, 1)  # ensure shape (1,1,C)
        img = torch.tile(color, (1, H, W))
        return ImageTensor.from_tensor(img, source="solid")

    def tensor(self) -> torch.Tensor:
        """
        Meant to extract tensor, useful during training/inference when
        the output will no longer be  avalid image.
        """
        return self.as_subclass(torch.Tensor)

    def to_numpy(self) -> Union[np.ndarray, List[np.ndarray]]:
        imgs = (self.permute(0, 2, 3, 1).contiguous().cpu().numpy() * 255).astype(
            np.uint8
        )
        if imgs.shape[0] == 1:
            return imgs[0]
        return [im for im in imgs]

    def to_pil(self) -> Union[Image.Image, List[Image.Image]]:
        imgs = self.to_numpy()
        if isinstance(imgs, list):
            return [Image.fromarray(i) for i in imgs]
        return Image.fromarray(imgs)

    def R(self) -> ImageTensor:
        return self[:, 0, :, :].unsqueeze(1)

    def G(self) -> ImageTensor:
        return self[:, 1, :, :].unsqueeze(1)

    def B(self) -> ImageTensor:
        return self[:, 2, :, :].unsqueeze(1)

    def BW(self, alpha=False) -> ImageTensor:
        if alpha:
            return self.mean(dim=1, keepdim=True)
        return self[:, :3, :, :].mean(dim=1, keepdim=True)

    def uv_sample(self, uv_idx: torch.Tensor, uv_co: torch.Tensor) -> torch.Tensor:
        """
        Vertex-based bilinear texture sampling and aggregation.

        self   : (B, C, H, W) BCHW texture
        uv_idx : (N,)  vertex index for each UV sample
        uv_co  : (N, 2) UV coordinates in [0,1]

        Returns:
            sampled : (B, nV, C) per-vertex averaged values
        """
        B, C, H, W = self.shape
        device = uv_co.device
        N = uv_idx.shape[0]
        nV = uv_idx.max().item() + 1

        # Convert UVs to pixel coordinates
        x = uv_co[:, 0] * (W - 1)
        y = uv_co[:, 1] * (H - 1)

        x0 = x.floor().long().clamp(0, W - 1)
        y0 = y.floor().long().clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        # Fractional part for bilinear interpolation
        fx = (x - x0.float()).view(1, 1, N)
        fy = (y - y0.float()).view(1, 1, N)
        inv_fx = 1 - fx
        inv_fy = 1 - fy

        # Sample corners (B, C, N)
        tl = self[:, :, y0, x0]
        tr = self[:, :, y0, x1]
        bl = self[:, :, y1, x0]
        br = self[:, :, y1, x1]

        # Bilinear interpolation
        vals = (
            tl * (inv_fx * inv_fy)
            + tr * (fx * inv_fy)
            + bl * (inv_fx * fy)
            + br * (fx * fy)
        )
        vals = vals.permute(0, 2, 1)  # (B, N, C)

        # Prepare output and frequency accumulator
        sampled = torch.zeros((B, nV, C), device=device)
        freqs = torch.zeros((nV,), device=device)

        # Expand uv_idx for batch dimension
        batch_offsets = (torch.arange(B, device=device) * nV).view(B, 1).repeat(1, N)
        scatter_idx = uv_idx.view(1, N).repeat(B, 1) + batch_offsets
        sampled_flat = sampled.view(B * nV, C)

        # Flatten vals to (B*N, C) for index_add
        vals_flat = vals.reshape(B * N, C)
        sampled_flat.index_add_(0, scatter_idx.reshape(-1), vals_flat)

        # Frequencies (shared across batch)
        freqs.index_add_(0, uv_idx, torch.ones_like(uv_idx, dtype=torch.float))

        # Reshape back to (B, nV, C)
        sampled = sampled_flat.view(B, nV, C)

        # Normalize by frequency
        sampled = sampled / freqs.view(1, nV, 1)
        return sampled.squeeze()
