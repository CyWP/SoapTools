import bpy
import torch

from bpy.props import EnumProperty
from bpy.types import PropertyGroup
from typing import List, Tuple

from ..logger import LOGGER


def get_torch_devices() -> List[Tuple[str]]:
    """Dynamically list available torch devices."""
    devices = [("CPU", "CPU", "Use the CPU for computation.")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append((f"cuda:{i}", f"GPU", f"GPU {i}: {name}"))
    return devices


class TorchDevice(PropertyGroup):
    device: EnumProperty(
        name="Device",
        description="Compute device used for torch operations",
        items=lambda self, context: get_torch_devices(),
        default=0,
    )

    def has_options(self) -> bool:
        return len(get_torch_devices()) > 1

    def draw(self, layout):
        if self.has_options():
            row = layout.row()
            row.prop(self, "device", expand=True)

    def get_device(self) -> torch.device:
        try:
            if torch.cuda.is_available() and self.device is not None:
                return torch.device(self.device)
        except Exception as e:
            LOGGER.debug(f"Error getting torch device, defaulting to CPU:\n{e}")
            return torch.device("cpu")
