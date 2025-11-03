import bpy
import torch


def get_torch_devices(self, context):
    """Dynamically list available torch devices."""
    devices = [("cpu", "CPU", "Use the CPU for computations")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append((f"cuda:{i}", f"CUDA:{i}", f"GPU {i}: {name}"))
    return devices


class GlobalSettings(bpy.types.PropertyGroup):
    device: bpy.props.EnumProperty(
        name="Device",
        description="Compute device used for torch operations",
        items=get_torch_devices,
    )  # type: ignore
