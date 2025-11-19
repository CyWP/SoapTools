import bpy
import torch

from bpy.props import (
    BoolProperty,
    EnumProperty,
)
from bpy.types import PropertyGroup, Object
from typing import Tuple

from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.enums import BlendEnums
from ..utils.blend_data.vertex_groups import harden_vertex_group


class SimpleVertexGroup(PropertyGroup):
    group: EnumProperty(
        name="Vertex group",
        items=BlendEnums.vertex_groups,
        default=0,
    )  # type: ignore
    strict: BoolProperty(
        name="Strict",
        description="Prevents vertex group weights from smoothing and propagating after a subdivision pass.",
        default=True,
    )  # type: ignore

    def draw(self, layout, text: str = None):
        layout.prop(self, "group", text=text if text else "")
        layout.prop(self, "strict")

    def get_group(
        self, obj: Object, device: torch.device, none_valid=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nV = len(obj.data.vertices)
        if self.group == "NONE":
            if none_valid:
                return (
                    torch.ones((nV,), device=device),
                    torch.arange(0, nV, 1, dtype=torch.long, device=device),
                )
            raise ValueError("A vertex group must be selected.")
        if self.strict:
            harden_vertex_group(obj, self.group)
        return BlendTorch.vg2tensor(obj, self.group, device=device)
