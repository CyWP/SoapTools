import bpy
import torch

from bpy.props import (
    BoolProperty,
    EnumProperty,
)
from bpy.types import PropertyGroup, Object
from typing import Tuple

from ..utils.blend_data.bridges import vg2tensor
from ..utils.blend_data.vertex_groups import vertex_group_items, harden_vertex_group


class SimpleVertexGroup(PropertyGroup):
    group: EnumProperty(
        name="Vertex group",
        items=vertex_group_items,
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
        self, obj: Object, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nV = len(obj.data.vertices)
        if self.group == "NONE":
            return (
                torch.ones((nV,), device=device),
                torch.tensor([], device=device, dtype=torch.long),
            )
        if self.strict:
            harden_vertex_group(obj, self.group)
        return vg2tensor(obj, self.group, device=device)
