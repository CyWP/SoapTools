import bpy
import torch

from bpy.props import (
    IntProperty,
    EnumProperty,
)
from bpy.types import PropertyGroup, Image, Object
from typing import Union

from ..utils.blend_data.bridges import img2tensor, uv2tensor
from ..utils.blend_data.scene import material_items, bake_material
from ..utils.blend_data.uv import uv_map_items
from ..utils.img import ImageTensor

BAKE_CHANNELS = [
    ("DIFFUSE", "Diffuse", ""),
    ("EMIT", "Emit", ""),
    ("ROUGHNESS", "Roughness", ""),
    ("NORMAL", "Normal", ""),
]


class BakingSettings(PropertyGroup):

    uv_map: EnumProperty(
        name="UV Map",
        items=uv_map_items,
        description="Destination UV map for the baking process.",
    )  # type:ignore
    material: EnumProperty(
        name="Material",
        items=material_items,
        description="Material for which seelcted channel will be baked.",
    )  # type: ignore
    height: IntProperty(
        name="Height",
        description="Height of baked texture",
        default=512,
        min=1,
    )  # type:ignore
    width: IntProperty(
        name="Width",
        description="Width of baked texture",
        default=512,
        min=1,
    )  # type:ignore
    channel: EnumProperty(
        name="Channel",
        items=BAKE_CHANNELS,
        description="Channel to bake to an image.",
        default="EMIT",
    )  # type:ignore

    def draw(self, layout):
        row = layout.row()
        row.prop(self, "uv_map")
        row.prop(self, "material")
        row = layout.row()
        left = row.split(factor=0.24)
        left.prop(self, "height", text="H")
        mid = left.row()
        mid = mid.split(factor=0.33)
        mid.prop(self, "width", text="W")
        right = mid.row()
        right.prop(self, "channel")

    def validate_input(self):
        if self.uv_map == "NONE":
            raise ValueError("Please select a UV Map.")
        if self.material == "NONE":
            raise ValueError("Please select a material to bake.")

    def get_baked(self, obj: Object, pack: bool = True) -> Image:
        self.validate_input()
        return bake_material(
            obj,
            self.uv_map,
            self.material,
            self.width,
            self.height,
            bake_type="EMIT",
        )

    def get_map(
        self, obj: Object, device: torch.device, pack: bool = False
    ) -> torch.Tensor:
        img = img2tensor(self.get_baked(obj, pack=pack), device)
        uv_idx, uv_co = uv2tensor(obj, self.uv_map, device)
        return img.uv_sample(uv_idx, uv_co)
