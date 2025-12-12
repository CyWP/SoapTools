import bpy
import torch

from bpy.props import (
    IntProperty,
    EnumProperty,
)
from bpy.types import PropertyGroup, Image, Object

from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.enums import BlendEnums
from ..utils.blend_data.scene import bake_material

BAKE_CHANNELS = [
    ("DIFFUSE", "Diffuse", ""),
    ("EMIT", "Emit", ""),
    ("ROUGHNESS", "Roughness", ""),
    ("NORMAL", "Normal", ""),
]


class BakingSettings(PropertyGroup):

    uv_map: EnumProperty(
        name="UV Map",
        items=BlendEnums.uv_maps,
        description="Destination UV map for the baking process.",
    )
    material: EnumProperty(
        name="Material",
        items=BlendEnums.materials,
        description="Material for which seelcted channel will be baked.",
    )
    height: IntProperty(
        name="Height",
        description="Height of baked texture",
        default=512,
        min=1,
    )
    width: IntProperty(
        name="Width",
        description="Width of baked texture",
        default=512,
        min=1,
    )
    channel: EnumProperty(
        name="Channel",
        items=BAKE_CHANNELS,
        description="Channel to bake to an image.",
        default="EMIT",
    )

    def draw(self, layout):
        row = layout.row()
        row.prop(self, "uv_map")
        row = layout.row()
        row.prop(self, "material")
        row = layout.row()
        row.prop(self, "channel", expand=True)
        row = layout.row()
        row.prop(self, "height")
        row.prop(self, "width")

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
            bake_type=self.channel,
        )

    def get_map(
        self, obj: Object, device: torch.device, pack: bool = False
    ) -> torch.Tensor:
        img = BlendTorch.img2tensor(self.get_baked(obj, pack=pack), device)
        uv_idx, uv_co = BlendTorch.uv2tensor(obj, self.uv_map, device)
        return img.uv_sample(uv_idx, uv_co)
