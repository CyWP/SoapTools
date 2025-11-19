import bpy
import torch

from bpy.props import EnumProperty, PointerProperty
from bpy.types import PropertyGroup, Image, Object
from pathlib import Path
from typing import Any, Optional

from .baking import BakingSettings
from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.enums import BlendEnums
from ..utils.img import ImageTensor

IMG_SRC = [
    ("INTERNAL", "Internal", ""),
    ("EXTERNAL", "External", ""),
    ("MATERIAL", "Material", ""),
]
IMG_CHANNELS = [
    ("R", "R", ""),
    ("G", "G", ""),
    ("B", "B", "B"),
    ("A", "A", ""),
    ("BW", "BW", ""),
    ("BWA", "BWA", ""),
]


class ImageSettings(PropertyGroup):

    source: EnumProperty(
        name="Source",
        items=IMG_SRC,
        description="Source of image to load",
    )  # type:ignore
    internal_img: PointerProperty(type=Image)  # type:ignore
    path: bpy.props.StringProperty(
        name="Path",
        description="External source for loading image.",
        subtype="FILE_PATH",
    )  # type:ignore
    bake: PointerProperty(type=BakingSettings)  # type:ignore

    def draw(self, layout):
        row = layout.row()
        row.prop(self, "source", expand=True)
        row = layout.row()
        if self.source == "INTERNAL":
            row.prop(self, "internal_img")
        elif self.source == "EXTERNAL":
            row.prop(self, "path")
        else:
            self.bake.draw(layout)

    def get_image(
        self, device: torch.device, obj: Optional[Object] = None
    ) -> ImageTensor:
        if self.source == "INTERNAL":
            if self.internal_img is None:
                raise ValueError(f"Select an image to be loaded.")
            return BlendTorch.img2tensor(self.internal_img, device=device)
        elif self.source == "EXTERNAL":
            if self.path is None:
                raise ValueError(f"Select an image to be loaded.")
            return ImageTensor.from_file(Path(self.path), device=device)
        elif self.source == "MATERIAL":
            if obj is None:
                raise ValueError("Object cannot be None for baking a material.")
            img = self.bake.get_baked(obj, pack=False)
            return BlendTorch.img2tensor(img, device)
        else:
            raise ValueError(f"Source '{self.source}' is unrecognized for images.")

    def get_source_name(self) -> str:
        src = self.source
        if src == "INTERNAL":
            return self.internal_img.name
        if src == "EXTERNAL":
            return Path(self.path).name
        return self.bake.material


class ImageMappingSettings(PropertyGroup):

    img: PointerProperty(type=ImageSettings)  # type:ignore
    uv_map: EnumProperty(
        name="UV Map",
        items=BlendEnums.uv_maps,
        description="UV Map to use for mapping image.",
    )  # type:ignore
    channel: EnumProperty(
        name="Channel",
        items=IMG_CHANNELS,
        description="Channel to map to vertex group.",
        default="BWA",
    )  # type:ignore

    def draw(self, layout):
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Image Mapping")
        row = box.row()
        row.prop(self, "uv_map")
        row = box.row()
        row.prop(self, "channel", expand=True)
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Image Source")
        self.img.draw(box)

    def get_map(self, obj: Object, device: torch.device) -> torch.Tensor:
        if self.uv_map == "NONE":
            raise Exception("Please select a uv map.")
        img = self.img.get_image(device, obj=obj)
        if self.channel == "R":
            img = img.R()
        elif self.channel == "G":
            img = img.G()
        elif self.channel == "B":
            img = img.B()
        elif self.channel == "BW":
            img = img.BW(alpha=False)
        else:
            img = img.BW(alpha=True)
        uv_idx, uv_co = BlendTorch.uv2tensor(obj, self.uv_map, device)
        return img.uv_sample(uv_idx, uv_co)

    def create_vertex_group(self, obj: Object, device: torch.device) -> Any:
        mp = self.get_map(obj, device)
        name = "_".join(["soap", "img2vg", self.img.get_source_name()])
        vg = BlendTorch.tensor2vg(obj, name, mp)
        return vg
