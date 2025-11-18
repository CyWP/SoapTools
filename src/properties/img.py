import bpy
import torch

from bpy.props import EnumProperty, PointerProperty
from bpy.types import PropertyGroup, Image

from pathlib import Path

from ..utils.blend_data.bridges import img2tensor
from ..utils.img import ImageTensor

IMG_SRC = [("INTERNAL", "Internal", ""), ("EXTERNAL", "External", "")]


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

    def draw(self, layout):
        row = layout.row()
        left = row.split(factor=0.5)
        left.prop(self, "source", expand=True)
        right = left.row()
        if self.source == "INTERNAL":
            right.prop(self, "internal_img")
        else:
            right.prop(self, "path")

    def load(self, device: torch.device) -> ImageTensor:
        if self.source == "INTERNAL":
            if self.internal_img is None:
                raise ValueError(f"Select an image to be loaded.")
            return img2tensor(self.img, device=device)
        else:
            return ImageTensor.from_file(Path(self.path), device=device)
