import bpy
import torch

from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    CollectionProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup, UIList, Operator, Object
from typing import Optional

from ..utils.blend_data.bridges import vg2tensor
from ..utils.blend_data.vertex_groups import vertex_group_items, harden_vertex_group
from ..utils.math.remap import (
    remap_linear,
    remap_invert,
    remap_fill,
    remap_gaussian,
    remap_pulse,
    remap_saw,
    remap_sine,
    remap_smooth,
    remap_step,
    remap_threshold,
)


MAPPINGS = [
    ("LINEAR", "Linear", "x' = x"),
    ("INVERT", "Invert", "x' = 1-x"),
    ("FILL", "Fill", "x' = (x-x_min)/(x_max-x_min)"),
    ("SMOOTH", "Smooth", "x' = 3x^2-2x^3"),
    ("THRESHOLD", "Threshold", "x' = ceil(x-$threshold)"),
    ("GAUSSIAN", "Gaussian", "x' = exp(-((x-$mean)/$(4*variance))^2))"),
    ("SINE", "Sine", "x' = sin(x/$period+$phase)"),
    ("SAW", "Saw", "x' = (x+$phase)/$period-floor((x+$phase/2)/$period)"),
    ("PULSE", "Pulse", "x' = ceil(sin(x/$period+$phase))"),
    ("STEP", "Step", "x' = floor(x+x/steps)"),
]


class RemappingMode(PropertyGroup):

    map_type: EnumProperty(
        name="Mapping",
        items=MAPPINGS,
        default=0,
        description="Remapping modes of values in 0-1 range space.",
    )  # type: ignore
    period: FloatProperty(
        name="Period",
        description="Period of function, multiplied by 2*pi.",
        default=0.5,
        min=1e-6,
    )  # type:ignore
    phase: FloatProperty(
        name="Phase",
        description="Phase of function, multiplied by 2*pi.",
        default=0,
    )  # type:ignore
    mean: FloatProperty(
        name="mean",
        description="mean of function.",
        default=0.5,
        min=0,
    )  # type:ignore
    variance: FloatProperty(
        name="Variance",
        description="Variance of function.",
        default=1.0,
        min=0,
    )  # type:ignore
    steps: IntProperty(
        name="Steps",
        description="Number of steps",
        default=1,
        min=1,
    )  # type:ignore
    threshold: FloatProperty(
        name="Threshold",
        description="Threshold for function",
        default=0.5,
        min=0.0,
        max=1.0,
    )  # type: ignore

    def draw(self, layout):
        line = layout.row(align=True)
        map_type = self.map_type
        line.prop(self, "map_type", text="")
        if map_type in ("LINEAR", "SMOOTH", "FILL", "INVERT"):
            pass
        elif map_type == "THRESHOLD":
            line.prop(self, "threshold")
        elif map_type == "GAUSSIAN":
            line.prop(self, "mean")
            line.prop(self, "variance")
        elif map_type == "STEP":
            line.prop(self, "steps")
        elif map_type in ("SINE", "SAW", "PULSE"):
            line.prop(self, "period")
            line.prop(self, "phase")
        else:
            raise ValueError(f"'{map_type}' is an unrecognized map type.")
        return line

    def process(self, x: torch.Tensor) -> torch.Tensor:
        map_type = self.map_type
        if map_type == "LINEAR":
            return remap_linear(x)
        if map_type == "INVERT":
            return remap_invert(x)
        if map_type == "FILL":
            return remap_fill(x)
        if map_type == "SMOOTH":
            return remap_smooth(x)
        if map_type == "THRESHOLD":
            return remap_threshold(x, self.threshold)
        if map_type == "GAUSSIAN":
            return remap_gaussian(x, self.mean, self.variance)
        if map_type == "STEP":
            return remap_step(x, self.steps)
        if map_type == "SINE":
            return remap_sine(x, self.period, self.phase)
        if map_type == "SAW":
            return remap_saw(x, self.period, self.phase)
        if map_type == "PULSE":
            return remap_pulse(x, self.period, self.phase)
        raise ValueError(f"'{map_type}' is an unrecognized map type.")


class RemappingStack(PropertyGroup):
    modes: CollectionProperty(type=RemappingMode)  # type:ignore
    active_index: IntProperty(default=0)  # type:ignore

    def draw(self, layout):
        if len(self.modes) > 0:
            layout.template_list(
                "REMAP_UL_ModeList", "", self, "modes", self, "active_index", rows=5
            )
            op = layout.operator("soap.add_mode_operator", text="", icon="PLUS")
            op.data_path = self.modes.path_from_id()
        else:
            op = layout.operator("soap.add_mode_operator", text="Remap")
            op.data_path = self.modes.path_from_id()

    def process(self, x: torch.Tensor):
        for mode in self.modes:
            x = mode.process(x)
        return x


class REMAP_UL_ModeList(UIList):
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        if item:
            row = item.draw(layout)
            op = row.operator("soap.remove_mode_operator", text="", icon="TRASH")
            op.data_path = data.path_from_id()  # string path to the PropertyGroup
            op.idx = index


class REMAP_OT_AddModeOperator(Operator):
    bl_idname = "soap.add_mode_operator"
    bl_label = "Add Mode"
    bl_description = "Add remapping function to the vertex groups values [0, 1] before applying range or weights."
    bl_options = {"INTERNAL"}

    data_path: bpy.props.StringProperty()  # type: ignore

    def execute(self, context):
        target = eval(f"context.scene.{self.data_path}")
        target.add()
        return {"FINISHED"}


class REMAP_OT_RemoveModeOperator(Operator):
    bl_idname = "soap.remove_mode_operator"
    bl_label = "Remove Mode"
    bl_options = {"INTERNAL"}

    data_path: bpy.props.StringProperty()  # type: ignore
    idx: IntProperty()  # type: ignore

    def execute(self, context):
        target = eval(f"context.scene.{self.data_path}")  # resolve the PropertyGroup
        if 0 <= self.idx < len(target.modes):
            target.modes.remove(self.idx)
        return {"FINISHED"}


class ScalarVertexMapSettings(PropertyGroup):

    val_mode: EnumProperty(
        name="Value",
        items=[
            ("VALUE", "Value", ""),
            ("RANGE", "Range", ""),
        ],
        default="VALUE",
    )  # type:ignore
    val: FloatProperty(name="", description="Value", default=1)  # type:ignore
    r_0: FloatProperty(
        name="", description="Start of mapping range", default=0
    )  # type:ignore
    r_1: FloatProperty(
        name="", description="End of mapping range", default=1
    )  # type:ignore
    group: EnumProperty(
        name="Vertex group",
        items=vertex_group_items,
        default=0,
    )  # type: ignore
    strict: BoolProperty(
        name="Strict",
        description="Prevents vertex group weights from smoothing and propagating after a subdivision pass.",
        default=False,
    )  # type: ignore
    remap_stack: PointerProperty(type=RemappingStack)  # type: ignore

    def get_field(self, obj: Object, device: torch.device) -> Optional[torch.Tensor]:
        nV = len(obj.data.vertices)
        use_range = self.val_mode == "RANGE"
        if self.strict and self.group != "NONE":
            harden_vertex_group(obj, self.group)
        field, idx = (
            vg2tensor(obj, self.group, device=device)
            if self.group != "NONE"
            else (
                torch.ones((nV,), device=device),
                None,
            )
        )
        field = self.remap_stack.process(field)
        r0, r1, val = float(self.r_0), float(self.r_1), float(self.val)
        field = (r1 - r0) * field + r0 if use_range else val * field
        return field

    def draw(self, layout, name: str):
        use_range = self.val_mode == "RANGE"
        row = layout.row(align=True)
        row.label(text=name)
        row.prop(self, "val_mode", expand=True)
        row = layout.row(align=True)
        if use_range:
            row.prop(self, "r_0")
            row.prop(self, "r_1")
        else:
            row.prop(self, "val")
        row = layout.row(align=True)
        if len(self.remap_stack.modes) < 1:
            left = row.split(factor=0.5)
            left.prop(self, "group", text="")
            right = left.row()
            right.prop(self, "strict")
            self.remap_stack.draw(right)
        else:
            left = row.split(factor=0.8)
            left.prop(self, "group", text="")
            right = left.row()
            right.prop(self, "strict")
            row = layout.row()
            row.alignment = "CENTER"
            row.enabled = False
            row.label(text="Remapping Functions")
            self.remap_stack.draw(layout)
