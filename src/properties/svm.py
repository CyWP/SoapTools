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
from bpy.types import PropertyGroup, UIList, Operator
from typing import Optional

from ..utils.blend_data.bridges import vg2tensor
from ..utils.blend_data.vertex_groups import vertex_group_items, harden_vertex_group
from ..utils.math.remap import (
    remap_linear,
    remap_fill,
    remap_gaussian,
    remap_pulse,
    remap_saw,
    remap_sine,
    remap_smooth,
    remap_step,
    remap_threshold,
)


class ScalarVertexMapSettings(PropertyGroup):

    mode: EnumProperty(
        name="Mode",
        items=[("BASIC", "Basic", ""), ("ADVANCED", "Advanced", "")],
        default=0,
    )  # type:ignore
    group: EnumProperty(
        name="Vertex group",
        items=vertex_group_items,
        default=0,
    )  # type: ignore
    strict: BoolProperty(
        name="Strict",
        description="Avoids vertex group weights to smooth and propagate after a subdivision pass.",
        default=True,
    )  # type: ignore
    invert: BoolProperty(
        name="Invert",
        description="Invert values from vertex group in 0-1 range.",
        default=False,
    )  # type: ignore
    remap_stack: PointerProperty(type=RemappingStack)  # type: ignore

    def get_map(
        self, context, range_0: float, range_1: float, device: torch.device
    ) -> Optional[torch.Tensor]:
        if self.group == "NONE":
            return
        obj = context.active_object
        if self.strict:
            harden_vertex_group(obj, self.group)
        field = vg2tensor(obj, self.group, device=device)
        if self.invert:
            field = 1 - field
        if self.mode == "BASIC":
            return range_1 * field
        field = self.remap_stack.process(field)
        return (range_1 - range_0) * field + range_0

    def draw(self, layout):
        layout.prop(self, "mode", expand=True)
        mode = self.mode
        row = layout.row()
        row.prop(self, "group")
        row.prop(self, "strict")
        row.prop(self, "invert")
        if mode == "ADVANCED":
            self.remap_stack.draw(layout)


MAPPINGS = [
    ("LINEAR", "Linear", "x' = x"),
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
        default=1.0,
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
        line = layout.row()
        map_type = self.map_type
        line.prop(self, "map_type")
        if map_type in ("LINEAR", "SMOOTH", "FILL"):
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
        layout.template_list(
            "REMAP_UL_ModeList", "", self, "modes", self, "active_index", rows=4
        )
        layout.operator("soap.add_mode_operator", text="Add Remap").collection_ptr = (
            self
        )

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
            op = row.operator(
                "soap.remove_mode_operator", text="", icon="X", emboss=False
            )
            op.collection_ptr = data
            op.idx = index


class REMAP_OT_AddModeOperator(Operator):
    bl_idname = "soap.add_mode_operator"
    bl_label = "Add Mode"
    bl_options = {"INTERNAL"}

    collection_ptr: PointerProperty(type=RemappingStack)  # type:ignore

    def execute(self, context):
        self.collection_ptr.modes.add()
        return {"FINISHED"}


class REMAP_OT_RemoveModeOperator(Operator):
    bl_idname = "soap.remove_mode_operator"
    bl_label = "Remove Mode"
    bl_options = {"INTERNAL"}

    collection_ptr: PointerProperty(type=RemappingStack)  # type: ignore
    idx: IntProperty()  # type:ignore

    def execute(self, context):
        coll = self.collection_ptr
        if 0 <= self.idx < len(coll.modes):
            coll.modes.remove(self.idx)
        return {"FINISHED"}
