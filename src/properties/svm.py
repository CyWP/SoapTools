import bpy
import torch

from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    CollectionProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup, UIList, Operator, Object
from typing import Optional

from .img import ImageMappingSettings
from .symbolic import SymbolicExpression
from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.enums import BlendEnums
from ..utils.blend_data.vertex_groups import harden_vertex_group
from ..utils.math.remap import Remap

MAPPINGS = [
    ("LINEAR", "Linear", "x' = x"),
    ("EXPRESSION", "Expression", "Enter custom mathematical expression for map x."),
    (
        "REMAP_POINT",
        "Remap Point",
        "x' = x/(x+((1-x)*dest*(1-src)/(src*(1-dest)))), smoothly remaps src to dest.",
    ),
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
    )
    expression: PointerProperty(type=SymbolicExpression)
    period: FloatProperty(
        name="Period",
        description="Period of function, multiplied by 2*pi.",
        default=0.5,
        min=1e-6,
    )
    phase: FloatProperty(
        name="Phase",
        description="Phase of function, multiplied by 2*pi.",
        default=0,
    )
    mean: FloatProperty(
        name="mean",
        description="mean of function.",
        default=0.5,
        min=0,
    )
    variance: FloatProperty(
        name="Variance",
        description="Variance of function.",
        default=1.0,
        min=0,
    )
    steps: IntProperty(
        name="Steps",
        description="Number of steps",
        default=1,
        min=1,
    )
    threshold: FloatProperty(
        name="Threshold",
        description="Threshold for function",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    remap_src: FloatProperty(
        name="Value",
        description="Value to be remapped",
        default=0,
    )
    remap_dest: FloatProperty(
        name="Destination",
        description="Destination of remapped value in 0-1 space",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )

    def draw(self, layout):
        line = layout.row(align=True)
        map_type = self.map_type
        line.prop(self, "map_type", text="")
        if map_type in ("LINEAR", "SMOOTH", "FILL", "INVERT"):
            pass
        elif map_type == "EXPRESSION":
            line.prop(self.expression, "expression", text="")
        elif map_type == "REMAP_POINT":
            line.prop(self, "remap_src")
            line.prop(self, "remap_dest")
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

    def process(self, x: torch.Tensor, r0: float = 0, r1: float = 1) -> torch.Tensor:
        map_type = self.map_type
        map_min, map_max = min(r0, r1), max(r0, r1)
        if map_type == "REMAP_POINT":
            if self.remap_src < map_min or self.remap_src > map_max:
                raise ValueError("Remaping value cannot be outside of mapping range.")
            src_norm = float(self.remap_src - map_min) / (map_max - map_min)
            return Remap.map_val(x, src_norm, self.remap_dest)
        elif map_type == "EXPRESSION":
            return self.expression.eval({"x": x})
        if map_type == "LINEAR":
            return Remap.linear(x)
        if map_type == "INVERT":
            return Remap.invert(x)
        if map_type == "FILL":
            return Remap.fill(x)
        if map_type == "SMOOTH":
            return Remap.smooth(x)
        if map_type == "THRESHOLD":
            return Remap.threshold(x, self.threshold)
        if map_type == "GAUSSIAN":
            return Remap.gaussian(x, self.mean, self.variance)
        if map_type == "STEP":
            return Remap.step(x, self.steps)
        if map_type == "SINE":
            return Remap.sine(x, self.period, self.phase)
        if map_type == "SAW":
            return Remap.saw(x, self.period, self.phase)
        if map_type == "PULSE":
            return Remap.pulse(x, self.period, self.phase)
        raise ValueError(f"'{map_type}' is an unrecognized map type.")


class RemappingStack(PropertyGroup):
    modes: CollectionProperty(type=RemappingMode)
    active_index: IntProperty(default=0)

    def draw(self, layout, compact: bool = True):
        if len(self.modes) > 0 or not compact:
            layout.template_list(
                "SOAP_UL_ModeList", "", self, "modes", self, "active_index", rows=3
            )
            op = layout.operator(
                "soap.add_mode_operator", text="Add Function", icon="PLUS"
            )
            op.data_path = self.modes.path_from_id()
        else:
            op = layout.operator("soap.add_mode_operator", text="Remap")
            op.data_path = self.modes.path_from_id()

    def process(self, x: torch.Tensor, r0: float = 0, r1: float = 1):
        for mode in self.modes:
            x = mode.process(x, r0, r1)
        return x


class SOAP_UL_ModeList(UIList):
    bl_options = {"DEFAULT", "FILTER"}

    def draw_filter(self, context, layout):
        # Even empty forces Blender to include the header row
        layout.label(text="Remapping Functions")

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        if item:
            row = item.draw(layout)
            op = row.operator("soap.remove_mode_operator", text="", icon="TRASH")
            op.data_path = data.path_from_id()  # string path to the PropertyGroup
            op.idx = index


class SOAP_OT_AddModeOperator(Operator):
    bl_idname = "soap.add_mode_operator"
    bl_label = "Add Mode"
    bl_description = "Add remapping function to the vertex groups values [0, 1] before applying range or weights."
    bl_options = {"INTERNAL"}

    data_path: StringProperty()

    def execute(self, context):
        target = eval(f"context.scene.{self.data_path}")
        target.add()
        return {"FINISHED"}


class SOAP_OT_RemoveModeOperator(Operator):
    bl_idname = "soap.remove_mode_operator"
    bl_label = "Remove Mode"
    bl_options = {"INTERNAL"}

    data_path: bpy.props.StringProperty()
    idx: IntProperty()

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
    )
    target_obj: PointerProperty(type=Object)
    apply_after: EnumProperty(
        name="Apply after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=lambda self, context: BlendEnums.modifiers(
            self, context, object=self.target_obj, use_active=False
        ),
    )
    use_active: BoolProperty(
        name="Use Active", description="use active object for reference", default=True
    )
    val: FloatProperty(name="", description="Value", default=1)
    r_0: FloatProperty(name="", description="Start of mapping range", default=0)
    r_1: FloatProperty(name="", description="End of mapping range", default=1)
    group: EnumProperty(
        name="Vertex group",
        items=lambda self, context: BlendEnums.vertex_groups(
            self, context, object=self.target_obj, use_active=self.use_active
        ),
        default=0,
    )
    strict: BoolProperty(
        name="Strict",
        description="Prevents vertex group weights from smoothing and propagating after a subdivision pass.",
        default=False,
    )
    remap_stack: PointerProperty(type=RemappingStack)
    map_source: EnumProperty(
        name="Source",
        items=[
            ("VERTEX GROUP", "Vertex Group", ""),
            ("IMAGE", "Image", ""),
            ("CONSTANT", "Constant", ""),
        ],
        description="Source for creating scalar value map.",
        default="CONSTANT",
    )
    img_map: PointerProperty(type=ImageMappingSettings)

    def get_field(self, obj: Object, device: torch.device) -> Optional[torch.Tensor]:
        nV = len(obj.data.vertices)
        constant_field = self.map_source == "CONSTANT"
        use_range = self.val_mode == "RANGE" and not constant_field
        r0, r1, val = float(self.r_0), float(self.r_1), float(self.val)
        if self.strict and self.group != "NONE":
            harden_vertex_group(obj, self.group)
        if self.map_source == "VERTEX GROUP":
            field, idx = BlendTorch.vg2tensor(obj, self.group, device=device)
        elif self.map_source == "IMAGE":
            field = self.img_map.get_map(obj, device)
        else:
            field = torch.ones((nV,), device=device)
        if not constant_field:
            field = self.remap_stack.process(field, r0=r0, r1=r1)
        field = (r1 - r0) * field + r0 if use_range else val * field
        return field

    def draw(self, layout, name: str = None):
        constant_field = self.map_source == "CONSTANT"
        use_range = self.val_mode == "RANGE" and not constant_field
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Map Source")
        row = box.row(align=True)
        row.prop(self, "map_source", expand=True)
        if self.map_source == "VERTEX GROUP":
            row = box.row()
            left = row.split(factor=0.8)
            left.prop(self, "group")
            right = left.row()
            right.prop(self, "strict")
        elif self.map_source == "IMAGE":
            self.img_map.draw(box)
        box = layout.box()
        row = box.row(align=True)
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Map value")
        if not constant_field:
            row = box.row()
            row.prop(self, "val_mode", expand=True)
        row = box.row(align=True)
        if use_range:
            row.prop(self, "r_0")
            row.prop(self, "r_1")
        else:
            row.prop(self, "val")
        if len(self.remap_stack.modes) >= 1 and not constant_field:
            row = box.row()
            row.alignment = "CENTER"
            row.enabled = False
        if not constant_field:
            self.remap_stack.draw(box)
