import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, EnumProperty
from bpy.types import PropertyGroup


def vertex_group_items(caller, context):
    obj = context.active_object
    if obj and obj.type == "MESH" and obj.vertex_groups:
        return [(vg.name, vg.name, "") for vg in obj.vertex_groups]
    return [("NONE", "None", "")]


class ScalarVertexMapSettings(PropertyGroup):

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


MAPPINGS = [
    ("LINEAR", "Linear", "x' = x"),
    ("SMOOTH", "Smooth", "x' = 3x^2-2x^3"),
    ("BOOLEAN", "Boolean", "x' = ceil(x-$threshold)"),
    (
        "GAUSSIAN",
        "Gaussian",
        "x' = exp(-((x-$mean)/$(4*variance))^2))",
    ),
    ("SINE", "Sine", "x' = sin(x/$period+$phase)"),
    ("SAW", "Saw", "x' = (x+$phase)/$period-floor((x+$phase/2)/$period)"),
    ("STEP", "Step", "x' = floor(x*($steps+1)/$steps)"),
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
