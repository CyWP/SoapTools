import bpy
import torch

from bpy.props import (
    StringProperty,
    IntProperty,
    EnumProperty,
    CollectionProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup, Operator, Object, Context
from typing import Dict, List, Tuple, Union

from .svm import ScalarVertexMapSettings
from .symbolic import SymbolicExpression


class MapOperationSettings(PropertyGroup):

    exp: PointerProperty(type=SymbolicExpression)
    vars: CollectionProperty(type=ScalarVertexMapSettings)
    var_names: EnumProperty(
        name="Variables", items=lambda self, context: self.get_vars(context)
    )

    output_format: EnumProperty(
        name="Format",
        items=[("CLAMP", "Clamp", ""), ("REMAP", "Remap", "")],
        description="Method for dealing with out of range values.",
        default="CLAMP",
    )

    def get_vars(self, context: Context) -> List[Tuple[str]]:
        vars = []
        if len(self.vars) == 0:
            return [("NONE", "Add variable", "")]
        for i in range(len(self.vars)):
            char = self.var_from_idx(i)
            vars.append((char, char, ""))
        return vars

    def var_from_idx(self, idx: int) -> str:
        i = idx if idx < 4 else idx + 1  # e is taken by a constant
        return str(chr(97 + i))

    def idx_from_var(self, var: str) -> int:
        if not var:
            return -1
        c = ord(var[0]) - 97
        return c if c < 4 else c - 1

    def current_idx(self) -> int:
        return self.idx_from_var(self.var_names)

    def draw(self, layout):
        row = layout.row()
        row.prop(self.exp, "expression")
        box = layout.box()
        row = box.row(align=True)
        row.alignment = "CENTER"
        row.label(text="Variables")
        op = row.operator("soap.add_map_var", text="", icon="PLUS")
        op.data_path = self.vars.path_from_id()
        op = row.operator("soap.remove_map_var", text="", icon="REMOVE")
        op.data_path = self.vars.path_from_id()
        var_idx = self.current_idx()
        op.idx = var_idx
        row = box.row()
        row.prop(self, "var_names", expand=True)
        if var_idx >= 0:
            current_map = self.vars[var_idx]
            current_map.draw(box)
        row = box.row()
        row.prop(self, "output_format", expand=True)

    def get_field(self, obj: Object, device: torch.device) -> torch.Tensor:
        vars = dict()
        if len(self.vars) == 0:
            raise ValueError("At least one variable must be selected.")
        for i, vmap in enumerate(self.vars):
            vars[self.var_from_idx(i)] = vmap.get_field(obj, device)
        x = self.exp.eval(vars, tensor=True)
        if self.output_format == "CLAMP":
            return torch.clamp(x, 0, 1)
        else:
            x_min, x_max = x.min(), x.max()
            return (x - x_min) / (x_max - x_min)


class SOAP_OT_AddMapVariable(Operator):
    bl_idname = "soap.add_map_var"
    bl_label = "Add Variable"
    bl_description = "Add variable to remapping function"
    bl_options = {"INTERNAL"}

    data_path: StringProperty()

    def execute(self, context):
        target = eval(f"context.scene.{self.data_path}")  # resolve the PropertyGroup
        target.add()
        return {"FINISHED"}


class SOAP_OT_RemoveMapVariable(Operator):
    bl_idname = "soap.remove_map_var"
    bl_label = "Remove Variable"
    bl_description = "Remove variable from remapping function"
    bl_options = {"INTERNAL"}

    data_path: StringProperty()
    idx: IntProperty()

    def execute(self, context):
        target = eval(f"context.scene.{self.data_path}")  # resolve the PropertyGroup
        if 0 <= self.idx < len(target):
            target.remove(self.idx)
        return {"FINISHED"}
