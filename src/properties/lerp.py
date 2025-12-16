import bpy
import torch

from bpy.props import (
    PointerProperty,
    EnumProperty,
    IntProperty,
    BoolProperty,
    StringProperty,
    CollectionProperty,
)
from bpy.types import Context, Object, Operator, PropertyGroup
from typing import Tuple, List, Optional

from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.enums import BlendEnums
from ..utils.blend_data.mesh_obj import apply_first_n_modifiers
from ..utils.blend_data.scene import (
    duplicate_mesh_object,
    link_to_same_scene_collections,
    temp_copy,
)
from .svm import ScalarVertexMapSettings


class InterpolationTarget(PropertyGroup):
    weights_map: PointerProperty(type=ScalarVertexMapSettings)

    def draw(self, layout):
        row = layout.row()
        row.prop(self.weights_map, "target_obj")
        if self.weights_map.target_obj is not None:
            row.prop(self.weights_map, "apply_after")
            self.weights_map.draw(layout)

    def get_field(
        self, device: torch.device, reference: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        target = self.weights_map.target_obj
        apply_after = int(self.weights_map.apply_after)
        if target is None:
            raise Exception("Target mesh cannot be None.")
        if target.type != "MESH":
            raise Exception("Target must be mesh object.")
        with temp_copy(target, apply_after=apply_after) as tmp_obj:
            V, F = BlendTorch.mesh2tensor(tmp_obj, device=device)
            W = self.weights_map.get_field(tmp_obj, device).unsqueeze(1)
        if reference is not None:
            V = V - reference
        return W * V, F

    def get_mesh(self) -> Object:
        target = self.weights_map.target_obj
        apply_after = int(self.weights_map.apply_after)
        if target is None:
            raise Exception("Target mesh cannot be None.")
        if target.type != "MESH":
            raise Exception("Target must be mesh object.")
        new_obj = duplicate_mesh_object(target, deep=True)
        link_to_same_scene_collections(target, new_obj)
        if apply_after > 0:
            apply_first_n_modifiers(new_obj, apply_after)
        return new_obj


class InterpolationSettings(PropertyGroup):
    reference: PointerProperty(type=Object)
    apply_after: EnumProperty(
        name="Apply after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=lambda self, context: BlendEnums.modifiers(
            self, context, object=self.reference, use_active=False
        ),
    )
    targets: CollectionProperty(type=InterpolationTarget)
    target_names: EnumProperty(
        name="Variables", items=lambda self, context: self.get_target_names(context)
    )

    def get_target_names(self, context: Context) -> List[Tuple[str]]:
        vars = []
        if len(self.targets) == 0:
            return [("NONE", "Add target", "")]
        for i, target in enumerate(self.targets):
            vars.append((f"{i}", f"{i}", ""))
        return vars

    def current_idx(self) -> int:
        tname = self.target_names
        if tname:
            return int(self.target_names)
        return -1

    def draw(self, layout):
        layout.prop(self, "reference", text="Reference")
        if self.reference:
            layout.prop(self, "apply_after")
        box = layout.box()
        row = box.row(align=True)
        row.alignment = "CENTER"
        row.label(text="Targets")
        op = row.operator("soap.add_lerp_var", text="", icon="PLUS")
        op.data_path = self.targets.path_from_id()
        op = row.operator("soap.remove_lerp_var", text="", icon="REMOVE")
        op.data_path = self.targets.path_from_id()
        if len(self.targets) > 0:
            target_idx = self.current_idx()
            op.idx = target_idx
            row = box.row()
            row.prop(self, "target_names", expand=True)
            if target_idx >= 0:
                current_target = self.targets[target_idx]
                current_target.draw(box)

    def get_field(self, device: torch.device) -> Tuple[torch.Tensor]:
        if len(self.targets) < 1:
            raise Exception("Interpolation needs at least one target.")
        ref_V = None
        if self.reference is not None:
            with temp_copy(self.reference, apply_after=int(self.apply_after)) as tmp:
                ref_V, _ = BlendTorch.mesh2tensor(tmp, device=device)
        V, F = self.targets[0].get_field(device, reference=ref_V)
        if len(self.targets) == 1:
            return V, F
        for t in self.targets[1:]:
            v, _ = t.get_field(device, reference=ref_V)
            if v.shape != V.shape:
                raise Exception(
                    "Cannot interpolate between meshes that do not have the same number of vertices."
                )
            V = V + v
        if ref_V is not None:
            V = V + ref_V
        return V, F


class SOAP_OT_AddInterpolationVariable(Operator):
    bl_idname = "soap.add_lerp_var"
    bl_label = "Add Target"
    bl_description = "Add Target to interpolation."
    bl_options = {"INTERNAL"}

    data_path: StringProperty()

    def execute(self, context):
        target = eval(f"context.scene.{self.data_path}")  # resolve the PropertyGroup
        target.add()
        target[-1].weights_map.use_active = False
        return {"FINISHED"}


class SOAP_OT_RemoveInterpolationVariable(Operator):
    bl_idname = "soap.remove_lerp_var"
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
