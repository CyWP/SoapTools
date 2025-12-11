import bpy
import torch

from bpy.types import Operator, Context

from ..utils.blend_data.scene import (
    duplicate_mesh_object,
    link_to_same_scene_collections,
)
from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.mesh_obj import (
    apply_first_n_modifiers,
    safe_delete,
    safe_select,
)
from ..utils.blend_data.operators import process_operator
from ..utils.math.problems import solve_flation


@process_operator
class SOAP_OT_Inflation(Operator):
    bl_idname = "soap.inflate"
    bl_label = "SoapTools: Flation"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Generate displacement with constraints on a mesh object."
    bl_options = {"REGISTER", "UNDO"}

    suffix: str = "_inflated"

    @classmethod
    def poll(cls, context: Context):
        return True

    def invoke(self, context: Context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Context):
        layout = self.layout
        settings = context.scene.soap_settings.flation

        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Settings")
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(settings.fixed_verts, "group", text="Pinned")
        right = left.row()
        right.prop(settings.fixed_verts, "strict")
        row = box.row()
        row.prop(settings, "apply_after")
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Constraints")
        row = box.row()
        row.prop(settings, "active_constraint", expand=True)
        active = settings.active_constraint
        if active == "DISPLACEMENT":
            settings.displacement.draw(box, "Displacement")
        elif active == "LAPLACIAN":
            settings.laplacian.draw(box, "Laplacian")
        elif active == "ALPHA":
            settings.alpha.draw(box, "Alpha")
        elif active == "BETA":
            settings.beta.draw(box, "Beta")
        else:
            raise ValueError(f"'{active} is an invalid constraint.")
        settings.solver.draw(layout)

    def setup(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.flation
        device = context.scene.soap_settings.device.get_device()
        self.solver_config = settings.solver.get_config(device)
        vg_fixed = settings.fixed_verts.group
        vg_fixed_strict = settings.fixed_verts.strict
        apply_after = int(settings.apply_after)

        new_obj = duplicate_mesh_object(obj, deep=True)
        new_obj.name = f"{new_obj.name}{SOAP_OT_Inflation.suffix}"
        new_obj.data.name = f"{new_obj.data.name}{SOAP_OT_Inflation.suffix}"
        if apply_after > 0:
            link_to_same_scene_collections(obj, new_obj)
            apply_first_n_modifiers(
                new_obj,
                apply_after,
                [vg_fixed] if vg_fixed_strict and vg_fixed != "NONE" else [],
            )

        self.V, self.F = BlendTorch.mesh2tensor(new_obj, device=device)
        self.N = BlendTorch.vn2tensor(new_obj, device=device)
        _, self.fixed_idx = settings.fixed_verts.get_group(new_obj, device)
        self.disp_field = settings.displacement.get_field(new_obj, device)
        self.laplacian_field = settings.laplacian.get_field(new_obj, device)
        neg_mask = self.laplacian_field < 0.0
        # This is totally arbitrary but makes negative constraints more controllable
        self.laplacian_field[neg_mask] = self.laplacian_field[neg_mask] / 10000
        self.alpha_field = settings.alpha.get_field(new_obj, device)
        self.beta_field = settings.beta.get_field(new_obj, device)
        self.new_obj = new_obj
        self.original_obj = obj

    def process(self) -> torch.Tensor:
        return solve_flation(
            self.solver_config,
            self.V,
            self.F,
            self.N,
            target_offset=self.disp_field,
            fixed_idx=self.fixed_idx,
            lambda_lap=self.laplacian_field,
            beta_normal=self.beta_field,
            alpha_tangent=self.alpha_field,
        )

    def coalesce(self, context: Context):
        new_V = self._result
        BlendTorch.tensor2mesh_update(self.new_obj, new_V)
        link_to_same_scene_collections(self.original_obj, self.new_obj)
        safe_select(self.new_obj)

    def rescind(self, context: Context):
        safe_delete(self.new_obj)
