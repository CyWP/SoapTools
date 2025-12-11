import bpy
import torch

from bpy.types import Context, Event, Operator

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
from ..utils.math.problems import solve_minimal_surface


@process_operator
class SOAP_OT_MinimalSurface(Operator):
    bl_idname = "soap.minsrf"
    bl_label = "SoapTools: Minimal Surface"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Generate a minimal surface with constraints from a mesh object."
    bl_options = {"REGISTER", "UNDO"}

    suffix: str = "_soaped"

    @classmethod
    def poll(cls, context: Context):
        return True

    def invoke(self, context: Context, event: Event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Context):
        layout = self.layout
        settings = context.scene.soap_settings.minsrf

        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Settings")
        row = box.row()
        row.prop(settings, "apply_after")
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(settings.fixed_verts, "group", text="Pinned")
        right = left.row()
        right.prop(settings.fixed_verts, "strict")
        settings.solver.draw(layout)

    def setup(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.minsrf
        device = context.scene.soap_settings.device.get_device()
        self.solver_config = settings.solver.get_config(device)
        vg = settings.fixed_verts.group
        apply_after = int(settings.apply_after)
        strict_vgs = [vg] if settings.fixed_verts.strict else []

        if vg == "NONE":
            self.report({"ERROR"}, "A vertex group must be selected for constrainst.")
            return {"CANCELLED"}

        new_obj = duplicate_mesh_object(obj, deep=True)
        new_obj.name = f"{new_obj.name}{SOAP_OT_MinimalSurface.suffix}"
        new_obj.data.name = f"{new_obj.data.name}{SOAP_OT_MinimalSurface.suffix}"
        if apply_after > 0:
            link_to_same_scene_collections(obj, new_obj)
            apply_first_n_modifiers(new_obj, apply_after, strict_vgs)
            for coll in new_obj.users_collection:
                coll.objects.unlink(new_obj)

        self.V, self.F = BlendTorch.mesh2tensor(new_obj, device=device)

        _, self.fixed_idx = BlendTorch.vg2tensor(new_obj, vg, device=device)

        self.new_obj = new_obj
        self.original_obj = obj

    def process(self) -> torch.Tensor:
        return solve_minimal_surface(self.solver_config, self.V, self.F, self.fixed_idx)

    def coalesce(self, context: Context):
        new_V = self._result
        BlendTorch.tensor2mesh_update(self.new_obj, new_V)
        link_to_same_scene_collections(self.original_obj, self.new_obj)
        safe_select(self.new_obj)

    def rescind(self, context: Context):
        safe_delete(self.new_obj)
