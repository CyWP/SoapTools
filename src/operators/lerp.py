import bpy
import torch

from bpy.types import Context, Operator

from ..utils.blend_data.scene import (
    duplicate_mesh_object,
    link_to_same_scene_collections,
    transfer_object_state,
)
from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.mesh_obj import (
    apply_first_n_modifiers,
    safe_delete,
    safe_select,
)


class SOAP_OT_Interpolate(Operator):
    bl_idname = "soap.lerp"
    bl_label = "SoapTools: Interpolate Meshes"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Transform the base mesh into its transformed one."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        return True

    def invoke(self, context: Context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh with vertex groups")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Context):
        settings = context.scene.soap_settings.lerp
        layout = self.layout
        box = layout.box()
        settings.draw(box)

    def execute(self, context: Context):
        settings = context.scene.soap_settings.lerp
        device = context.scene.soap_settings.device.get_device()
        V, F = settings.get_field(device)
        name = "soap_lerp_gen"
        new_obj = settings.targets[0].get_mesh()
        parent = settings.reference
        BlendTorch.tensor2mesh_update(new_obj, V)
        if parent is not None:
            transfer_object_state(parent, new_obj)
            link_to_same_scene_collections(parent, new_obj)
        if not new_obj.users_collection:
            lc = context.view_layer.active_layer_collection
            lc.collection.objects.link(new_obj)
        safe_select(new_obj)
        return {"FINISHED"}
