import bpy
import torch

from bpy.types import Context, Operator

from ..utils.blend_data.blendtorch import BlendTorch


class MESH_OT_RemapVGroup(Operator):
    bl_idname = "soap.vgremap"
    bl_label = "SoapTools: Remap Vertex Group"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Remap vertex group valeus using predefined stackable functions."
    bl_options = {"REGISTER", "UNDO"}

    _suffix = "_remapped"

    @classmethod
    def poll(cls, context: Context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def invoke(self, context: Context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh with vertex groups")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Context):
        settings = context.scene.soap_settings.vgremap
        layout = self.layout
        box = layout.box()
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(settings.group, "group", text="")
        right = left.row()
        right.prop(settings.group, "strict")
        settings.remap.draw(layout, compact=False)

    def execute(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.vgremap
        device = torch.device("cpu")
        og_name = settings.group.group
        W, idx = settings.group.get_group(obj, device, none_valid=False)
        W = settings.remap.process(W)
        BlendTorch.tensor2vg(obj, f"{og_name}{MESH_OT_RemapVGroup._suffix}", W)
        return {"FINISHED"}
