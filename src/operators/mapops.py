import bpy
import torch

from bpy.types import Context, Event, Operator

from ..utils.blend_data.blendtorch import BlendTorch


class SOAP_OT_OperateMaps(Operator):
    bl_idname = "soap.op_maps"
    bl_label = "SoapTools: Map operations"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Combine different maps using mathematic equations."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def invoke(self, context: Context, event: Event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh with vertex groups")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Context):
        settings = context.scene.soap_settings.mapops
        layout = self.layout
        box = layout.box()
        settings.draw(box)

    def execute(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.mapops
        device = torch.device("cpu")
        vg_name = settings.exp.expression
        W = settings.get_field(obj, device)
        BlendTorch.tensor2vg(obj, vg_name, W)
        return {"FINISHED"}
