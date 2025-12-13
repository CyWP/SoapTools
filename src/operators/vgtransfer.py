import bpy
import torch

from bpy.types import Context, Operator, Event

from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.scene import temp_copy


class SOAP_OT_TransferVGroup(Operator):
    bl_idname = "soap.vgtransfer"
    bl_label = "SoapTools: Transfer Vertex Group"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Remap vertex group values using predefined stackable functions."
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
        settings = context.scene.soap_settings.vgtransfer
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.prop(settings, "target")
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(settings.group, "group")
        right = left.row()
        right.prop(settings.group, "strict")
        row = box.row()
        row.prop(settings, "apply_after")

    def execute(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.vgtransfer
        device = context.scene.soap_settings.device.get_device()
        name = settings.group.group
        strict_vgs = [name] if settings.group.strict else []
        apply_after = int(settings.apply_after)
        W = None
        if apply_after > 0:
            with temp_copy(obj, apply_after=apply_after, strict_vgs=strict_vgs) as tmp:
                W, _ = settings.group.get_group(tmp, device, none_valid=False)
        else:
            W, _ = settings.group.get_group(obj, device, none_valid=False)
        BlendTorch.tensor2vg(settings.target, name, W)
        return {"FINISHED"}
