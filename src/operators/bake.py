import bpy
import torch

from bpy.types import Context


class MESH_OT_BakeChannel(bpy.types.Operator):
    bl_idname = "soap.bake"
    bl_label = "SoapTools: Bake material channel"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Bake one of a material's output channels."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def invoke(self, context: Context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh with vertex groups")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.bake
        settings.get_baked(obj, pack=True)
        return {"FINISHED"}

    def draw(self, context: Context):
        layout = self.layout
        settings = context.scene.soap_settings.bake
        settings.draw(layout)
