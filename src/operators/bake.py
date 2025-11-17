import bpy

from bpy.types import Context

from ..utils.blend_data.mesh_obj import bake_material


class MESH_OT_BakeColor(bpy.types.Operator):
    bl_idname = "soap.bakecolor"
    bl_label = "SoapTools: Bake material color"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Bake a material's color to an image, which can be used to map constraints. When mapped, images are considered to be grayscale."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: Context):
        obj = context.active_object
        bake_material()
        return {"FINISHED"}
