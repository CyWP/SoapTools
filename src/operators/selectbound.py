import bpy

from bpy.types import Context

from ..utils.blend_data.mesh_obj import select_boundary


class MESH_OT_SelectBoundary(bpy.types.Operator):
    bl_idname = "soap.selectbound"
    bl_label = "SoapTools: Select Boundary"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Reduces current mesh's vertex selection to its boundary."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: Context):
        obj = context.active_object
        select_boundary(obj)
        return {"FINISHED"}
