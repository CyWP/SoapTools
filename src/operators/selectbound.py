import bpy

from bpy.types import Context

from ..utils.blend_data.mesh_obj import select_boundary


class MESH_OT_HardenVGroup(bpy.types.Operator):
    bl_idname = "soap.selectbound"
    bl_label = "SoapTools: Select Boundary"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Reduces current mesh selection to its boundary."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.vghard
        vg_name: str = settings.group
        do_copy = settings.copy

        if vg_name in (None, "NONE"):
            self.report({"ERROR"}, "A vertex group must be selected.")
            return {"CANCELLED"}

        # If copy is True, create a new vertex group
        if do_copy:
            new_name = vg_name + self._suffix
            target_vg = get_vertex_group_copy(obj, vg_name, new_name, caller=self)
        else:
            target_vg = obj.vertex_groups[vg_name]
        harden_vertex_group(obj, target_vg.name)

        return {"FINISHED"}
