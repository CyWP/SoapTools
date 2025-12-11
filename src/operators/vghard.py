import bpy

from bpy.types import Context, Event, Operator

from ..utils.blend_data.vertex_groups import harden_vertex_group, get_vertex_group_copy


class SOAP_OT_HardenVGroup(Operator):
    bl_idname = "soap.vghard"
    bl_label = "SoapTools: Harden vertex group"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Keeps local maxima of vertex groups, sets rest to 0."
    bl_options = {"REGISTER", "UNDO"}

    _suffix = "_hardened"

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
        settings = context.scene.soap_settings.vghard
        layout = self.layout
        box = layout.box()
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(settings, "group", text="")
        right = left.row()
        right.prop(settings, "copy", text="Copy")

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
