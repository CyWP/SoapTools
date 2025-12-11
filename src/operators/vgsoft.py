import bpy

from bpy.types import Context, Operator

from ..utils.blend_data.vertex_groups import (
    soften_vertex_group_inwards,
    soften_vertex_group_outwards,
    get_vertex_group_copy,
)


class SOAP_OT_SoftenVGroup(Operator):
    bl_idname = "soap.vgsoft"
    bl_label = "SoapTools: Soften vertex group"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Softly expand vertex group by a defined number of rings."
    bl_options = {"REGISTER", "UNDO"}

    _suffix = "_softened"

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
        settings = context.scene.soap_settings.vgsoft
        layout = self.layout
        box = layout.box()
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(settings, "group", text="")
        right = left.row()
        right.prop(settings, "copy", text="Copy")
        row = box.row()
        row.prop(settings, "rings")
        row.prop(settings, "direction", expand=True)

    def execute(self, context: Context):
        obj = context.active_object
        settings = context.scene.soap_settings.vgsoft
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
        if settings.direction == "IN":
            soften_vertex_group_inwards(obj, target_vg.name, settings.rings)
        else:
            soften_vertex_group_outwards(obj, target_vg.name, settings.rings)

        return {"FINISHED"}
