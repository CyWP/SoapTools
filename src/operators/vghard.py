import bpy

from ..utils.blend_data.vertex_groups import (
    harden_vertex_group,
    get_vertex_group_copy,
    vertex_group_items,
)


class MESH_OT_HardenVGroup(bpy.types.Operator):
    bl_idname = "soap.vghard"
    bl_label = "SoapTools: Harden vertex group"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Keeps local maxima of vertex groups, sets rest to 0."
    bl_options = {"REGISTER", "UNDO"}

    suffix: str = "_hardened"

    vertex_group: bpy.props.EnumProperty(
        name="Vertex Group",
        items=vertex_group_items,
    )  # type: ignore
    copy: bpy.props.BoolProperty(
        name="Create Copy",
        description="Apply hardening to a copy of the vertex group.",
        default=True,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def invoke(self, context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh with vertex groups")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "vertex_group")
        layout.prop(self, "copy")

    def execute(self, context):
        obj = context.active_object
        vg_name: str = self.vertex_group
        do_copy = self.copy

        if vg_name in (None, "NONE"):
            self.report({"ERROR"}, "A vertex group must be selected.")
            return {"CANCELLED"}

        # If copy is True, create a new vertex group
        if do_copy:
            new_name = vg_name + self.suffix
            target_vg = get_vertex_group_copy(obj, vg_name, new_name, caller=self)
        else:
            target_vg = obj.vertex_groups[vg_name]
        harden_vertex_group(obj, target_vg.name)

        return {"FINISHED"}
