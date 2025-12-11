import bpy

from bpy.types import Context, Event, Operator


class SOAP_OT_ImageToVG(Operator):
    bl_idname = "soap.img2vg"
    bl_label = "SoapTools: Image to vertex group"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Project an Image to a vertex group using a UV map."
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
        layout = self.layout
        settings = context.scene.soap_settings.imgmap
        settings.draw(layout)

    def execute(self, context: Context):
        obj = context.active_object
        device = context.scene.soap_settings.device.get_device()
        settings = context.scene.soap_settings.imgmap
        settings.create_vertex_group(obj, device)
        return {"FINISHED"}
