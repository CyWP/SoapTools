import bpy

from bpy.types import Context, Operator


class SOAP_OT_ChangeDevice(Operator):
    bl_idname = "soap.device"
    bl_label = "SoapTools: Change Default Device"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Chnage default device for applicable computation."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: Context):
        return context.scene.soap_settings.device.has_options()

    def draw(self, context: Context):
        settings = context.scene.soap_settings
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Select Device")
        row = layout.row()
        row.prop(settings, "device")

    def execute(self, context: Context):
        return {"FINISHED"}
