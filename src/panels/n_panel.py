import bpy

from bpy.types import Panel, Context

MESH_OPERATORS = {
    "soap.minsrf": "Minimal Surface",
    "soap.inflate": "Flation",
    "soap.lerp": "Interpolate",
}

MAP_OPERATORS = {
    "soap.bake": "Bake Material",
    "soap.img2vg": "Image to Vertex Group",
    "soap.vghard": "Harden Vertex Group",
    "soap.vgsoft": "Soften Vertex Group",
    "soap.vgtransfer": "Transfer Vertex Group",
    "soap.vgremap": "Remap Vertex Group",
    "soap.op_maps": "Map Operations",
}

MISC_OPERATORS = {"soap.selectbound": "Select Boundary"}


class VIEW3D_PT_NPanel(Panel):
    bl_label = "SoapTools"
    bl_idname = "VIEW3D_PT_NPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Soap"

    def draw(self, context: Context):
        settings = context.scene.soap_settings
        layout = self.layout
        settings.device.draw(layout)


class VIEW3D_PT_MeshOps(Panel):
    bl_label = "Mesh Operations"
    bl_idname = "VIEW3D_PT_MeshOps"
    bl_parent_id = "VIEW3D_PT_NPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Soap"

    def draw(self, context: Context):
        layout = self.layout
        for op_id, op_name in MESH_OPERATORS.items():
            row = layout.row()
            row.operator(op_id, text=op_name)


class VIEW3D_PT_MapOps(Panel):
    bl_label = "Map Operations"
    bl_idname = "VIEW3D_PT_MapOps"
    bl_parent_id = "VIEW3D_PT_NPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Soap"

    def draw(self, context: Context):
        layout = self.layout
        for op_id, op_name in MAP_OPERATORS.items():
            row = layout.row()
            row.operator(op_id, text=op_name)
