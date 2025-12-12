import bpy
import torch

from bpy.types import Context, Operator

from ..utils.blend_data.scene import (
    duplicate_mesh_object,
    link_to_same_scene_collections,
)
from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.blend_data.mesh_obj import (
    apply_first_n_modifiers,
    safe_delete,
    safe_select,
)
from ..utils.blend_data.operators import process_operator


@process_operator
class SOAP_OT_Interpolate(Operator):
    bl_idname = "soap.lerp"
    bl_label = "SoapTools: Interpolate Meshes"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Transform the base mesh into its transformed one."
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

    def draw(self, context: Context):
        settings = context.scene.soap_settings.lerp
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Settings")
        row = box.row()
        row.prop(settings, "target", text="Target")
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(settings.fixed_verts, "group", text="Pinned")
        right = left.row()
        right.prop(settings.fixed_verts, "strict")
        row = box.row()
        row.prop(settings, "apply_after")
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Interpolation Weights")
        row = box.row()
        settings.weights_map.draw(box, "Interpolation weights")

    def setup(self, context: Context):
        settings = context.scene.soap_settings.lerp
        src_obj = context.active_object
        tgt_obj = settings.target
        device = context.scene.soap_settings.device.get_device()
        vg_fixed = settings.fixed_verts.group
        vg_fixed_strict = settings.fixed_verts.strict
        apply_after = int(settings.apply_after)

        new_obj = duplicate_mesh_object(src_obj, deep=True)
        new_obj.name = f"{new_obj.name}_lerp_{tgt_obj.name}"
        new_obj.data.name = f"{new_obj.data.name}_lerp_{tgt_obj.data.name}"
        if apply_after > 0:
            link_to_same_scene_collections(src_obj, new_obj)
            apply_first_n_modifiers(
                new_obj,
                apply_after,
                [vg_fixed] if vg_fixed_strict and vg_fixed != "NONE" else [],
            )

        self.new_obj = new_obj
        self.src_obj = src_obj
        self.src_V, _ = BlendTorch.mesh2tensor(new_obj, device=device)
        self.tgt_V, _ = BlendTorch.mesh2tensor(tgt_obj, device=device)
        if settings.fixed_verts.group == "NONE":
            self.fixed_idx = torch.tensor([], device=device, dtype=torch.long)
        else:
            _, self.fixed_idx = settings.fixed_verts.get_group(new_obj, device)

        if self.src_V.shape != self.tgt_V.shape:
            raise Exception("Both objects must have the same amount of vertices.")

        self.W = settings.weights_map.get_field(new_obj, device).unsqueeze(1)

    def process(self) -> torch.Tensor:
        src_mean = self.src_V.mean(dim=0).unsqueeze(0)
        tgt_mean = self.tgt_V.mean(dim=0).unsqueeze(0)
        new_V = (
            self.W * (self.tgt_V - tgt_mean)
            + (1 - self.W) * (self.src_V - src_mean)
            + src_mean
        )
        new_V[self.fixed_idx] = self.src_V[self.fixed_idx]
        return new_V

    def coalesce(self, context: Context):
        new_V = self._result
        BlendTorch.tensor2mesh_update(self.new_obj, new_V)
        link_to_same_scene_collections(self.src_obj, self.new_obj)
        safe_select(self.new_obj)

    def rescind(self, context: Context):
        safe_delete(self.new_obj)
