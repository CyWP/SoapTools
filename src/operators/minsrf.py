import bpy

from bpy.types import Context

from ..utils.blend_data.scene import (
    duplicate_mesh_object,
    link_to_same_scene_collections,
)
from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.jobs import BackgroundJob
from ..utils.blend_data.mesh_obj import apply_first_n_modifiers
from ..utils.math.problems import solve_minimal_surface


class MESH_OT_MinimalSurface(bpy.types.Operator):
    bl_idname = "soap.minsrf"
    bl_label = "SoapTools: Minimal Surface"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Generate a minimal surface with constraints from a mesh object."
    bl_options = {"REGISTER", "UNDO"}

    suffix: str = "_soaped"

    @classmethod
    def poll(cls, context: Context):
        return True

    def invoke(self, context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Context):
        layout = self.layout
        op_set = context.scene.soap_settings.minsrf

        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Settings")
        row = box.row()
        row.prop(op_set, "apply_after")
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(op_set.fixed_verts, "group", text="Pinned")
        right = left.row()
        right.prop(op_set.fixed_verts, "strict")
        op_set.solver.draw(layout)

    def execute(self, context: Context):
        obj = context.active_object
        op_set = context.scene.soap_settings.minsrf
        device = op_set.solver.get_device()
        config = op_set.solver.get_config()
        vg = op_set.fixed_verts.group
        apply_after = int(op_set.apply_after)
        strict_vgs = [vg] if op_set.fixed_verts.strict else []

        if vg == "NONE":
            self.report({"ERROR"}, "A vertex group must be selected for constrainst.")
            return {"CANCELLED"}

        new_obj = duplicate_mesh_object(obj, deep=True)
        new_obj.name = f"{new_obj.name}{MESH_OT_MinimalSurface.suffix}"
        new_obj.data.name = f"{new_obj.data.name}{MESH_OT_MinimalSurface.suffix}"
        if apply_after > 0:
            link_to_same_scene_collections(obj, new_obj)
            apply_first_n_modifiers(new_obj, apply_after, strict_vgs)
            for coll in new_obj.users_collection:
                coll.objects.unlink(new_obj)

        V, F = BlendTorch.mesh2tensor(new_obj, device=device)

        _, idx = BlendTorch.vg2tensor(new_obj, vg, device=device)

        self.new_obj = new_obj
        self.original_obj = obj
        self._job = BackgroundJob(solve_minimal_surface, config, V, F, idx)

        # Add a timer
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {"RUNNING_MODAL"}

    def modal(self, context: Context, event):
        if event.type == "TIMER" and self._job.is_done():
            new_V = self._job.get_result()
            BlendTorch.tensor2mesh_update(self.new_obj, new_V)
            link_to_same_scene_collections(self.original_obj, self.new_obj)
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            self.report({"INFO"}, "Inflation complete")
            try:
                bpy.ops.object.select_all(action="DESELECT")
            except RuntimeError:
                pass
            finally:
                self.new_obj.select_set(True)
            context.view_layer.objects.active = self.new_obj
            return {"FINISHED"}
        return {"PASS_THROUGH"}

    def cancel(self, context: Context):
        self._job = None
        try:
            context.window_manager.event_timer_remove(self._timer)
        except:
            pass
        self.report({"INFO"}, "Minimal surface computation canceled.")
        return {"CANCELLED"}
