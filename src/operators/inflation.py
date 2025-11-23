import bpy
import torch

from bpy.types import Operator, Context

from ..utils.blend_data.scene import (
    duplicate_mesh_object,
    link_to_same_scene_collections,
)
from ..utils.blend_data.blendtorch import BlendTorch
from ..utils.jobs import BackgroundJob
from ..utils.blend_data.mesh_obj import apply_first_n_modifiers
from ..utils.math.problems import solve_flation


class MESH_OT_Inflation(Operator):
    bl_idname = "soap.inflate"
    bl_label = "SoapTools: Flation"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Generate displacement with constraints on a mesh object."
    bl_options = {"REGISTER", "UNDO"}

    suffix: str = "_inflated"

    @classmethod
    def poll(cls, context: Context):
        return True

    def invoke(self, context: Context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Context):
        layout = self.layout
        op_set = context.scene.soap_settings.flation

        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Settings")
        row = box.row()
        left = row.split(factor=0.8)
        left.prop(op_set.fixed_verts, "group", text="Pinned")
        right = left.row()
        right.prop(op_set.fixed_verts, "strict")
        row = box.row()
        row.prop(op_set, "apply_after")
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Constraints")
        row = box.row()
        row.prop(op_set, "active_constraint", expand=True)
        active = op_set.active_constraint
        if active == "DISPLACEMENT":
            op_set.displacement.draw(box, "Displacement")
        elif active == "LAPLACIAN":
            op_set.laplacian.draw(box, "Laplacian")
        elif active == "ALPHA":
            op_set.alpha.draw(box, "Alpha")
        elif active == "BETA":
            op_set.beta.draw(box, "Beta")
        else:
            raise ValueError(f"'{active} is an invalid constraint.")
        op_set.solver.draw(layout)

    def execute(self, context: Context):
        obj = context.active_object
        op_set = context.scene.soap_settings.flation
        device = op_set.solver.get_device()
        solver_config = op_set.solver.get_config()
        vg_fixed = op_set.fixed_verts.group
        vg_fixed_strict = op_set.fixed_verts.strict
        apply_after = int(op_set.apply_after)

        new_obj = duplicate_mesh_object(obj, deep=True)
        new_obj.name = f"{new_obj.name}{MESH_OT_Inflation.suffix}"
        new_obj.data.name = f"{new_obj.data.name}{MESH_OT_Inflation.suffix}"
        if apply_after > 0:
            link_to_same_scene_collections(obj, new_obj)
            apply_first_n_modifiers(
                new_obj,
                apply_after,
                [vg_fixed] if vg_fixed_strict and vg_fixed != "NONE" else [],
            )
            # for coll in new_obj.users_collection:
            #     coll.objects.unlink(new_obj)

        V, F = BlendTorch.mesh2tensor(new_obj, device=device)
        N = BlendTorch.vn2tensor(new_obj, device=device)
        _, fixed_idx = op_set.fixed_verts.get_group(new_obj, device)
        disp_field = op_set.displacement.get_field(new_obj, device)
        laplacian_field = op_set.laplacian.get_field(new_obj, device)
        neg_mask = laplacian_field < 0.0
        laplacian_field[neg_mask] = laplacian_field[neg_mask] / 10000
        alpha_field = op_set.alpha.get_field(new_obj, device)
        beta_field = op_set.beta.get_field(new_obj, device)

        self.new_obj = new_obj
        self.original_obj = obj
        self._job = BackgroundJob(
            solve_flation,
            solver_config,
            V,
            F,
            N,
            target_offset=disp_field,
            fixed_idx=fixed_idx,
            lambda_lap=laplacian_field,
            beta_normal=beta_field,
            alpha_tangent=alpha_field,
        )
        # Add a timer
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {"RUNNING_MODAL"}

    def modal(self, context, event):

        if event.type == "TIMER" and self._job.is_done():
            new_V = self._job.get_result()
            BlendTorch.tensor2mesh_update(self.new_obj, new_V)
            link_to_same_scene_collections(self.original_obj, self.new_obj)
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            self._result = None
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

    def cancel(self, context):
        self._job = None
        try:
            context.window_manager.event_timer_remove(self._timer)
        except:
            pass
        self.report({"INFO"}, "Minimal surface computation canceled.")
        return {"CANCELLED"}
