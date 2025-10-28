import bpy
import multiprocessing as mp

from ..utils.blend_data import duplicate_mesh_object, link_to_same_scene_collections
from ..utils.bridges import mesh2np, vg2np
from ..utils.dec import solve_minimal_surface_with_fixed
from ..utils.mesh_obj import apply_first_n_modifiers, update_mesh_vertices


def vertex_group_items(caller, context):
    obj = context.active_object
    if obj and obj.type == "MESH" and obj.vertex_groups:
        return [(vg.name, vg.name, "") for vg in obj.vertex_groups]
    return [("NONE", "None", "")]


def modifier_items(caller, context):
    obj = context.active_object
    opts = []
    if obj and obj.type == "MESH" and obj.modifiers:
        opts = [(str(i + 1), mod.name, "") for i, mod in enumerate(obj.modifiers)]
    return [("0", "None", ""), *opts]


class MESH_OT_MinimalSurface(bpy.types.Operator):
    bl_idname = "soap.minsrf"
    bl_label = "SoapTools: Minimal Surface"
    bl_icon = "NODE_MATERIAL"
    bl_description = "Generate a minimal surface with constraints from a mesh object."
    bl_options = {"REGISTER", "UNDO"}

    suffix: str = "_soaped"

    vertex_group: bpy.props.EnumProperty(
        name="Pinned Vertices",
        items=vertex_group_items,  # type: ignore
        default=0,
    )

    modifier: bpy.props.EnumProperty(
        name="Apply After",
        items=modifier_items,  # type: ignore
        default=0,
    )

    preserve: bpy.props.BoolProperty(
        name="Harden Vertex group",
        description="Avoids vertex group weights to smooth and propagate after a subdivision pass.",
        default=True,  # type: ignore
    )

    @classmethod
    def poll(cls, context):
        return True

    def invoke(self, context, event):
        if not self.poll(context):
            self.report({"ERROR"}, "Active object must be a mesh")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "vertex_group")
        layout.prop(self, "modifier")
        layout.prop(self, "preserve")

    def execute(self, context):
        obj = context.active_object
        vg = self.vertex_group
        mod = self.modifier
        preserve = self.preserve

        if vg == "NONE":
            self.report({"ERROR"}, "A vertex group must be selected for constrainst.")
            return {"CANCELLED"}
        try:
            mod = int(mod)
        except:
            mod = 0

        new_obj = duplicate_mesh_object(obj, deep=True)
        new_obj.name = f"{new_obj.name}{MESH_OT_MinimalSurface.suffix}"
        new_obj.data.name = f"{new_obj.data.name}{MESH_OT_MinimalSurface.suffix}"
        if mod > 0:
            link_to_same_scene_collections(obj, new_obj)
            apply_first_n_modifiers(new_obj, mod, vg, preserve_vg=preserve)
            for coll in new_obj.users_collection:
                coll.objects.unlink(new_obj)

        V, F = mesh2np(new_obj)

        _, idx = vg2np(new_obj, vg)

        # Launch worker process
        self.q = mp.Queue()
        self.p = mp.Process(
            target=lambda V, F, idx, q: q.put(
                solve_minimal_surface_with_fixed(V, F, idx)
            ),
            args=(V, F, idx, self.q),
        )
        self.p.start()
        self.new_obj = new_obj
        self.original_obj = obj

        # Add a timer
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "TIMER":
            if not self.q.empty():
                new_V = self.q.get()
                update_mesh_vertices(self.new_obj, new_V)
                link_to_same_scene_collections(self.original_obj, self.new_obj)
                self.p.join(timeout=1)
                context.window_manager.event_timer_remove(self._timer)
                self.report({"INFO"}, "Minimal surface computation complete.")
                try:
                    bpy.ops.object.select_all(action="DESELECT")
                    self.new_obj.select_set(True)
                except RuntimeError:
                    pass
                context.view_layer.objects.active = self.new_obj
                return {"FINISHED"}
        return {"PASS_THROUGH"}

    def cancel(self, context):
        try:
            self.p.terminate()
        except Exception:
            pass
        context.window_manager.event_timer_remove(self._timer)
        self.report({"INFO"}, "Minimal surface computation canceled.")
        return {"CANCELLED"}
