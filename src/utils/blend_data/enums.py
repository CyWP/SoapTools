import bpy

from bpy.types import Context, Object
from typing import List, Tuple, Any, Optional


class BlendEnums:
    """
    Class that stores enum functions when related to the active obejct or some scene context.
    """

    @staticmethod
    def modifiers(
        caller: Any,
        context: Context,
        object: Optional[Object] = None,
        use_active: bool = True,
    ):
        obj = context.active_object if object is None and use_active else object
        if obj and obj.type == "MESH" and obj.modifiers:
            data = [(str(i + 1), mod.name, "") for i, mod in enumerate(obj.modifiers)]
            return [("0", "None", ""), *data]
        return [("0", "None", "")]

    @staticmethod
    def materials(caller: Any, context: Context) -> List[Tuple]:
        items = [("NONE", "None", "")]
        for mat in bpy.data.materials:
            items.append((mat.name, mat.name, f"Material: {mat.name}"))
        return items

    @staticmethod
    def uv_maps(caller: Any, context: Context) -> List[Tuple]:
        obj = context.active_object
        items = [("NONE", "None", "")]
        if obj is None or obj.type != "MESH":
            return items
        for uv in obj.data.uv_layers:
            items.append((uv.name, uv.name, f"UV Map: {uv.name}"))
        return items

    @staticmethod
    def vertex_groups(
        caller: Any,
        context: Context,
        object: Optional[Object] = None,
        use_active: bool = True,
    ):
        obj = context.active_object if object is None and use_active else object
        if obj and obj.type == "MESH" and obj.vertex_groups:
            data = [(vg.name, vg.name, "") for vg in obj.vertex_groups]
            return [("NONE", "None", ""), *data]
        return [("NONE", "None", "")]
