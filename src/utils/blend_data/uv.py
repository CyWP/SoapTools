import bpy

from bpy.types import Context
from typing import List, Tuple


def uv_map_items(self, context: Context) -> List[Tuple]:
    """Return a list of UV maps on the active object for an EnumProperty."""
    obj = context.active_object
    items = [("NONE", "None", "")]
    if obj is None or obj.type != "MESH":
        return items
    for uv in obj.data.uv_layers:
        items.append((uv.name, uv.name, f"UV Map: {uv.name}"))
    return items
