import bpy
from bpy.props import PropertyGroup, EnumProperty


class SolverSettings(PropertyGroup):

    device: EnumProperty()
