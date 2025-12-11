import bpy
import torch

from bpy.props import StringProperty
from bpy.types import PropertyGroup
from typing import Dict, Union

from ..utils.math.symbolic import Parser, TorchParser


class SymbolicExpression(PropertyGroup):
    expression: StringProperty(
        name="Expression",
        description="Mathematical expression used for mapping",
        default="",
    )  # type:ignore

    def eval(
        self, vars: Dict[str, Union[float, torch.Tensor]], tensor: bool = True
    ) -> Union[float, torch.Tensor]:
        if tensor:
            return TorchParser().compute(self.expression, vars=vars)
        return Parser().compute(self.expression, vars=vars)
