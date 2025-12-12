import bpy

from bpy.types import Context, Operator

from ..dev.cuda import CUDAHelper
from ..utils.blend_data.operators import process_operator
from ..logger import LOGGER


@process_operator
class SOAP_OT_CudaTorch(Operator):
    """Detect CUDA and install the matching PyTorch wheel"""

    bl_idname = "soap.cudatorch"
    bl_label = "Install CUDA PyTorch for GPU computation"
    bl_description = "Replace current PyTorch installation with a CUDA-enabled one. This may take a few minutes."
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        return CUDAHelper().upgrade_eligible()

    def setup(self, context: Context):
        pass

    def process(self):
        LOGGER.info(
            f"Installing a cuda-enabled version of PyTorch. This may take a few minutes."
        )
        CUDAHelper().install_cuda_torch()

    def coalesce(self, context: Context):
        LOGGER.info(
            "PyTorch installation successful. Restart Blender to enact changes."
        )

    def rescind(self, context: Context):
        pass
