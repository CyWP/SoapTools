from .cuda import CUDAHelper
from .pip import PipHelper

from ..logger import LOGGER


def check_deps() -> bool:
    try:
        import torch
        import scipy
    except:
        return False
    return True


def install_deps_dev():
    LOGGER.debug("Installing Dev dependencies...")
    CUDAHelper(torch_version=(2, 6)).install_cuda_torch()
    PipHelper.install("scipy")
