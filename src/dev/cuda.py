import importlib
import re
import subprocess
import sys

from typing import Tuple, Optional

from ..utils.singleton import Singleton
from ..logger import LOGGER

CUDA_WHEEL_MAP = {
    (12, 8): "https://download.pytorch.org/whl/cu128",
    (12, 4): "https://download.pytorch.org/whl/cu124",
    (11, 8): "https://download.pytorch.org/whl/cu118",
}


class CUDAHelper(Singleton):

    def initialize(
        self,
        torch_version: Optional[Tuple[int]] = None,
        cuda_min: Tuple[int] = (11, 8),
    ):
        self.cuda_min = cuda_min
        self.torch_version = (
            self.get_torch_version() if torch_version == None else torch_version
        )
        if self.torch_version is None:
            raise EnvironmentError(
                "No PyTorch installation detected and no installation version was specified."
            )
        self.cuda_version = self.detect_cuda_version()
        self.torch_cpu = not self.has_torch_cuda()
        self.cuda_valid = self.has_valid_cuda()

    def upgrade_eligible(self) -> bool:
        is_eligible = self.torch_cpu and self.cuda_valid
        LOGGER.debug(f"Is eligible for upgrade to CUDA: {is_eligible}.")
        return is_eligible

    def detect_cuda_version(self) -> Optional[Tuple[int]]:
        try:
            out = subprocess.check_output(["nvidia-smi"], text=True)
            match = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", out)
            if match:
                major, minor = map(int, match.groups())
                LOGGER.debug(f"Cuda version detected: {major}.{minor}.")
                return major, minor
        except Exception:
            LOGGER.debug("No cuda installation detected on system.")
            return None

    def get_torch_version(self) -> Optional[Tuple[int, int, int]]:
        if importlib.util.find_spec("torch") is None:
            return None

        try:
            import torch

            version_str = torch.__version__.split("+")[0]
            parts = version_str.split(".")
            version = tuple(int(p) for p in parts[:3])
            LOGGER.debug(f"Detected torch version: {version}.")
            return version
        except Exception:
            LOGGER.debug("No torch installation detected.")
            return None

    def has_valid_cuda(self) -> bool:
        version = self.detect_cuda_version()
        if version is None:
            return False
        major, minor = version
        minmaj, minmin = self.cuda_min
        if major < minmaj or (major == minmaj and minor < minmin):
            LOGGER.debug(
                f"Currently installed cuda version {major}.{minor} is too old. Current minimum is 11.8."
            )
            return False
        return True

    def has_torch_cuda(self) -> bool:
        if importlib.util.find_spec("torch") is None:
            LOGGER.debug("PyTorch is not installed")
            return False
        try:
            import torch

            if torch.version.cuda is None:
                LOGGER.debug(
                    f"PyTorch installed (v{torch.__version__}) but no CUDA support"
                )
                return False
            LOGGER.debug(
                f"PyTorch installed (v{torch.__version__}) with CUDA {torch.version.cuda}"
            )
            return True
        except Exception as e:
            LOGGER.debug(f"Failed to check PyTorch CUDA support: {e}")
            return False

    def pick_best_cuda_repo(self) -> Optional[str]:
        if self.cuda_version is None:
            return None

        major, minor = self.cuda_version
        for (maj, min_), url in sorted(CUDA_WHEEL_MAP.items(), reverse=True):
            if (major, minor) >= (maj, min_):
                return url
        return None

    def install_cuda_torch(self):
        repo = self.pick_best_cuda_repo()
        if repo is None:
            raise Exception(f"No valid PyTorch wheel for CUDA: {self.cuda_version}.")
        pybin = sys.executable
        torch_ver_str = ".".join(str(v) for v in self.torch_version)
        subprocess.check_call(
            [
                pybin,
                "-m",
                "pip",
                "install",
                f"torch=={torch_ver_str}",
                "--force-reinstall",
                "--index-url",
                repo,
            ]
        )
