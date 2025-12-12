import bpy
import os
import subprocess
import sys
import importlib.metadata
from typing import Optional, List, Dict, Set
from pathlib import Path

from .cuda import CUDAHelper
from ..logger import LOGGER

# I only use this for testing/dev.


def get_manifest_path() -> Path:
    addon_dir = Path(__file__).parent
    manifest_dir = addon_dir / "soaptools"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / "installed_packages.json"


class PipHelper:

    INSTALL_RECORD = get_manifest_path()

    @classmethod
    def _load_manifest(cls) -> Set[str]:
        if not cls.INSTALL_RECORD.exists():
            return set()
        return set(
            pkg.strip()
            for pkg in cls.INSTALL_RECORD.read_text().splitlines()
            if pkg.strip()
        )

    @classmethod
    def _save_manifest(cls, pkgs: Set[str]):
        cls.INSTALL_RECORD.write_text("\n".join(sorted(pkgs)))

    @classmethod
    def _record_install(cls, pkg_spec: str):
        pkgs = cls._load_manifest()
        pkgs.add(pkg_spec)
        cls._save_manifest(pkgs)

    @classmethod
    def _record_uninstall(cls, pkg_spec: str):
        pkgs = cls._load_manifest()
        if pkg_spec in pkgs:
            pkgs.remove(pkg_spec)
            cls._save_manifest(pkgs)

    @staticmethod
    def _get_installed_version(package: str) -> Optional[str]:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _get_dependencies(package: str) -> List[str]:
        try:
            meta = importlib.metadata.metadata(package)
        except importlib.metadata.PackageNotFoundError:
            return []
        requires = meta.get_all("Requires") or []
        parsed = []
        for req in requires:
            parsed.append(req.split()[0])
        return parsed

    @staticmethod
    def _all_reverse_deps() -> Dict[str, Set[str]]:
        reverse = {}
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            reqs = dist.metadata.get_all("Requires") or []
            for req in reqs:
                dep = req.split()[0]
                reverse.setdefault(dep, set()).add(name)
        return reverse

    @classmethod
    def install(cls, package: str, version: Optional[str] = None, force: bool = False):
        installed_version = cls._get_installed_version(package)
        pkg_spec = f"{package}=={version}" if version else package

        if installed_version is None:
            LOGGER.debug(f"[PipHelper] Installing {pkg_spec}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])
            cls._record_install(pkg_spec)
            return

        if version and installed_version == version:
            LOGGER.debug(f"[PipHelper] {pkg_spec} already installed")
            return

        if force or (version and installed_version != version):
            LOGGER.debug(f"[PipHelper] Reinstalling {pkg_spec}")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    pkg_spec,
                ]
            )
            cls._record_install(pkg_spec)
            return

        LOGGER.debug(f"[PipHelper] {package} already installed → skipping")

    @classmethod
    def uninstall(cls, package: str, auto_prune: bool = True):
        manifest = cls._load_manifest()
        pkg_specs = [p for p in manifest if p.split("==")[0] == package]

        if not pkg_specs:
            LOGGER.warning(
                f"[PipHelper] {package} not uninstalled: not installed by SoapTools"
            )
            return

        pkg_spec = pkg_specs[0]
        LOGGER.debug(f"[PipHelper] Uninstalling {pkg_spec}")

        deps_before = set(cls._get_dependencies(package))

        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", package])
        cls._record_uninstall(pkg_spec)

        if not auto_prune:
            return

        reverse = cls._all_reverse_deps()

        manifest = cls._load_manifest()
        prune_list = []

        for dep in deps_before:
            matching = [p for p in manifest if p.split("==")[0] == dep]
            if not matching:
                continue
            if dep not in reverse or len(reverse[dep]) == 0:
                prune_list.append((dep, matching[0]))

        if prune_list:
            for dep, spec in prune_list:
                LOGGER.debug(f"[PipHelper] Pruning dependency {dep}")
                subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", dep])
                cls._record_uninstall(spec)

        LOGGER.debug("[PipHelper] Uninstall complete.")

    @staticmethod
    def clean_installed():
        if not RECORD_FILE.exists():
            LOGGER.debug("No recorded installs.")
            return

        with open(RECORD_FILE, "r") as f:
            installed = json.load(f)

        if not installed:
            LOGGER.debug("No recorded installs.")
            return

        for pkg in installed:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "uninstall", "-y", pkg]
                )
                LOGGER.debug(f"Uninstalled: {pkg}")
            except subprocess.CalledProcessError:
                LOGGER.debug(f"Failed to uninstall: {pkg}")

        RECORD_FILE.unlink(missing_ok=True)
        LOGGER.debug("Clean complete.")
