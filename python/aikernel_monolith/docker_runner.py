"""Docker runner for the AIKernel.Monolith container."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from . import constants


def select_image() -> str:
    """Return the platform-specific monolith image."""
    return constants.WINDOWS_IMAGE if platform.system().lower().startswith("win") else constants.LINUX_IMAGE


def ensure_mounts() -> list[Path]:
    """Create local external mount directories for the monolith CLI."""
    paths = [constants.CONFIG_DIR, constants.MODEL_DIR, constants.VFS_DIR, constants.CAPABILITY_DIR]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_container(detach: bool = True) -> subprocess.CompletedProcess[str]:
    """Run the monolith container with external mounts."""
    ensure_mounts()
    command = [
        "docker",
        "run",
        "--name",
        constants.CONTAINER_NAME,
        "--rm",
        "-p",
        "8080:8080",
        "-v",
        f"{constants.CONFIG_DIR}:/config",
        "-v",
        f"{constants.MODEL_DIR}:/models",
        "-v",
        f"{constants.VFS_DIR}:/vfs",
        "-v",
        f"{constants.CAPABILITY_DIR}:/capabilities",
        "-e",
        "AIKERNEL_CONFIG_ROOT=/config",
        "-e",
        "AIKERNEL_MODEL_ROOT=/models",
        "-e",
        "AIKERNEL_VFS_ROOT=/vfs",
        "-e",
        "AIKERNEL_CAPABILITY_ROOT=/capabilities",
    ]
    if detach:
        command.append("-d")
    command.append(select_image())
    return subprocess.run(command, check=False, text=True, capture_output=True)


def stop_container() -> subprocess.CompletedProcess[str]:
    """Stop the running monolith container."""
    return subprocess.run(["docker", "stop", constants.CONTAINER_NAME], check=False, text=True, capture_output=True)


def inspect_container() -> subprocess.CompletedProcess[str]:
    """Inspect whether the monolith container is running."""
    return subprocess.run(["docker", "ps", "--filter", f"name={constants.CONTAINER_NAME}", "--format", "{{.Names}}\t{{.Status}}"], check=False, text=True, capture_output=True)


def print_sample_config() -> subprocess.CompletedProcess[str]:
    """Print sample configuration from the selected monolith image."""
    return subprocess.run(["docker", "run", "--rm", select_image(), "--print-sample-config"], check=False, text=True, capture_output=True)
