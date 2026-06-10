"""Constants for the AIKernel.Monolith Python CLI."""

from pathlib import Path

APP_NAME = "aikernel-monolith"
CONTAINER_NAME = "aikernel-monolith"
LINUX_IMAGE = "aikernel/monolith:linux"
WINDOWS_IMAGE = "aikernel/monolith:windows"
BASE_DIR = Path.home() / ".aikernel"
CONFIG_DIR = BASE_DIR / "config"
MODEL_DIR = BASE_DIR / "models"
VFS_DIR = BASE_DIR / "vfs"
CAPABILITY_DIR = BASE_DIR / "capabilities"
