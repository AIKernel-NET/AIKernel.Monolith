"""Configuration generator for AIKernel.Monolith."""

from __future__ import annotations

from . import constants
from .docker_runner import ensure_mounts, print_sample_config


def initialize_config() -> str:
    """Initialize local config directories and copy sample output into a guide file."""
    ensure_mounts()
    result = print_sample_config()
    output = result.stdout if result.returncode == 0 else result.stderr
    target = constants.CONFIG_DIR / "sample-configs.txt"
    target.write_text(output, encoding="utf-8")
    return str(target)
