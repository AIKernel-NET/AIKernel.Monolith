import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aikernel_monolith import config_generator, constants
from aikernel_monolith.docker_runner import select_image


class CliHelpersTests(unittest.TestCase):
    def test_select_image_uses_linux_container(self):
        self.assertEqual(select_image(), "aikernel/monolith:linux")

    def test_init_writes_a_single_guide_to_the_config_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounts = {name.upper() + "_DIR": root / name for name in ("config", "model", "vfs", "capability")}
            output = subprocess.CompletedProcess([], 0, stdout="--- sample-routing.yaml ---\n")
            with patch.multiple(constants, **mounts), patch.object(config_generator, "print_sample_config", return_value=output):
                path = Path(config_generator.initialize_config())
            self.assertEqual(path, root / "config" / "sample-configs.txt")
            self.assertEqual(path.read_text(encoding="utf-8"), output.stdout)
            self.assertEqual(list(path.parent.iterdir()), [path])
            self.assertTrue(all(mount.is_dir() for mount in mounts.values()))
