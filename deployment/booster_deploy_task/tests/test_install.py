import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('team15_task_installer', ROOT / 'install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def test_copy_preserves_upstream_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'deploy'; source = Path(temporary)/'task'
            for name in ('scripts/deploy.py', 'booster_deploy/robots/k1.py',
                         'booster_deploy/controllers/base_controller.py',
                         'booster_deploy/controllers/booster_robot_controller.py',
                         'booster_deploy/controllers/controller_cfg.py', 'booster_deploy/utils/registry.py'):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'# original\n')
            (root/'tasks').mkdir()
            (source/'models').mkdir(parents=True)
            (source/'__init__.py').write_bytes(b'# task\n')
            (source/'models/proprio_hold.onnx').write_bytes(b'fake model for copy test')
            (source/'__pycache__').mkdir()
            (source/'__pycache__/cached.pyc').write_bytes(b'no')
            receipt, dest = installer.install(root, source)
            self.assertTrue(receipt['upstream_source_unchanged'])
            self.assertFalse((dest/'__pycache__').exists())
            self.assertEqual((dest/'__init__.py').read_bytes(), b'# task\n')
            with self.assertRaises(FileExistsError):
                installer.install(root, source)

    def test_wrong_root_fails_without_creating_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                installer.install(root, root)
            self.assertFalse((root/'tasks/team15_hold').exists())


if __name__ == '__main__':
    unittest.main()
