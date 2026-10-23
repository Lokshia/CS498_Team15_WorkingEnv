"""Pure-stdlib source-release tests; no simulator, dependencies, or training."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_release.py"
SPEC = importlib.util.spec_from_file_location("release_packaging_under_test", SCRIPT)
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "source with spaces"
        self.root.mkdir()
        for folder in (*packaging.DIRECTORY_SUFFIXES, "docs"):
            (self.root / folder).mkdir()
        for name in packaging.REQUIRED_FILES:
            self.write(name, f"fixture: {name}\n")
        for name in ("src/lifting_lab/__init__.py", "scripts/bootstrap.ps1",
                     "scripts/bootstrap.sh", "scripts/package_release.py",
                     "tests/test_example.py", "configs/team_protocol.json"):
            self.write(name, f"source: {name}\n")
        self.output = self.base / "release with spaces.zip"

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def rewrite(self, transform):
        with zipfile.ZipFile(self.output) as archive:
            entries = [(entry.filename, archive.read(entry)) for entry in archive.infolist()]
        with zipfile.ZipFile(self.output, "w", zipfile.ZIP_STORED) as archive:
            for name, data in entries:
                archive.writestr(name, transform(name, data))

    def test_clean_release_unique_members_and_original_manifest(self):
        report = packaging.build_release(self.root, self.output)
        self.assertTrue(report["verified"])
        with zipfile.ZipFile(self.output) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(name.startswith("lifting-lab/") for name in names))
            manifest = json.loads(archive.read(packaging.MANIFEST_NAME))
            self.assertEqual(set(names), {"lifting-lab/" + name for name in manifest}
                             | {packaging.MANIFEST_NAME})
            self.assertEqual(report["files"], len(manifest))
            for name, digest in manifest.items():
                self.assertEqual(digest, hashlib.sha256(archive.read("lifting-lab/" + name)).hexdigest())
        self.write("README.md", "Changed locally after building\n")
        self.assertEqual(packaging.verify_release(self.output), report)

    def test_excludes_caches_credentials_runs_old_docs_and_binaries(self):
        excluded = (
            ".env", ".venv/hidden.py", "outputs/result.py", "validation/README.md",
            "docs/QUICKSTART_ZH.md", "docs/extra.md", "docs/secrets.json",
            "src/.env", "src/.env.py", "src/secrets.py", "src/__pycache__/leak.py",
            "src/.pytest_cache/leak.py", "src/nested/outputs/leak.py",
            "scripts/cache/leak.py", "scripts/run.log", "scripts/model.pt",
            "tests/example.pyc", "tests/video.mp4", "configs/credentials.json",
            "configs/private_token.json", "configs/nested/.env.json",
        )
        for name in excluded:
            self.write(name, "must never be shared\n")
        packaging.build_release(self.root, self.output)
        with zipfile.ZipFile(self.output) as archive:
            names = set(archive.namelist())
        for name in excluded:
            self.assertNotIn("lifting-lab/" + name, names)

    def test_does_not_overwrite(self):
        packaging.build_release(self.root, self.output)
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            packaging.build_release(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), original)

    def test_missing_required_document_fails_before_writing(self):
        (self.root / "docs" / "SIM_TO_REAL_ZH.md").unlink()
        with self.assertRaises(FileNotFoundError):
            packaging.build_release(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_payload_change_with_valid_crc_fails_original_hash(self):
        packaging.build_release(self.root, self.output)
        self.rewrite(lambda name, data: b"modified" if name == "lifting-lab/README.md" else data)
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "SHA256 mismatch"):
            packaging.verify_release(self.output)

    def test_crc_damage_is_detected(self):
        packaging.build_release(self.root, self.output)
        self.rewrite(lambda name, data: data)
        with zipfile.ZipFile(self.output) as archive:
            entry = archive.getinfo("lifting-lab/README.md")
        damaged = bytearray(self.output.read_bytes())
        filename_size, extra_size = struct.unpack_from("<HH", damaged, entry.header_offset + 26)
        data_offset = entry.header_offset + 30 + filename_size + extra_size
        damaged[data_offset] ^= 1
        self.output.write_bytes(damaged)
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "CRC"):
            packaging.verify_release(self.output)

    def test_duplicate_archive_member_is_rejected(self):
        packaging.build_release(self.root, self.output)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.output, "a") as archive:
                archive.writestr("lifting-lab/README.md", b"duplicate")
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "Duplicate"):
            packaging.verify_release(self.output)

    def test_case_colliding_members_are_rejected(self):
        packaging.build_release(self.root, self.output)
        with zipfile.ZipFile(self.output, "a") as archive:
            archive.writestr("lifting-lab/TESTS/test_example.py", b"case alias")
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "case-colliding"):
            packaging.verify_release(self.output)

    def test_manifest_duplicate_key_is_rejected(self):
        packaging.build_release(self.root, self.output)
        duplicate = json.dumps({"README.md": "0" * 64})[:-1] + ', "README.md": "' + "0" * 64 + '"}'
        self.rewrite(lambda name, data: duplicate.encode() if name == packaging.MANIFEST_NAME else data)
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "Duplicate manifest key"):
            packaging.verify_release(self.output)

    def test_manifest_missing_member_is_rejected(self):
        packaging.build_release(self.root, self.output)

        def remove_member(name, data):
            if name == packaging.MANIFEST_NAME:
                manifest = json.loads(data)
                del manifest["configs/team_protocol.json"]
                return json.dumps(manifest).encode()
            return data

        self.rewrite(remove_member)
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "members differ"):
            packaging.verify_release(self.output)

    def test_traversal_or_unapproved_member_is_rejected(self):
        packaging.build_release(self.root, self.output)
        with zipfile.ZipFile(self.output, "a") as archive:
            archive.writestr("lifting-lab/../escape.py", b"not source")
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "Unapproved"):
            packaging.verify_release(self.output)

    def test_symlink_escape_is_rejected(self):
        outside = self.base / "outside.py"
        outside.write_text("private", encoding="utf-8")
        try:
            (self.root / "src" / "linked.py").symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Host cannot create a symlink: {exc}")
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "symlink/junction"):
            packaging.build_release(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_windows_junction_reparse_attribute_is_detected(self):
        details = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        with patch.object(Path, "lstat", return_value=details):
            self.assertTrue(packaging._is_reparse_point(self.root / "junction"))

    def test_reparse_directory_is_rejected_without_traversal(self):
        linked = self.root / "src" / "junction"
        linked.mkdir()
        original = packaging._is_reparse_point
        with patch.object(packaging, "_is_reparse_point", side_effect=lambda p: p == linked or original(p)):
            with self.assertRaisesRegex(packaging.ReleaseValidationError, "symlink/junction"):
                packaging.build_release(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_zip_symlink_member_is_rejected(self):
        packaging.build_release(self.root, self.output)
        entry = zipfile.ZipInfo("lifting-lab/src/link.py")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.output, "a") as archive:
            archive.writestr(entry, "../../outside.py")
        with self.assertRaisesRegex(packaging.ReleaseValidationError, "Non-regular"):
            packaging.verify_release(self.output)

    def test_verify_cli_without_output_uses_only_stdlib(self):
        packaging.build_release(self.root, self.output)
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(SCRIPT), "--verify", str(self.output)],
                                cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["verified"])

    def test_verify_cli_reports_corruption_without_third_party_imports(self):
        self.output.write_bytes(b"not a ZIP")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(SCRIPT), "--verify", str(self.output)],
                                cwd=self.base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("damaged release", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
