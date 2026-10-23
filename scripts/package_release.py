"""Build or verify a portable, explicitly allowlisted source-only release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile
import zlib


PREFIX = "lifting-lab/"
MANIFEST_NAME = PREFIX + "RELEASE_MANIFEST.json"
ROOT_FILES = (
    "pyproject.toml", "uv.lock", ".python-version", ".gitignore",
    "README.md", "THIRD_PARTY_NOTICES.md",
)
DIRECTORY_SUFFIXES = {
    "src": frozenset({".py"}),
    "scripts": frozenset({".py", ".ps1", ".sh"}),
    "tests": frozenset({".py"}),
    "configs": frozenset({".json"}),
}
DOC_FILES = ("ENVIRONMENT_DESIGN.md", "TEAM_PROTOCOL.md", "SIM_TO_REAL_ZH.md")
REQUIRED_FILES = frozenset(ROOT_FILES) | frozenset("docs/" + name for name in DOC_FILES)
EXCLUDED_NAMES = frozenset({
    "__pycache__", "venv", "cache", "caches", "build", "dist", "node_modules",
    "validation", "output", "outputs", "log", "logs", "run", "runs",
    "checkpoints", "wandb", "vendor", "assets",
})
SENSITIVE_WORDS = frozenset({"credential", "credentials", "secret", "secrets", "password", "passwords", "token", "tokens"})


class ReleaseValidationError(ValueError):
    """A release does not satisfy its manifest or portable-path contract."""


def _excluded_name(name: str) -> bool:
    lowered = name.lower()
    words = set(re.split(r"[^a-z0-9]+", lowered))
    return (lowered.startswith(".") or lowered in EXCLUDED_NAMES
            or Path(lowered).stem in EXCLUDED_NAMES
            or bool(words & SENSITIVE_WORDS))


def _allowed_relative(name: str) -> bool:
    # Reject Windows separators, alternate data streams, traversal, and aliases.
    if not isinstance(name, str) or "\\" in name or ":" in name or "\x00" in name:
        return False
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    if name in REQUIRED_FILES:
        return True
    return (len(parts) >= 2 and parts[0] in DIRECTORY_SUFFIXES
            and not any(_excluded_name(part) for part in parts[1:])
            and Path(parts[-1]).suffix in DIRECTORY_SUFFIXES[parts[0]])


def _is_reparse_point(path: Path) -> bool:
    details = path.lstat()
    return (stat.S_ISLNK(details.st_mode)
            or bool(getattr(details, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def _checked_path(path: Path, root: Path, *, directory: bool = False) -> Path:
    if _is_reparse_point(path):
        raise ReleaseValidationError(f"Refusing symlink/junction/reparse point: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ReleaseValidationError(f"Path escapes the source root: {path}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ReleaseValidationError(f"Expected {'directory' if directory else 'regular file'}: {path}")
    return path


def collect_release_files(root: Path) -> list[Path]:
    """Return only approved source files; never follow source links or junctions."""
    root = Path(root)
    if _is_reparse_point(root):
        raise ReleaseValidationError(f"Refusing linked source root: {root}")
    root = root.resolve(strict=True)
    files = [_checked_path(root / name, root) for name in ROOT_FILES]

    def visit(directory: Path) -> None:
        _checked_path(directory, root, directory=True)
        for path in sorted(directory.iterdir()):
            # Check before filtering: do not inspect a linked target, even if its
            # name looks like a cache or an otherwise ignored file.
            if _is_reparse_point(path):
                raise ReleaseValidationError(f"Refusing symlink/junction/reparse point: {path}")
            if _excluded_name(path.name):
                continue
            if path.is_dir():
                visit(path)
            elif _allowed_relative(path.relative_to(root).as_posix()):
                files.append(_checked_path(path, root))

    for folder in DIRECTORY_SUFFIXES:
        visit(root / folder)
    _checked_path(root / "docs", root, directory=True)
    files.extend(_checked_path(root / "docs" / name, root) for name in DOC_FILES)
    names = [path.relative_to(root).as_posix() for path in files]
    if len({name.casefold() for name in names}) != len(names):
        raise ReleaseValidationError("Source paths collide on a case-insensitive filesystem")
    return sorted(files)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseValidationError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def verify_release(path: Path) -> dict:
    """Verify the stored manifest, not the current source tree. Extract nothing.

    These checks detect accidental damage; an embedded manifest is not an
    authenticity signature against someone who can replace the entire ZIP.
    """
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(set(names)) != len(names) or len({name.casefold() for name in names}) != len(names):
                raise ReleaseValidationError("Duplicate or case-colliding ZIP members")
            if MANIFEST_NAME not in names:
                raise ReleaseValidationError("Release manifest is missing")
            for entry in entries:
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if entry.is_dir() or kind not in (0, stat.S_IFREG) or (entry.external_attr & 0x400):
                    raise ReleaseValidationError(f"Non-regular ZIP member: {entry.filename}")
                if entry.filename != MANIFEST_NAME and (
                    not entry.filename.startswith(PREFIX)
                    or not _allowed_relative(entry.filename[len(PREFIX):])
                ):
                    raise ReleaseValidationError(f"Unapproved ZIP member: {entry.filename}")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ReleaseValidationError(f"CRC mismatch: {bad_member}")
            manifest = json.loads(archive.read(MANIFEST_NAME), object_pairs_hook=_unique_object)
            if not isinstance(manifest, dict):
                raise ReleaseValidationError("Manifest must be a filename-to-SHA256 object")
            for name, digest in manifest.items():
                if not _allowed_relative(name) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ReleaseValidationError(f"Invalid manifest entry: {name}")
            if not REQUIRED_FILES.issubset(manifest):
                raise ReleaseValidationError("Required release files are missing from manifest")
            expected = {PREFIX + name for name in manifest} | {MANIFEST_NAME}
            if set(names) != expected:
                raise ReleaseValidationError("ZIP members differ from the stored manifest")
            for name, digest in manifest.items():
                if hashlib.sha256(archive.read(PREFIX + name)).hexdigest() != digest:
                    raise ReleaseValidationError(f"SHA256 mismatch: {name}")
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError, RuntimeError, zlib.error) as exc:
        raise ReleaseValidationError(f"Unreadable or damaged release: {exc}") from exc
    return {"archive": str(path.resolve()), "files": len(manifest),
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verified": True}


def build_release(root: Path, output: Path) -> dict:
    """Create a new release; never overwrite an existing archive."""
    root, output = Path(root), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    files = collect_release_files(root)
    root = root.resolve(strict=True)
    # Read each file once so the bytes hashed are exactly the bytes archived.
    contents = {p.relative_to(root).as_posix(): _checked_path(p, root).read_bytes() for p in files}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(PREFIX + name, data)
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
    return verify_release(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="Build a new ZIP; never overwrite")
    action.add_argument("--verify", type=Path, help="Check an existing ZIP against its stored manifest")
    args = parser.parse_args(argv)
    try:
        result = (verify_release(args.verify) if args.verify is not None else
                  build_release(Path(__file__).resolve().parents[1], args.output))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
