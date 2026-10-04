"""Versioned, content-checked bundles with deterministic archives."""

from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from . import __version__


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_path(root, name):
    root = Path(root).resolve()
    p = PurePosixPath(name)
    if not name or str(p) != name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"Unsafe bundle path: {name!r}")
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Bundle path escapes root: {name}")
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlinks are not bundle members: {name}")
    return target


def seal(root, kind, metadata=None):
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        safe_path(root, name)
        if path.is_file() and name != "manifest.json":
            records.append(
                dict(path=name, bytes=path.stat().st_size, sha256=sha256(path))
            )
    manifest = dict(
        schema_version=1,
        package_version=__version__,
        kind=kind,
        metadata=metadata or {},
        files=records,
    )
    write_json(root / "manifest.json", manifest)
    return manifest


def verify(root):
    root = Path(root)
    safe_path(root, "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema")
    seen = set()
    for record in manifest["files"]:
        name = record["path"]
        if name in seen or name == "manifest.json":
            raise ValueError(f"Duplicate or reserved member: {name}")
        seen.add(name)
        path = safe_path(root, name)
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Missing or corrupt bundle member: {name}")
    actual = set()
    for p in root.rglob("*"):
        name = p.relative_to(root).as_posix()
        safe_path(root, name)
        if p.is_file() and name != "manifest.json":
            actual.add(name)
    if actual != seen:
        raise ValueError(f"Unlisted members: {sorted(actual - seen)}")
    return manifest


def archive(root, output):
    root, output = Path(root), Path(output)
    manifest = verify(root)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Write archives outside the bundle")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".partial")
    try:
        with temp.open("xb") as raw:
            compressed = output.name.endswith(".gz")
            stream = (
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
                if compressed
                else raw
            )
            try:
                with tarfile.open(
                    fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT
                ) as tar:
                    for name in sorted(
                        ["manifest.json"] + [r["path"] for r in manifest["files"]]
                    ):
                        p = safe_path(root, name)
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = p.stat().st_size, 0o644, 0
                        with p.open("rb") as f:
                            tar.addfile(info, f)
            finally:
                if compressed:
                    stream.close()
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return sha256(output)


def unpack(source, output):
    """Extract regular files only; verify before making the directory visible."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".atomic-unpack-", dir=output.parent
    ) as tmp:
        root = Path(tmp) / "bundle"
        root.mkdir()
        seen = set()
        with tarfile.open(source, "r|*") as tar:
            for member in tar:
                if not member.isfile() or member.name in seen:
                    raise ValueError(f"Not a unique regular file: {member.name}")
                seen.add(member.name)
                p = safe_path(root, member.name)
                p.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, p.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        manifest = verify(root)
        root.rename(output)
    return manifest
