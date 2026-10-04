import io
import json
from pathlib import Path
import tarfile
import pytest
from atomic_features.bundles import seal, verify, archive, unpack, safe_path


def test_roundtrip_and_deterministic_archive(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "x").write_bytes(b"example\x00bytes")
    seal(source, "test", {"threshold": 0.7})
    a, b = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    assert archive(source, a) == archive(source, b)
    unpack(a, tmp_path / "unpacked")
    assert verify(source) == verify(tmp_path / "unpacked")
    (tmp_path / "unpacked/x").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        verify(tmp_path / "unpacked")


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "x/../../escape", "a\\b", "a//b", "./a"]
)
def test_reject_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


def test_unlisted_and_duplicate_files(tmp_path):
    (tmp_path / "a").write_text("a")
    seal(tmp_path, "test")
    (tmp_path / "b").write_text("b")
    with pytest.raises(ValueError, match="Unlisted"):
        verify(tmp_path)
    (tmp_path / "b").unlink()
    p = tmp_path / "manifest.json"
    m = json.loads(p.read_text())
    m["files"] *= 2
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Duplicate"):
        verify(tmp_path)


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate"])
def test_unsafe_archive_leaves_no_output(tmp_path, kind):
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as tar:
        member = tarfile.TarInfo("../escape" if kind == "traversal" else "x")
        member.size = 1
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp"
        tar.addfile(member, io.BytesIO(b"x"))
        if kind == "duplicate":
            tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        unpack(archive_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
