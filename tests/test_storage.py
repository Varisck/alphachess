import uuid

import pytest

from alphachess.storage import Storage


@pytest.fixture
def storage():
    # Unique root per test — fsspec's memory backend is a process-global singleton.
    return Storage(f"memory://alphachess-test-{uuid.uuid4().hex}")


def test_write_read_roundtrip(storage):
    storage.write_bytes("foo.bin", b"hello")
    assert storage.read_bytes("foo.bin") == b"hello"


def test_write_creates_nested_dirs(storage):
    storage.write_bytes("a/b/c/file.txt", b"nested")
    assert storage.read_bytes("a/b/c/file.txt") == b"nested"


def test_exists(storage):
    assert storage.exists("missing.bin") is False
    storage.write_bytes("there.bin", b"x")
    assert storage.exists("there.bin") is True


def test_list_empty_dir_returns_empty(storage):
    assert storage.list("nonexistent") == []


def test_list_returns_sorted_names(storage):
    for name in ["c.txt", "a.txt", "b.txt"]:
        storage.write_bytes(f"d/{name}", b"x")
    assert storage.list("d") == ["a.txt", "b.txt", "c.txt"]


def test_list_filters_by_suffix(storage):
    storage.write_bytes("d/model.pt", b"x")
    storage.write_bytes("d/notes.txt", b"x")
    storage.write_bytes("d/other.pt", b"x")
    assert storage.list("d", suffix=".pt") == ["model.pt", "other.pt"]


def test_newest_returns_none_on_empty(storage):
    assert storage.newest("models", ".pt") is None


def test_newest_returns_lex_max(storage):
    for gen in [2, 0, 5, 1]:
        storage.write_bytes(f"models/{gen:06d}.pt", b"x")
    assert storage.newest("models", ".pt") == "000005.pt"


def test_atomic_put_roundtrip(storage):
    storage.atomic_put("models/000000.pt", b"weights")
    assert storage.read_bytes("models/000000.pt") == b"weights"


def test_atomic_put_leaves_no_tmp_file(storage):
    storage.atomic_put("models/000000.pt", b"weights")
    assert storage.list("models") == ["000000.pt"]


def test_newest_ignores_tmp_sidecars(storage):
    """Simulate a half-written file: an orphaned .tmp sidecar must not be
    returned as 'newest' because it does not end in the requested suffix."""
    storage.write_bytes("models/000000.pt", b"done")
    storage.write_bytes("models/000001.pt.tmp", b"in-progress")
    assert storage.newest("models", ".pt") == "000000.pt"


def test_separate_storage_instances_are_isolated():
    a = Storage("memory://run-a")
    b = Storage("memory://run-b")
    a.write_bytes("x.bin", b"from-a")
    assert b.exists("x.bin") is False


def test_local_file_backend(tmp_path):
    uri = tmp_path.as_uri()  # file:///... with correct Windows drive handling
    s = Storage(uri)
    s.write_bytes("data.bin", b"local")
    assert s.read_bytes("data.bin") == b"local"
    assert (tmp_path / "data.bin").read_bytes() == b"local"


def test_local_file_atomic_put(tmp_path):
    s = Storage(tmp_path.as_uri())
    s.atomic_put("models/000000.pt", b"weights")
    assert (tmp_path / "models" / "000000.pt").read_bytes() == b"weights"
    assert not (tmp_path / "models" / "000000.pt.tmp").exists()