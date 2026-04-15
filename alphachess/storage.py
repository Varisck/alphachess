"""Storage abstraction over fsspec.

One class, one interface, many backends. Application code passes a URI
(``file:///path``, ``s3://bucket/prefix``, ``memory://``) and never imports
``fsspec`` or ``os.path`` directly.
"""

from __future__ import annotations

import fsspec


class Storage:
    # resolve the URI once — fsspec picks the backend, we keep the base path
    def __init__(self, root_uri: str):
        self.root_uri = root_uri
        self._fs, self._root = fsspec.url_to_fs(root_uri)

    # join a rel path onto the root, normalize slashes
    def _full(self, rel_path: str) -> str:
        rel = rel_path.replace("\\", "/").lstrip("/")
        if not self._root:
            return rel
        return f"{self._root.rstrip('/')}/{rel}" if rel else self._root

    # dump bytes to rel_path, making parent dirs as needed
    def write_bytes(self, rel_path: str, data: bytes) -> None:
        path = self._full(rel_path)
        parent = path.rsplit("/", 1)[0]
        if parent and parent != path:
            self._fs.makedirs(parent, exist_ok=True)
        with self._fs.open(path, "wb") as f:
            f.write(data)

    # read the whole file at rel_path
    def read_bytes(self, rel_path: str) -> bytes:
        with self._fs.open(self._full(rel_path), "rb") as f:
            return f.read()

    # list filenames (not full paths) in rel_dir, optionally filtered by suffix
    def list(self, rel_dir: str, suffix: str | None = None) -> list[str]:
        path = self._full(rel_dir)
        if not self._fs.exists(path):
            return []
        entries = self._fs.ls(path, detail=False)
        names = [e.rsplit("/", 1)[-1] for e in entries]
        if suffix is not None:
            names = [n for n in names if n.endswith(suffix)]
        return sorted(names)

    # does rel_path exist on the backend
    def exists(self, rel_path: str) -> bool:
        return self._fs.exists(self._full(rel_path))

    # write to {rel_path}.tmp then rename, so readers never catch a half-written file
    def atomic_put(self, rel_path: str, data: bytes) -> None:
        final = self._full(rel_path)
        tmp = final + ".tmp"
        parent = final.rsplit("/", 1)[0]
        if parent and parent != final:
            self._fs.makedirs(parent, exist_ok=True)
        with self._fs.open(tmp, "wb") as f:
            f.write(data)
        self._fs.mv(tmp, final)

    # largest filename in rel_dir ending in suffix, or None — zero-padded names
    # make lex order == numeric order, so this finds the current generation
    def newest(self, rel_dir: str, suffix: str) -> str | None:
        names = self.list(rel_dir, suffix=suffix)
        return names[-1] if names else None
