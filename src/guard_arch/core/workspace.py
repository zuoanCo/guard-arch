"""Workspace root and filesystem sandbox."""

from pathlib import Path


class SandboxError(PermissionError):
    """Raised when a path escapes the workspace root."""


class Workspace:
    """A rooted directory that all file operations are confined to."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def resolve(self, path: str | Path) -> Path:
        """Resolve `path` against the workspace root, rejecting escapes."""
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        resolved = p.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise SandboxError(f"path escapes workspace: {path}")
        return resolved

    def is_within(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
        except SandboxError:
            return False
        return True


class SandboxManager:
    """Gatekeeper used by tools before touching the filesystem."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def check(self, path: str | Path) -> Path:
        return self.workspace.resolve(path)
