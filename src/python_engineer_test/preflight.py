from __future__ import annotations

import shutil
from pathlib import Path


def require_file(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Expected a file path, got: {p}")
    return p


def require_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")
    if not p.is_dir():
        raise ValueError(f"Expected a directory path, got: {p}")
    return p


def require_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Required command not found on PATH: {command!r}. "
            "Install it and ensure it is available on PATH."
        )
    return resolved
