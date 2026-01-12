"""
cli/utils.py

Small CLI utilities (pure helpers, no business logic).

Design goals
------------
- Keep helpers generic and dependency-light.
- No imports from heavy simulation modules here.
- Provide consistent JSON writing and path handling across CLI scripts.

Note
----
This module is safe to import from any CLI entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import datetime as _dt
import json


def utc_timestamp_compact() -> str:
    """
    Return a compact UTC timestamp suitable for filenames.

    Format: YYYYMMDDTHHMMSSZ
    """
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists and return the resolved Path.

    Parameters
    ----------
    path : Path
        Directory to create (parents allowed).

    Returns
    -------
    Path
        Resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def write_json(path: Path, obj: Dict[str, Any], *, sort_keys: bool = True, indent: int = 2) -> None:
    """
    Write a JSON file with stable formatting (human diff-friendly).

    Parameters
    ----------
    path : Path
        Output file path.
    obj : dict
        JSON-serializable dictionary.
    sort_keys : bool
        Sort keys for stable diffs.
    indent : int
        Indentation level.
    """
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    """
    Read a JSON file into a dict.

    Raises
    ------
    ValueError
        If the file does not contain a JSON object.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at top-level: {path}")
    return data


def coerce_int(x: Any, *, name: str, default: Optional[int] = None) -> int:
    """
    Coerce a value to int with a helpful error message.

    Parameters
    ----------
    x : Any
        Value to parse.
    name : str
        Field name for error messages.
    default : int | None
        Default if x is None.

    Returns
    -------
    int
    """
    if x is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return int(default)
    try:
        return int(x)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer, got {x!r}") from exc