"""
core/config/__init__.py

Configuration boundary package for the radar pipeline.

This package contains:
- Case loading and schema validation (loaders.py)
- Unit normalization (units.py)
- Run manifest creation (manifest.py)

Design note
-----------
Keep this package "boring": strict, deterministic, and side-effect free.
"""

from __future__ import annotations

from core.config.loaders import load_case, load_schema, LoadOptions
from core.config.manifest import write_case_manifest

__all__ = [
    "load_case",
    "load_schema",
    "LoadOptions",
    "write_case_manifest",
]