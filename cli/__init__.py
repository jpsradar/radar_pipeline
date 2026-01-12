"""
cli/__init__.py

CLI package for the radar pipeline.

This package intentionally contains only lightweight helpers and entrypoints.
Heavy imports must remain inside command functions to keep CLI startup fast and
to avoid import-time side effects.
"""