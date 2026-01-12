# core/contracts/__init__.py
"""
core/contracts

Small, explicit "contracts" that must be emitted in outputs so the repo is honest:
- validity: what statistical/physics assumptions apply for a run

Keep this package lightweight and import-safe.
"""

from .validity import validity_for_model_based, validity_for_monte_carlo  # noqa: F401