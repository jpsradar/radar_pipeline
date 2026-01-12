"""
core/config/loaders.py

Config loading + schema validation for the radar performance/simulation pipeline.

This module is intentionally "boring and strict":
- Loads a case configuration from YAML or JSON.
- Validates it against a JSON Schema contract (versioned in configs/schemas).
- Normalizes/expands the configuration (hooks for unit normalization and path resolution).
- Produces a clean in-memory dict ready to be consumed by the simulation engine.

Inputs
------
- case_path: Path to a YAML/JSON case file (e.g., configs/cases/demo_pd_noise.yaml)
- schema_dir: Directory containing JSON schema files (e.g., configs/schemas)
- schema_name: Which schema to validate against (default: "case.schema.json")

Outputs
-------
- A Python dict representing the validated case configuration.

CLI usage (example)
-------------------
This module is typically invoked by cli/run_case.py, but you can also run it directly:

    python -c "from core.config.loaders import load_case; \
              cfg = load_case('configs/cases/demo_pd_noise.yaml'); \
              print(cfg.keys())"

Design notes
------------
- We keep validation at the boundary: errors should be raised early with helpful messages.
- We do not silently coerce types (unless explicitly enabled).
- Unit conversion/normalization is delegated to core/config/units.py (hook function).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json


# ----------------------------
# Exceptions
# ----------------------------

class ConfigError(Exception):
    """Base exception for configuration-related errors."""


class ConfigIOError(ConfigError):
    """Raised when reading a config file fails (missing file, invalid YAML/JSON, etc.)."""


class SchemaValidationError(ConfigError):
    """Raised when JSON schema validation fails."""


class DependencyError(ConfigError):
    """Raised when an optional dependency (PyYAML/jsonschema) is not available."""


# ----------------------------
# Public API
# ----------------------------

@dataclass(frozen=True)
class LoadOptions:
    """
    Options controlling how the case is loaded and post-processed.

    Attributes
    ----------
    strict : bool
        If True, schema validation errors are fatal (recommended).
    resolve_paths : bool
        If True, resolves known path-like fields relative to the case file directory.
    normalize_units : bool
        If True, calls the unit normalization hook in core/config/units.py (if available).
    """
    strict: bool = True
    resolve_paths: bool = True
    normalize_units: bool = True


def load_case(
    case_path: Union[str, Path],
    *,
    schema_dir: Union[str, Path] = "configs/schemas",
    schema_name: str = "case.schema.json",
    options: Optional[LoadOptions] = None,
) -> Dict[str, Any]:
    """
    Load a YAML/JSON case file, validate it against a JSON Schema, and post-process it.

    Parameters
    ----------
    case_path
        Path to YAML/JSON case file.
    schema_dir
        Directory containing JSON Schemas.
    schema_name
        Filename of the schema to use for validation.
    options
        LoadOptions controlling strictness and post-processing.

    Returns
    -------
    dict
        Validated and (optionally) normalized configuration dictionary.

    Raises
    ------
    ConfigIOError
        If the case file cannot be read/parsed.
    SchemaValidationError
        If schema validation fails and options.strict is True.
    DependencyError
        If required optional dependencies are missing.
    """
    opts = options or LoadOptions()

    case_path = Path(case_path)
    if not case_path.exists():
        raise ConfigIOError(f"Case file not found: {case_path}")

    raw_cfg = _read_yaml_or_json(case_path)

    # Defensive check: users sometimes accidentally pass a schema file as a "case".
    # This catches the classic failure mode seen in logs:
    # "On instance: {'$schema': ..., 'properties': ... }"
    _guard_case_is_not_a_schema(instance=raw_cfg, case_path=case_path)

    schema_path = _resolve_schema_path(
        schema_dir=schema_dir,
        schema_name=schema_name,
        project_cwd=Path.cwd(),
    )
    if not schema_path.exists():
        raise ConfigIOError(f"Schema file not found: {schema_path}")

    # Validate early (boundary check).
    _validate_against_schema(instance=raw_cfg, schema_path=schema_path, strict=opts.strict)

    # Post-processing hooks
    cfg = dict(raw_cfg)  # shallow copy; deep copy is unnecessary unless mutating nested content
    if opts.resolve_paths:
        cfg = _resolve_relative_paths(cfg, base_dir=case_path.parent)

    if opts.normalize_units:
        cfg = _normalize_units_hook(cfg)

    # Final defensive check: post-processing must preserve dict contract.
    if not isinstance(cfg, dict):
        raise ConfigError("Internal error: load_case post-processing must return a dict")

    return cfg


def load_schema(schema_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON schema from disk.

    Parameters
    ----------
    schema_path
        Path to schema JSON.

    Returns
    -------
    dict
        Parsed schema as a dict.

    Raises
    ------
    ConfigIOError
        If reading/parsing fails.
    """
    schema_path = Path(schema_path)
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigIOError(f"Failed to read/parse schema JSON: {schema_path}. Error: {exc}") from exc


# ----------------------------
# Internal helpers
# ----------------------------

def _read_yaml_or_json(path: Path) -> Dict[str, Any]:
    """
    Read a YAML or JSON file into a dict.

    Notes
    -----
    - YAML support requires PyYAML.
    - JSON is handled via the standard library.
    """
    suffix = path.suffix.lower()

    if suffix in {".json"}:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigIOError(f"Failed to read/parse JSON: {path}. Error: {exc}") from exc

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigIOError(f"Top-level JSON must be an object/dict: {path}")
        return data

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise DependencyError(
                "PyYAML is required to load YAML case files. Install with: pip install pyyaml"
            ) from exc

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigIOError(f"Failed to read/parse YAML: {path}. Error: {exc}") from exc

        if data is None:
            # Empty YAML file => treat as empty dict for clearer downstream errors.
            data = {}

        if not isinstance(data, dict):
            raise ConfigIOError(f"Top-level YAML must be a mapping/dict: {path}")

        return data

    raise ConfigIOError(f"Unsupported config file extension '{suffix}' for: {path}")


def _resolve_schema_path(*, schema_dir: Union[str, Path], schema_name: str, project_cwd: Path) -> Path:
    """
    Resolve schema path robustly.

    Why this exists
    ---------------
    Users often run the CLI from a different working directory. Relying purely on
    relative paths can silently break and cause confusing validation behavior.

    Resolution order
    ----------------
    1) schema_dir relative to current working directory (typical CLI usage)
    2) schema_dir relative to repository root inferred from this file location
    """
    schema_dir_p = Path(schema_dir)

    # 1) CWD-based resolution
    p1 = (project_cwd / schema_dir_p / schema_name).resolve()
    if p1.exists():
        return p1

    # 2) Repo-root-ish fallback: <repo>/core/config/loaders.py -> <repo>/configs/schemas/...
    repo_root_guess = Path(__file__).resolve().parents[2]
    p2 = (repo_root_guess / schema_dir_p / schema_name).resolve()
    return p2


def _validate_against_schema(*, instance: Dict[str, Any], schema_path: Path, strict: bool) -> None:
    """
    Validate a config dict against a JSON Schema.

    If strict=False, validation warnings are printed but not raised.

    Notes
    -----
    - Requires the 'jsonschema' package.
    - Uses validator_for(schema) to support Draft 2020-12 features (e.g., unevaluatedProperties).
    """
    try:
        import jsonschema  # type: ignore
    except Exception as exc:
        raise DependencyError(
            "jsonschema is required for schema validation. Install with: pip install jsonschema"
        ) from exc

    schema = load_schema(schema_path)

    try:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    except Exception as exc:
        # This catches schema compilation issues (bad $ref, invalid schema, etc.).
        raise SchemaValidationError(f"Schema validator initialization failed for {schema_path}: {exc}") from exc

    if not errors:
        return

    # Build a compact but actionable error message.
    lines = [f"Schema validation failed for schema={schema_path.name}:"]
    for e in errors[:10]:
        loc = "$"
        if e.path:
            loc = "$." + ".".join(str(p) for p in e.path)
        lines.append(f" - {loc}: {e.message}")
    if len(errors) > 10:
        lines.append(f" - ... ({len(errors) - 10} more errors)")

    msg = "\n".join(lines)
    if strict:
        raise SchemaValidationError(msg)

    print(f"[WARN] {msg}")


def _guard_case_is_not_a_schema(*, instance: Dict[str, Any], case_path: Path) -> None:
    """
    Detect the common operator error where a schema JSON is passed as a case file.

    Heuristic (conservative)
    ------------------------
    If the loaded object contains typical schema keys, this is almost certainly not a case.
    """
    if not isinstance(instance, dict):
        return

    schema_like_keys = {"$schema", "$id", "properties", "required", "type", "$defs", "oneOf"}
    hits = schema_like_keys.intersection(instance.keys())
    if hits:
        raise ConfigIOError(
            "Loaded case file looks like a JSON Schema, not a case config. "
            f"case_path={case_path} keys={sorted(hits)}. "
            "Double-check you passed the correct --case file (e.g., configs/cases/*.yaml), "
            "not configs/schemas/*.json."
        )


def _resolve_relative_paths(cfg: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    """
    Resolve known path-like fields relative to the case file directory.

    This is conservative on purpose: we only resolve keys that look like paths and
    are common in radar simulation configs (e.g., terrain maps, external patterns).

    Rules
    -----
    - Any string value for keys matching "*_path" or "*_file" is treated as a path.
    - If the path is relative, it is resolved against base_dir.
    - Non-string values are ignored.

    You can expand this later with an explicit allowlist per sub-config.
    """
    def resolve_value(key: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        k = key.lower()
        if not (k.endswith("_path") or k.endswith("_file")):
            return value

        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((base_dir / p).resolve())

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: walk(resolve_value(k, v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    return walk(cfg)


def _normalize_units_hook(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook into core/config/units.py to normalize units and dB/linear conversions.

    The units module is expected to provide a function with signature:

        normalize_case_config(cfg: dict) -> dict

    If the hook is not implemented yet, this function returns cfg unchanged.
    """
    try:
        # Local import to avoid import-time coupling.
        from core.config import units  # type: ignore
    except Exception:
        return cfg

    normalize_fn = getattr(units, "normalize_case_config", None)
    if normalize_fn is None:
        return cfg

    try:
        normalized = normalize_fn(cfg)
        if not isinstance(normalized, dict):
            raise TypeError("normalize_case_config must return a dict")
        return normalized
    except Exception as exc:
        raise ConfigError(f"Unit normalization failed: {exc}") from exc