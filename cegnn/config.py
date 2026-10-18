"""YAML config loading with environment-variable expansion.

Paths in ``configs/*.yaml`` may reference ``${CEGNN_DATA_ROOT}`` and
``${CEGNN_RESULTS_ROOT}``; both accept an inline default using the shell
syntax ``${VAR:-default}``. Nothing in the repo hard-codes a machine path.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_vars(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` using the process environment."""
    def _sub(m):
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(
            f"Environment variable {name} is not set and no default was given "
            f"in the config (use e.g. export {name}=/path)."
        )
    return _VAR.sub(_sub, value)


def _expand_tree(node: Any) -> Any:
    if isinstance(node, str):
        return expand_vars(node)
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    return node


def load_config(path: str, make_results_dir: bool = True) -> Dict[str, Any]:
    """Load a YAML config, expand env vars, and (optionally) create ``paths.results``."""
    with open(path, "r") as f:
        cfg = _expand_tree(yaml.safe_load(f))
    if make_results_dir:
        Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    return cfg


def results_dir(cfg: dict, name: str) -> Path:
    """``<paths.results>/<name>``, created on demand."""
    p = Path(cfg["paths"]["results"]) / name
    p.mkdir(parents=True, exist_ok=True)
    return p
