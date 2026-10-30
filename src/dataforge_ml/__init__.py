"""DataForgeML public API.

The package root exports the Operations and nothing else (ADR-0099). Each
Operation is a typed function an Agent calls by name; its arguments and its
Result are the whole contract. Everything else — config objects, splitters,
imputers, records — is internal and reachable only through submodule imports.

Every name is imported on first use, not when the package is imported. The
``dataforge`` CLI imports this package, and ``dataforge --help`` and
``dataforge schema`` must answer without importing pandas, polars or
scikit-learn (ADR-0112).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "c2st",
    "compose_imputer",
    "describe_c2st_columns",
    "describe_columns",
    "describe_target",
    "evaluate_imputation",
    "explain_routing",
    "export_frame",
    "fit_unit",
    "impute",
    "list_handles",
    "load",
    "missingness",
    "profile",
    "query_frame",
    "resolve_recipe",
    "route",
    "split",
    "top_associations",
    "value_summary",
]


def __getattr__(name: str) -> Any:
    # Each Operation lives in ``operations/<name>.py`` under its own name.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".operations.{name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
