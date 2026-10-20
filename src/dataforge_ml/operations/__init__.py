"""Every Operation, one thin file each. Importing this package registers them all.

An Operation file holds the signature, the docstring written for Agents, the
Wire Records only it returns, and one call into internal code. Agents import
Operations from ``dataforge_ml``.
"""

from . import (  # noqa: F401 - importing each module registers its Operation
    describe_columns,
    describe_target,
    explain_routing,
    export_frame,
    fit_unit,
    list_handles,
    load,
    missingness,
    profile,
    resolve_recipe,
    route,
    split,
    top_associations,
    value_summary,
)
