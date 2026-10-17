"""Every Operation, one thin file each. Importing this package registers them all.

An Operation file holds the signature, the docstring written for Agents, the
Wire Records only it returns, and one call into internal code. Agents import
Operations from ``dataforge_ml``.
"""

from . import (  # noqa: F401
    describe_columns,
    describe_target,
    export_frame,
    list_handles,
    load,
    missingness,
    profile,
    route,
    split,
    top_associations,
    value_summary,
)
