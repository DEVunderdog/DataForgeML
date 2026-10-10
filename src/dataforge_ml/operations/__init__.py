"""Every Operation, one thin file each. Importing this package registers them all.

An Operation file holds the signature, the docstring written for Agents, the
Wire Records only it returns, and one call into internal code. Agents import
Operations from ``dataforge_ml``.
"""

from . import export_frame, list_handles, load  # noqa: F401
