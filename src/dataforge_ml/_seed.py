"""The fixed library default seed (ADR-0106).

``seed=null`` on an Operation means this value, never "random": the same call
on the same data gives the same answer.
"""

from __future__ import annotations

#: The seed every stochastic step uses when the caller passes none.
DEFAULT_SEED = 42
