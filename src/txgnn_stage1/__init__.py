"""Independent, source-reviewed TxGNN Step 1 research implementation.

The graph constructor accepts already normalized/indexed split tables.
Raw-asset preprocessing and split reproduction are tracked as open work.
"""
from .graph import build_graph, assert_no_leakage
__all__ = ["build_graph", "assert_no_leakage"]
