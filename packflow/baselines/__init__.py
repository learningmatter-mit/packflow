"""Baseline crystal-structure-prediction methods used for comparison.

Currently a thin wrapper around the Genarris CSP package (pinned as the
``external/Genarris`` submodule).
"""

from .genarris_wrapper import GenarrisWrapper

__all__ = ["GenarrisWrapper"]
