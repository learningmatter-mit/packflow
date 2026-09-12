"""Result serialization and summary reporting for evaluations.

These wrap the proven ``save_results`` / ``print_summary`` implementations that
live in :mod:`packflow.evaluation.evaluator` (kept there to stay adjacent to the
metric aggregation they depend on), exposed here as a dedicated reporting module.
"""

from __future__ import annotations

from .evaluator import save_results, print_summary

__all__ = ["save_results", "print_summary"]
