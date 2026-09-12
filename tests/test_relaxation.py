"""Relaxation package: imports cleanly without fairchem, exposes the API."""

import importlib
import os

import pytest


def test_relaxation_imports_without_fairchem():
    """`packflow.relaxation` must import even when fairchem/ase are absent."""
    relax = importlib.import_module("packflow.relaxation")
    for name in ("relax_crystal", "relax_batch", "energy_and_forces"):
        assert callable(getattr(relax, name))


def test_workers_are_not_auto_imported():
    """The fairchem-only workers must NOT be imported at package import time."""
    import packflow.relaxation  # noqa: F401

    # Importing the package should not pull in the worker modules (they
    # hard-import fairchem and only run via subprocess in the fairchem env).
    import sys
    assert "packflow.relaxation.workers.uma_metrics" not in sys.modules


def test_worker_scripts_exist():
    """The fairchem workers must ship as runnable scripts."""
    from packflow.relaxation import relax

    for worker in ("uma_metrics.py", "uma_energy_forces.py", "uma_h_relax.py"):
        assert os.path.exists(os.path.join(relax.WORKERS_DIR, worker)), worker


def test_relax_worker_path_points_into_workers():
    from packflow.relaxation import relax

    assert relax.WORKERS_DIR.endswith(os.path.join("relaxation", "workers"))
