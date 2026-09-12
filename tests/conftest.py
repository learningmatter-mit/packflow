"""Shared pytest fixtures/config for the PackFlow test suite.

Tests are CPU-only and fast by default. Tests that need the heavy ML stack
(torch + model) or the local checkpoints are guarded so the suite still collects
and runs the light tests in a minimal environment.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


requires_torch = pytest.mark.skipif(not _has("torch"), reason="torch not installed")


@pytest.fixture(scope="session")
def zoo():
    from packflow import checkpoints

    return checkpoints.load_zoo()


@pytest.fixture(scope="session")
def local_models(zoo):
    """Model-zoo names whose checkpoint file is present locally."""
    from packflow import checkpoints

    return [n for n in zoo if checkpoints.local_path(n).exists()]
