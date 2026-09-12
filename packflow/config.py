"""Central configuration for PackFlow.

Every filesystem path and environment-variable lookup used across the package
funnels through this module, so the whole pipeline can be steered with a small,
documented set of env vars instead of scattered ``os.environ.get`` calls.

Environment variables
---------------------
PACKFLOW_CHECKPOINT_DIR  Where model-zoo checkpoints live (default: packaged dir).
PACKFLOW_HF_REPO         Hugging Face repo id used to download checkpoints.
PACKFLOW_DATA_DIR        Root for processed datasets / refcodes (default: <repo>/data).
PACKFLOW_EXPERIMENTS_DIR Raw training-run dir for ablation checkpoints.
FAIRCHEM_PYTHON          Python interpreter that has fairchem/UMA installed.
FAIRCHEM_CACHE_DIR       Cache dir for UMA model weights.
GENARRIS_PYTHON          Python interpreter for the Genarris baseline.
GENARRIS_ROOT            Path to the Genarris checkout (default: external/Genarris).
CSD_DATABASE_PATH        Path to the local CSD database (preprocessing / GT hydrogens).
WANDB_ENTITY             Default Weights & Biases entity for GRPO runs.
PACKFLOW_EVAL_STAGING_DIR  Optional fast scratch dir for UMA staging during eval.
PACKFLOW_FONT            Path to a .ttf used by the plotting figures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent          # .../packflow/packflow
REPO_ROOT = PACKAGE_DIR.parent                         # .../packflow


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val else default


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser() if val else default


# ---------------------------------------------------------------------------
# Checkpoints / model zoo
# ---------------------------------------------------------------------------
def checkpoint_dir() -> Path:
    """Directory holding ``<model>/best_model.pt`` and ``model_zoo.json``."""
    return _env_path("PACKFLOW_CHECKPOINT_DIR", PACKAGE_DIR / "checkpoints")


def hf_repo() -> str:
    """Hugging Face Hub repo id checkpoints are (optionally) downloaded from."""
    return _env("PACKFLOW_HF_REPO", "aksub99/packflow-checkpoints")


# ---------------------------------------------------------------------------
# Data / experiments
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    return _env_path("PACKFLOW_DATA_DIR", REPO_ROOT / "data")


def refcodes_dir() -> Path:
    """Directory of shipped CSD refcode lists (``train.txt``, ``test.txt``, …)."""
    return data_dir() / "refcodes"


def load_refcodes(name: str) -> list:
    """Load a one-refcode-per-line list from ``data/refcodes/<name>``."""
    path = refcodes_dir() / name
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def experiments_dir() -> Path:
    """Raw training-run directory (fallback source for ablation checkpoints)."""
    return _env_path("PACKFLOW_EXPERIMENTS_DIR", REPO_ROOT.parent)


# ---------------------------------------------------------------------------
# External tools / interpreters
# ---------------------------------------------------------------------------
def fairchem_python() -> str:
    """Interpreter with fairchem/UMA installed (UMA runs in a subprocess)."""
    return _env("FAIRCHEM_PYTHON", sys.executable)


def fairchem_cache_dir() -> str:
    return _env("FAIRCHEM_CACHE_DIR", str(REPO_ROOT / "uma_cache"))


def genarris_python() -> str:
    return _env("GENARRIS_PYTHON", "python")


def genarris_root() -> str:
    return _env("GENARRIS_ROOT", str(REPO_ROOT / "external" / "Genarris"))


# ---------------------------------------------------------------------------
# CSD / preprocessing
# ---------------------------------------------------------------------------
def csd_database_path() -> str:
    return _env("CSD_DATABASE_PATH", "")


def eval_staging_dir() -> Optional[str]:
    """Optional fast node-local scratch dir for UMA staging during evaluation.

    Unset (default) stages intermediates under the evaluation ``output_dir``.
    Set this to a fast local mount (e.g. a compute-node scratch disk) to avoid
    hammering shared storage during large UMA relaxation runs.
    """
    return _env("PACKFLOW_EVAL_STAGING_DIR")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def wandb_entity() -> Optional[str]:
    return _env("WANDB_ENTITY")


def font_path() -> str:
    """Path to a .ttf for plotting (empty string -> matplotlib default)."""
    return _env("PACKFLOW_FONT", "")
