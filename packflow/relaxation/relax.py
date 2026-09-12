"""High-level, importable API for UMA relaxation / energy scoring.

The actual UMA compute happens in the fairchem-only worker scripts under
``packflow/relaxation/workers/`` (they hard-import fairchem). This module shells
out to them with the interpreter from ``FAIRCHEM_PYTHON`` so it stays importable
in any environment.

    from packflow.relaxation import relax_crystal
    metrics = relax_crystal("pred.cif", relaxation_steps=1000, device="cuda")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .. import config

WORKERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers")


def _run_worker(worker: str, args: List[str], fairchem_python: Optional[str] = None) -> str:
    """Run a worker script in the fairchem env and return its stdout."""
    python = fairchem_python or config.fairchem_python()
    script = os.path.join(WORKERS_DIR, worker)
    cmd = [python, script, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"UMA worker {worker} failed:\n{proc.stderr}")
    return proc.stdout


def _last_json_line(stdout: str) -> Any:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line and line[0] in "[{":
            return json.loads(line)
    raise ValueError("No JSON payload found in worker output.")


def relax_crystal(
    crystal_path: str,
    molecule_path: Optional[str] = None,
    relaxation_steps: int = 0,
    device: str = "cuda",
    fairchem_python: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Relax a single crystal (CIF) with UMA and return its metrics dict.

    Set ``relaxation_steps=0`` for a single-point (no relaxation) calculation.
    """
    args = ["--crystal", crystal_path, "--device", device,
            "--relaxation_steps", str(relaxation_steps)]
    if molecule_path:
        args += ["--molecule", molecule_path]
    if extra_args:
        args += extra_args
    return _last_json_line(_run_worker("uma_metrics.py", args, fairchem_python))


def relax_batch(
    manifest_path: str,
    device: str = "cuda",
    skip_relaxation: bool = False,
    fairchem_python: Optional[str] = None,
) -> Dict[str, Any]:
    """Run UMA over a batch manifest (JSON list of ``{id, crystal_path, ...}``)."""
    args = ["--batch_manifest", manifest_path, "--device", device]
    if skip_relaxation:
        args.append("--skip_relaxation")
    return _last_json_line(_run_worker("uma_metrics.py", args, fairchem_python))


def energy_and_forces(
    crystal_path: str,
    device: str = "cuda",
    fairchem_python: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-point UMA energy / per-atom forces & energies for one crystal."""
    args = ["--crystal", crystal_path, "--device", device]
    return _last_json_line(_run_worker("uma_energy_forces.py", args, fairchem_python))
