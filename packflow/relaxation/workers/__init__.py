"""Fairchem-environment UMA worker scripts (run via subprocess).

These modules hard-import ``fairchem``/``ase`` and are executed with the
interpreter given by ``FAIRCHEM_PYTHON`` (see :func:`packflow.config.fairchem_python`).
They are intentionally NOT imported by the rest of the package, so ``packflow``
imports cleanly in an environment without fairchem installed.

- ``uma_metrics``       -- per-crystal UMA relaxation + lattice energy.
- ``uma_energy_forces`` -- single-point UMA energy / per-atom forces & energies.
- ``uma_h_relax``       -- hydrogen-only relaxation + UMA features.
"""
