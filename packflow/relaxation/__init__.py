"""UMA-based relaxation, lattice energy, and hydrogen-addition tools.

Single source of truth for turning a generated (or ground-truth) crystal into
UMA energies/forces and relaxed structures -- the basis for the paper's Figure 5
(relaxation trajectories + relative lattice energy vs density) and the energy/
force tables.

Layout
------
- ``uma``      -- importable UMA energy/force calculator + GRPO reward helpers.
- ``relax``    -- high-level relax/score API (shells out to the workers).
- ``hydrogen`` -- add hydrogens to CSD ground-truth heavy-atom structures.
- ``pipeline`` -- end-to-end post-processing of an evaluation directory.
- ``workers/`` -- fairchem-environment subprocess scripts (NOT imported here;
  they hard-import fairchem and run via ``FAIRCHEM_PYTHON``).

Heavy deps (fairchem/ase/ccdc) are imported lazily, so this package imports fine
without them installed.
"""

from .relax import relax_crystal, relax_batch, energy_and_forces

__all__ = ["relax_crystal", "relax_batch", "energy_and_forces"]
