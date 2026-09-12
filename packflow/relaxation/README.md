# `packflow.relaxation` — UMA energy, relaxation, lattice energy & hydrogen addition

This is the single home for everything that turns a generated (or ground-truth)
crystal into **UMA** energies/forces and relaxed structures. It is the backbone of
the paper's **Figure 5** (relaxation trajectories + relative lattice energy vs
density) and the energy/force tables, and it provides the reward for
[`packflow.grpo`](../grpo).

## UMA / fairchem setup

UMA is provided by **fairchem**, pinned as a submodule for exact reproducibility
(not a pip dependency). Install it from the submodule:

```bash
git submodule update --init external/fairchem
uv pip install -e external/fairchem/packages/fairchem-core   # or the repo root, per fairchem's layout
```

The UMA model weights are fetched from HuggingFace. On offline compute nodes,
pre-download them and point the cache at a local dir:

```bash
export FAIRCHEM_CACHE_DIR=$PWD/uma_cache
export HF_HOME=$FAIRCHEM_CACHE_DIR
export HF_HUB_OFFLINE=1
```

Many drivers shell out to a fairchem-environment Python; select it via
`export FAIRCHEM_PYTHON=/path/to/fairchem/env/python` (defaults to the current
interpreter).

## Modules

Importable (no fairchem at import time):

| Module | Role |
|--------|------|
| `uma.py` | `UMAEnergyCalculator` + reward helpers (used by GRPO and below). |
| `relax.py` | High-level `relax_crystal` / `relax_batch` / `energy_and_forces` API (shells out to the workers). |
| `hydrogen.py` | Add hydrogens to CSD ground-truth structures (needs CSD license; set `CSD_DATABASE_PATH`). |
| `pipeline.py` | End-to-end postprocess of an evaluation directory (add H, relax, lattice energies). |

Fairchem-environment **workers** (run via subprocess with `FAIRCHEM_PYTHON`; hard-import fairchem):

| `workers/` | Role |
|------------|------|
| `uma_metrics.py` | Per-crystal UMA relaxation + lattice energy (called by the evaluator + pipelines). |
| `uma_energy_forces.py` | Single-point UMA energy/forces (+ per-atom energies). |
| `uma_h_relax.py` | Hydrogen-only relaxation + UMA features. |

## Python API

```python
from packflow.relaxation import relax_crystal, energy_and_forces
metrics = relax_crystal("pred.cif", relaxation_steps=1000, device="cuda")
sp = energy_and_forces("pred.cif", device="cuda")
```

## Figure 5 pipeline (typical)

```bash
# 1. Postprocess an eval dir: add H, relax with UMA, compute lattice energies.
packflow relax --eval_dir <evaluation_results_dir> --fairchem_python "$FAIRCHEM_PYTHON"

# 2. Ground-truth hydrogens for blind-test crystals (CSD license required).
python -m packflow.relaxation.hydrogen
```

The relaxed energies/forces written into the evaluation directory are the inputs
for the Figure-5 analysis (relaxation trajectories + relative lattice energy vs
density error).
