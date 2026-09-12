#!/usr/bin/env python3
"""
Compute UMA single-point energy / per-atom forces / per-atom energies for periodic crystals.

This script is intended to be run inside the `fairchem` environment (where fairchem is installed).
It is designed to be called from packflow-side scripts via subprocess, similar to
`scripts/calculate_uma_metrics.py`, but additionally captures per-atom energies.

Inputs:
  - Batch mode:  --batch_manifest /path/to/manifest.json
    manifest is a JSON list of entries like:
      {"id": "...", "crystal_path": "/abs/path/to/structure.cif"}

  - Single mode: --crystal /path/to/structure.cif

Output (JSON to stdout; last line):
  { "<id>": { "total_energy": float, "per_atom_forces": [[...]], "per_atom_force_norms": [...],
              "per_atom_energies": [... or null], "error": str or null } , ... }
"""

import argparse
import json
import os
import sys
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
from ase.io import read as ase_read
from ase.optimize import FIRE
from ase.constraints import FixAtoms

# Add repo root to sys.path so we can run this script from anywhere (mirrors extract_uma_features.py)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fairchem.core import pretrained_mlip, FAIRChemCalculator  # type: ignore
from fairchem.core.datasets.atomic_data import AtomicData  # type: ignore
from fairchem.core.datasets import data_list_collater  # type: ignore


class UMAEnergyForcesExtractor:
    """Minimal UMA extractor capturing per-atom energies via a forward hook."""

    def __init__(self, predictor, task_name: str = "omc", device: str = "cuda"):
        self.predictor = predictor
        self.task_name = task_name
        self.device = device

        # storage
        self.captured_node_energies = None
        self.hook_handles = []

        # underlying model (mirrors scripts/extract_uma_features.py)
        self.model = predictor.model.module

        self._register_energy_hook()

        self.a2g = partial(
            AtomicData.from_ase,
            task_name=task_name,
            r_edges=False,
            r_data_keys=["spin", "charge"],
            max_neigh=None,
            radius=6.0,
        )

    def _find_energy_block(self, module, path: str = ""):
        # Direct attribute check
        if hasattr(module, "energy_block"):
            return module.energy_block, f"{path}.energy_block" if path else "energy_block"

        # wrapper classes can have `.head`
        if hasattr(module, "head"):
            eb, eb_path = self._find_energy_block(module.head, f"{path}.head" if path else "head")
            if eb is not None:
                return eb, eb_path

        # children
        for name, child in module.named_children():
            if hasattr(child, "energy_block"):
                return child.energy_block, f"{path}.{name}.energy_block" if path else f"{name}.energy_block"
            if hasattr(child, "head") and hasattr(child.head, "energy_block"):
                return child.head.energy_block, f"{path}.{name}.head.energy_block" if path else f"{name}.head.energy_block"

        return None, None

    def _register_energy_hook(self):
        energy_head_found = False

        for head_name, head in self.model.output_heads.items():
            energy_block, _ = self._find_energy_block(head)
            if energy_block is None:
                continue

            def make_energy_hook():
                def energy_block_hook(module, input, output):
                    # output is per-atom energies before summation
                    self.captured_node_energies = output.detach().clone()

                return energy_block_hook

            self.hook_handles.append(energy_block.register_forward_hook(make_energy_hook()))
            energy_head_found = True
            break

        if not energy_head_found:
            # best-effort broad search
            for _, module in self.model.named_modules():
                if hasattr(module, "energy_block"):
                    def make_energy_hook():
                        def energy_block_hook(module, input, output):
                            self.captured_node_energies = output.detach().clone()
                        return energy_block_hook
                    self.hook_handles.append(module.energy_block.register_forward_hook(make_energy_hook()))
                    energy_head_found = True
                    break

        if not energy_head_found:
            # Per-atom energies will be unavailable, but energy+forces still work.
            print("WARNING: Could not find energy_block; per-atom energies will not be captured", file=sys.stderr)

    def extract(self, atoms) -> Dict[str, Any]:
        # Ensure UMA expects these keys
        atoms.info["charge"] = atoms.info.get("charge", 0)
        atoms.info["spin"] = atoms.info.get("spin", 0)

        self.captured_node_energies = None

        data_object = self.a2g(atoms)
        batch = data_list_collater([data_object], otf_graph=True).to(self.device)

        with torch.no_grad():
            pred = self.predictor.predict(batch, undo_element_references=True)

        out: Dict[str, Any] = {}
        if "energy" in pred:
            out["total_energy"] = float(pred["energy"].detach().cpu().numpy()[0])
        if "forces" in pred:
            forces = pred["forces"].detach().cpu().numpy()
            out["per_atom_forces"] = forces.tolist()
            out["per_atom_force_norms"] = np.linalg.norm(forces, axis=1).tolist()
        else:
            out["per_atom_forces"] = None
            out["per_atom_force_norms"] = None

        # Per-atom energies (best-effort, mirrors extract_uma_features.py)
        per_atom_final: Optional[np.ndarray] = None
        if self.captured_node_energies is not None:
            per_atom = self.captured_node_energies.squeeze()
            if per_atom.dim() == 0:
                per_atom = per_atom.unsqueeze(0)

            try:
                # Find an energy task and use its normalizer/element refs to compute final per-atom energies
                for task_name, task in self.predictor.tasks.items():
                    if "energy" in task_name.lower():
                        normalizer = task.normalizer
                        per_atom_denorm = normalizer.denorm(per_atom)
                        if task.element_references is not None:
                            atomic_numbers_tensor = torch.tensor(
                                atoms.get_atomic_numbers(),
                                device=per_atom_denorm.device,
                                dtype=torch.long,
                            )
                            elem_refs = task.element_references.element_references
                            per_atom_refs = elem_refs[atomic_numbers_tensor].to(per_atom_denorm.dtype)
                            per_atom_final = (per_atom_denorm + per_atom_refs).detach().cpu().numpy()
                        else:
                            per_atom_final = per_atom_denorm.detach().cpu().numpy()
                        break
                else:
                    per_atom_final = per_atom.detach().cpu().numpy()
            except Exception:
                per_atom_final = per_atom.detach().cpu().numpy()

        out["per_atom_energies"] = per_atom_final.tolist() if per_atom_final is not None else None
        return out


def relax_h_only_under_pbc(
    atoms,
    predictor,
    task_name: str,
    fmax: float = 0.01,
    steps: int = 100,
) -> Dict[str, Any]:
    """
    Constrained geometry optimization under PBC:
      - fix lattice (no cell filter; cell is unchanged)
      - fix heavy atoms (Z != 1)
      - relax hydrogens only (Z == 1)
    Returns dict with:
      - h_relax_converged: bool
      - h_relax_steps_taken: int (best-effort)
      - h_relax_final_fmax: float (H-only max force norm in eV/Å, best-effort)
    """
    zs = np.array(atoms.get_atomic_numbers(), dtype=np.int64)
    heavy_idx = np.where(zs != 1)[0].tolist()
    if heavy_idx:
        atoms.set_constraint(FixAtoms(indices=heavy_idx))

    # Attach UMA calculator for ASE optimization (uses same predictor as single-point)
    atoms.calc = FAIRChemCalculator(predictor, task_name=task_name)

    opt = FIRE(atoms)
    converged = bool(opt.run(fmax=float(fmax), steps=int(steps)))

    # Best-effort steps taken
    steps_taken = None
    if hasattr(opt, "nsteps"):
        try:
            steps_taken = int(opt.nsteps)
        except Exception:
            steps_taken = None
    if steps_taken is None and hasattr(opt, "get_number_of_steps"):
        try:
            steps_taken = int(opt.get_number_of_steps())
        except Exception:
            steps_taken = None

    # Best-effort final H-only fmax (eV/Å)
    final_fmax = None
    try:
        h_idx = np.where(zs == 1)[0]
        if h_idx.size > 0:
            forces = atoms.get_forces()
            h_forces = forces[h_idx]
            final_fmax = float(np.linalg.norm(h_forces, axis=1).max())
        else:
            final_fmax = 0.0
    except Exception:
        final_fmax = None

    return {
        "h_relax_converged": converged,
        "h_relax_steps_taken": steps_taken,
        "h_relax_final_fmax": final_fmax,
    }


def _load_entries(args) -> list[dict]:
    if args.batch_manifest:
        with open(args.batch_manifest, "r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError("batch_manifest must be a JSON list")
        return entries
    if args.crystal:
        return [{"id": "single_entry", "crystal_path": args.crystal}]
    raise ValueError("Must pass either --batch_manifest or --crystal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_manifest", help="Path to JSON list of {id, crystal_path}")
    parser.add_argument("--crystal", help="Path to single crystal CIF")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="uma-s-1p1")
    parser.add_argument("--task_name", default="omc")
    args = parser.parse_args()

    entries = _load_entries(args)

    # Predictor ONCE
    predictor = pretrained_mlip.get_predict_unit(args.model_name, device=args.device)
    extractor = UMAEnergyForcesExtractor(predictor, task_name=args.task_name, device=args.device)

    results: Dict[str, Any] = {}
    total = len(entries)
    print(f"Processing {total} UMA single-point entries on {args.device}...", file=sys.stderr)

    for i, entry in enumerate(entries):
        if (i + 1) % 25 == 0:
            print(f"Processing {i+1}/{total}...", file=sys.stderr)

        entry_id = entry.get("id", f"entry_{i}")
        crystal_path = entry.get("crystal_path")

        try:
            if not crystal_path:
                raise ValueError("Missing crystal_path")
            atoms = ase_read(crystal_path)
            # Optional: UMA H-only relaxation under PBC before single-point extraction
            relax_h = bool(entry.get("relax_h", False))
            relax_fmax = float(entry.get("relax_fmax", 0.01))
            relax_steps = int(entry.get("relax_steps", 100))
            h_relax_converged = None
            h_relax_steps_taken = None
            h_relax_final_fmax = None
            if relax_h:
                try:
                    relax_meta = relax_h_only_under_pbc(
                        atoms,
                        predictor=predictor,
                        task_name=args.task_name,
                        fmax=relax_fmax,
                        steps=relax_steps,
                    )
                    h_relax_converged = relax_meta.get("h_relax_converged")
                    h_relax_steps_taken = relax_meta.get("h_relax_steps_taken")
                    h_relax_final_fmax = relax_meta.get("h_relax_final_fmax")
                except Exception as e:
                    # If relaxation fails, still attempt single-point; record failure as non-converged.
                    h_relax_converged = False
                    h_relax_steps_taken = None
                    h_relax_final_fmax = None
                    print(f"WARNING: H-only UMA relaxation failed for {entry_id}: {e}", file=sys.stderr)

            out = extractor.extract(atoms)
            out["error"] = None
            out["h_relax_converged"] = h_relax_converged
            out["h_relax_steps_taken"] = h_relax_steps_taken
            out["h_relax_final_fmax"] = h_relax_final_fmax
            results[entry_id] = out
        except Exception as e:
            results[entry_id] = {
                "total_energy": None,
                "per_atom_forces": None,
                "per_atom_force_norms": None,
                "per_atom_energies": None,
                "h_relax_converged": None,
                "h_relax_steps_taken": None,
                "h_relax_final_fmax": None,
                "error": str(e),
            }

        # cleanup
        try:
            if crystal_path and os.path.exists(crystal_path):
                os.remove(crystal_path)
        except OSError:
            pass

    print(json.dumps(results))


if __name__ == "__main__":
    main()

