#!/usr/bin/env python3
"""
H-only relaxation under UMA (PBC on, cell fixed) + UMA feature extraction.

Run this in the `fairchem` environment.

Inputs:
  --batch_manifest <json> : list of entries:
    {
      "id": "REF_seed_0_pred" (unique),
      "crystal_path": "/abs/path/to/input.cif",
      "num_heavy": 54,
      "refcode": "REF",
      "seed_idx": 0,
      "kind": "pred" | "gt",
      "relax_fmax": 0.05,
      "relax_max_steps": 25
    }
  --out_pt <path> : torch.save(list_of_dicts) containing heavy arrays (forces, embeddings, per-atom energies, coords, etc)

Outputs (JSON to stdout, last line):
  {
    "<id>": {
      "pt_index": int,
      "h_relax_converged": bool,
      "h_relax_steps": int,
      "h_relax_trajectory": [float],
      "energy": float,
      "mean_force_norm": float,
      "max_force_norm": float
    },
    ...
  }
"""

import argparse
import json
import os
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from ase.io import read as ase_read
from ase.optimize import FIRE
from ase.constraints import FixAtoms


def cuda_sync_and_check() -> bool:
    """Synchronize CUDA and check for errors. Returns True if OK, False if error detected."""
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        # CUDA error detected (e.g., assertion failure)
        print(f"[CUDA ERROR] Detected: {e}", file=sys.stderr, flush=True)
        return False


def cuda_clear_error_state():
    """Attempt to clear CUDA error state so subsequent operations can proceed."""
    if not torch.cuda.is_available():
        return
    try:
        # Reset the current device to clear error state
        device = torch.cuda.current_device()
        torch.cuda.synchronize()
    except RuntimeError:
        pass  # Error already logged, just try to continue
    
    # Force a small CUDA operation to verify GPU is responsive
    try:
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"[CUDA WARNING] GPU may be in bad state after error: {e}", file=sys.stderr, flush=True)

# Compute nodes often have no internet; force HuggingFace/fairchem to use local cache only.
# This prevents noisy HTTPSConnectionPool retry logs and ensures deterministic behavior.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# fairchem imports
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets import data_list_collater


class UMAFeatureExtractorLite:
    """Capture node embeddings (scalar) + per-atom energies via hooks, plus energy/forces via predictor.predict()."""

    def __init__(self, predictor, task_name: str = "omc", device: str = "cuda"):
        self.predictor = predictor
        self.task_name = task_name
        self.device = device

        self.model = predictor.model.module
        self.captured_node_embeddings = None
        self.captured_node_energies = None
        self.hook_handles = []

        self._register_hooks()

        self.a2g = partial(
            AtomicData.from_ase,
            task_name=task_name,
            r_edges=False,
            r_data_keys=["spin", "charge"],
            max_neigh=None,
            radius=6.0,
        )

    def _find_energy_block(self, module, path: str = ""):
        if hasattr(module, "energy_block"):
            return module.energy_block, f"{path}.energy_block" if path else "energy_block"
        if hasattr(module, "head"):
            eb, eb_path = self._find_energy_block(module.head, f"{path}.head" if path else "head")
            if eb is not None:
                return eb, eb_path
        for name, child in module.named_children():
            if hasattr(child, "energy_block"):
                return child.energy_block, f"{path}.{name}.energy_block" if path else f"{name}.energy_block"
            if hasattr(child, "head") and hasattr(child.head, "energy_block"):
                return child.head.energy_block, f"{path}.{name}.head.energy_block" if path else f"{name}.head.energy_block"
        return None, None

    def _register_hooks(self):
        # Backbone hook for embeddings
        def backbone_hook(module, input, output):
            if isinstance(output, dict) and "node_embedding" in output:
                self.captured_node_embeddings = output["node_embedding"].detach().clone()

        self.hook_handles.append(self.model.backbone.register_forward_hook(backbone_hook))

        # Energy hook
        energy_head_found = False
        for _, head in self.model.output_heads.items():
            energy_block, _ = self._find_energy_block(head)
            if energy_block is None:
                continue

            def make_energy_hook():
                def energy_block_hook(module, input, output):
                    self.captured_node_energies = output.detach().clone()

                return energy_block_hook

            self.hook_handles.append(energy_block.register_forward_hook(make_energy_hook()))
            energy_head_found = True
            break

        if not energy_head_found:
            print("WARNING: Could not find energy_block; per-atom energies may be unavailable", file=sys.stderr)

    def extract(self, atoms) -> Dict[str, Any]:
        atoms.info["charge"] = atoms.info.get("charge", 0)
        atoms.info["spin"] = atoms.info.get("spin", 0)

        self.captured_node_embeddings = None
        self.captured_node_energies = None

        data_object = self.a2g(atoms)
        batch = data_list_collater([data_object], otf_graph=True).to(self.device)

        with torch.no_grad():
            pred = self.predictor.predict(batch, undo_element_references=True)

        out: Dict[str, Any] = {}
        out["energy"] = float(pred["energy"].detach().cpu().numpy()[0]) if "energy" in pred else None
        if "forces" in pred:
            forces = pred["forces"].detach().cpu().numpy()
            out["forces"] = forces
            out["per_atom_force_norms"] = np.linalg.norm(forces, axis=1)
            out["mean_force_norm"] = float(np.mean(out["per_atom_force_norms"]))
            out["max_force_norm"] = float(np.max(out["per_atom_force_norms"]))
        else:
            out["forces"] = None
            out["per_atom_force_norms"] = None
            out["mean_force_norm"] = None
            out["max_force_norm"] = None

        # Per-atom energies (best-effort; mirrors extract_uma_features.py)
        per_atom_final = None
        if self.captured_node_energies is not None:
            per_atom = self.captured_node_energies.squeeze()
            if per_atom.dim() == 0:
                per_atom = per_atom.unsqueeze(0)
            try:
                for task_name, task in self.predictor.tasks.items():
                    if "energy" in task_name.lower():
                        normalizer = task.normalizer
                        per_atom_denorm = normalizer.denorm(per_atom)
                        if task.element_references is not None:
                            atomic_numbers_tensor = torch.tensor(
                                atoms.get_atomic_numbers(), device=per_atom_denorm.device, dtype=torch.long
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
        out["per_atom_energies"] = per_atom_final

        # Embeddings (scalar)
        if self.captured_node_embeddings is not None:
            emb = self.captured_node_embeddings.detach().cpu().numpy()
            out["node_embeddings"] = emb
            out["node_embeddings_scalar"] = emb[:, 0, :]
        else:
            out["node_embeddings"] = None
            out["node_embeddings_scalar"] = None

        return out


def honly_relax(atoms, num_heavy: int, fmax: float, max_steps: int) -> Dict[str, Any]:
    """Run H-only relaxation: heavy atoms fixed, positions relax, cell fixed."""
    n = len(atoms)
    num_heavy = int(num_heavy)
    if num_heavy < 0 or num_heavy > n:
        raise ValueError(f"num_heavy out of bounds: {num_heavy} for n={n}")

    fixed = list(range(num_heavy))
    atoms.set_constraint(FixAtoms(indices=fixed))

    trajectory = []

    def log_step():
        e = atoms.get_potential_energy()
        trajectory.append(float(e))

    opt = FIRE(atoms)
    opt.attach(log_step, interval=1)
    converged = opt.run(fmax=float(fmax), steps=int(max_steps))

    return {
        "h_relax_converged": bool(converged),
        "h_relax_steps": int(len(trajectory)),
        "h_relax_trajectory": trajectory,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_manifest", required=True)
    ap.add_argument("--out_pt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_name", default="uma-s-1p1")
    ap.add_argument("--task_name", default="omc")
    ap.add_argument(
        "--chunk",
        action="store_true",
        help=(
            "If set, write `--out_pt` as an index file and store feature rows in multiple smaller shard .pt files "
            "under `<out_pt>.chunks/`. This avoids creating one huge monolithic .pt."
        ),
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Number of structures per .pt shard when --chunk is enabled.",
    )
    ap.add_argument(
        "--eta_log_every_sec",
        type=float,
        default=60.0,
        help="Print progress + ETA to stderr every N seconds (0 disables).",
    )
    args = ap.parse_args()

    with open(args.batch_manifest, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("batch_manifest must be a list")

    predictor = pretrained_mlip.get_predict_unit(args.model_name, device=args.device)
    extractor = UMAFeatureExtractorLite(predictor, task_name=args.task_name, device=args.device)

    pt_rows: List[Dict[str, Any]] = []  # used only when not chunking
    chunk_parts: List[str] = []
    chunk_dir = None
    chunk_rows: List[Dict[str, Any]] = []
    chunk_part_idx = 0
    out_json: Dict[str, Any] = {}

    print(f"Processing {len(entries)} entries on {args.device}...", file=sys.stderr)

    t0 = time.time()
    t_last = t0
    n_ok = 0
    n_err = 0

    for idx, entry in enumerate(entries):
        entry_id = entry.get("id", f"entry_{idx}")
        crystal_path = entry.get("crystal_path")
        num_heavy = entry.get("num_heavy")
        fmax = entry.get("relax_fmax", 0.05)
        max_steps = entry.get("relax_max_steps", 25)

        try:
            if crystal_path is None or num_heavy is None:
                raise ValueError("Missing crystal_path or num_heavy in entry")

            atoms = ase_read(crystal_path)
            atoms.info["charge"] = 0
            atoms.info["spin"] = 0

            # attach UMA calculator for relaxation
            calc = FAIRChemCalculator(predictor, task_name=args.task_name)
            atoms.calc = calc

            relax_info = honly_relax(atoms, int(num_heavy), float(fmax), int(max_steps))
            
            # Check for CUDA errors after relaxation (catches async assertion failures)
            if not cuda_sync_and_check():
                raise RuntimeError(f"CUDA error detected after H-only relaxation for {entry_id}")

            # Extract final features
            feats = extractor.extract(atoms)
            
            # Check for CUDA errors after feature extraction
            if not cuda_sync_and_check():
                raise RuntimeError(f"CUDA error detected after feature extraction for {entry_id}")

            row = {
                "id": entry_id,
                "refcode": entry.get("refcode"),
                "seed_idx": entry.get("seed_idx"),
                "kind": entry.get("kind"),
                "snapshot": "h_relaxed",
                "num_heavy": int(num_heavy),
                "num_atoms": int(len(atoms)),
                "atom_types": np.array(atoms.get_atomic_numbers(), dtype=np.int64),
                "positions": np.array(atoms.get_positions(), dtype=np.float32),
                "cell": np.array(atoms.cell.array, dtype=np.float32),
                "pbc": bool(np.all(atoms.pbc)),
                **relax_info,
                "energy": feats.get("energy"),
                "forces": feats.get("forces"),
                "per_atom_force_norms": feats.get("per_atom_force_norms"),
                "mean_force_norm": feats.get("mean_force_norm"),
                "max_force_norm": feats.get("max_force_norm"),
                "per_atom_energies": feats.get("per_atom_energies"),
                "node_embeddings": feats.get("node_embeddings"),
                "node_embeddings_scalar": feats.get("node_embeddings_scalar"),
            }

            # Pass-through extra metadata from the manifest (if present)
            for k in ("edge_index_h", "coords_h_pre_hrelax", "lattice_params", "lattice_matrix"):
                if k in entry and entry.get(k) is not None:
                    row[k] = entry.get(k)

            if args.chunk:
                if chunk_dir is None:
                    chunk_dir = args.out_pt + ".chunks"
                    os.makedirs(chunk_dir, exist_ok=True)

                # Flush shard if full
                if len(chunk_rows) >= int(args.chunk_size):
                    part_name = f"part_{chunk_part_idx:04d}.pt"
                    torch.save(chunk_rows, os.path.join(chunk_dir, part_name))
                    chunk_parts.append(part_name)
                    chunk_rows = []
                    chunk_part_idx += 1

                pt_index = len(chunk_rows)
                chunk_rows.append(row)
                pt_path_rel = f"{os.path.basename(chunk_dir)}/{f'part_{chunk_part_idx:04d}.pt'}"
            else:
                pt_index = len(pt_rows)
                pt_rows.append(row)
                pt_path_rel = os.path.basename(args.out_pt)

            out_json[entry_id] = {
                "pt_index": pt_index,
                "pt_path": pt_path_rel,
                "h_relax_converged": row["h_relax_converged"],
                "h_relax_steps": row["h_relax_steps"],
                "h_relax_trajectory_len": len(row["h_relax_trajectory"]),
                "energy": row["energy"],
                "mean_force_norm": row["mean_force_norm"],
                "max_force_norm": row["max_force_norm"],
            }
            n_ok += 1
        except Exception as e:
            out_json[entry_id] = {"error": str(e)}
            n_err += 1
            # Attempt to clear CUDA error state so subsequent structures can be processed
            cuda_clear_error_state()

        if args.eta_log_every_sec and args.eta_log_every_sec > 0:
            now = time.time()
            if (now - t_last) >= float(args.eta_log_every_sec) or (idx + 1) == len(entries):
                done = idx + 1
                elapsed = max(1e-6, now - t0)
                rate = done / elapsed
                remaining = len(entries) - done
                eta_sec = remaining / rate if rate > 0 else float("inf")
                print(
                    f"[HONLY] {done}/{len(entries)} done | ok={n_ok} err={n_err} | "
                    f"elapsed={elapsed/60:.1f} min | rate={rate:.2f}/s | ETA={eta_sec/60:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )
                t_last = now

        # cleanup input files to save space (optional)
        # Do NOT delete input CIFs here; caller typically uses a TemporaryDirectory.

    if args.chunk:
        if chunk_dir is None:
            chunk_dir = args.out_pt + ".chunks"
            os.makedirs(chunk_dir, exist_ok=True)
        # Flush any remaining rows
        if len(chunk_rows) > 0:
            part_name = f"part_{chunk_part_idx:04d}.pt"
            torch.save(chunk_rows, os.path.join(chunk_dir, part_name))
            chunk_parts.append(part_name)
        # Write a small index file at --out_pt that points to shards
        torch.save(
            {"sharded": True, "chunks_dir": os.path.basename(chunk_dir), "parts": chunk_parts},
            args.out_pt,
        )
    else:
        torch.save(pt_rows, args.out_pt)
    print(json.dumps(out_json))


if __name__ == "__main__":
    main()

