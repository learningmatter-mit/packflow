#!/usr/bin/env python3
"""
Postprocess an evaluation output directory to compute UMA quantities ONLY after:
  1) RDKit hydrogenation (AssignBondOrdersFromTemplate + AddHs(addCoords=True))
  2) UMA H-only relaxation under PBC (heavy atoms fixed; fmax=0.05; max_steps=25)
Optionally:
  3) UMA full relaxation (all atoms + lattice) starting from the H-only-relaxed structure,
     using scripts/calculate_uma_metrics.py, with fmax=0.05 and steps=full_relax_steps.

Writes:
  - all_seeds_data_hrelaxed_uma.json (same schema as input but with added UMA scalar fields + .pt indices)
  - hrelaxed_uma_features.pt (heavy arrays for GT+pred H-only-relaxed structures: forces, per-atom energies, embeddings, coords, etc.)
  - optionally full_relax_metrics.pt (heavy arrays for full relaxation results)

This script runs in the packflow environment and calls fairchem-side scripts via subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import JmolNN

# Ensure `import packflow` works regardless of current working directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from packflow.data.crystal_datamodule import CrystalDataset


FAIRCHEM_PY_DEFAULT = os.environ.get("FAIRCHEM_PYTHON", sys.executable)


def iter_top_level_json_array(path: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    buf = ""
    with open(path, "r") as f:
        # find '['
        while True:
            ch = f.read(1)
            if not ch:
                raise ValueError("Unexpected EOF while looking for '['")
            if ch.isspace():
                continue
            if ch != "[":
                raise ValueError(f"Expected '[' at start of JSON array, got {ch!r}")
            break

        while True:
            chunk = f.read(1024 * 1024)
            if not chunk and not buf:
                break
            buf += chunk
            idx = 0
            n = len(buf)

            while True:
                while idx < n and buf[idx].isspace():
                    idx += 1
                if idx < n and buf[idx] == ",":
                    idx += 1
                    continue
                while idx < n and buf[idx].isspace():
                    idx += 1
                if idx < n and buf[idx] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    break
                yield obj
                idx = end

            buf = buf[idx:]


def load_smiles_by_refcode(processed_data_dir: str) -> Dict[str, str]:
    test_pt = os.path.join(processed_data_dir, "test.pt")
    ds = CrystalDataset(test_pt)
    mapping: Dict[str, str] = {}
    for d in ds.cached_data:
        refcode = d.get("refcode")
        smiles = d.get("smiles")
        if refcode and smiles:
            mapping[str(refcode)] = str(smiles)
    return mapping


def _edge_index_to_undirected_bonds(edge_index: Any) -> List[Tuple[int, int]]:
    if edge_index is None:
        return []
    if not (isinstance(edge_index, (list, tuple)) and len(edge_index) == 2):
        raise ValueError("edge_index must be [2, E] list")
    src, dst = edge_index[0], edge_index[1]
    if len(src) != len(dst):
        raise ValueError("edge_index src/dst lengths differ")
    bonds = set()
    for a, b in zip(src, dst):
        a = int(a)
        b = int(b)
        if a == b:
            continue
        i, j = (a, b) if a < b else (b, a)
        bonds.add((i, j))
    return sorted(bonds)


def _assign_bond_orders_from_full_smiles(mol: Chem.Mol, smiles: str) -> Chem.Mol:
    """
    Assign bond orders from the full SMILES template (containing all Z molecules).
    
    This function only uses the complete SMILES string as the template to ensure
    all molecules in the unit cell get correct bond orders assigned. Using partial
    (single-molecule) templates would only fix 1/Z of the structure.
    
    Args:
        mol: RDKit molecule with all single bonds
        smiles: Full SMILES string (Z copies of molecule separated by '.')
    
    Returns:
        Molecule with correct bond orders, or raises ValueError if template matching fails.
    """
    tpl = Chem.MolFromSmiles(smiles)
    if tpl is None:
        raise ValueError(f"Failed to parse SMILES template: {smiles[:100]}...")
    return AllChem.AssignBondOrdersFromTemplate(tpl, mol)


def build_heavy_rdkit_mol(atom_types: List[int], edge_index: Any, cart_coords: np.ndarray, smiles: str) -> Chem.Mol:
    if cart_coords.shape != (len(atom_types), 3):
        raise ValueError("coords shape mismatch")
    rw = Chem.RWMol()
    for z in atom_types:
        rw.AddAtom(Chem.Atom(int(z)))
    for i, j in _edge_index_to_undirected_bonds(edge_index):
        if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conf, assignId=True)
    for i, (x, y, z) in enumerate(cart_coords):
        mol.GetConformer(0).SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
    mol = _assign_bond_orders_from_full_smiles(mol, smiles)
    Chem.SanitizeMol(mol)
    for a in mol.GetAtoms():
        a.UpdatePropertyCache(strict=False)
    return mol


def hydrogenate_from_smiles_template(atom_types: List[int], edge_index: Any, cart_coords: np.ndarray, smiles: str) -> Tuple[List[int], List[List[int]], np.ndarray, int]:
    """Return (atom_types_with_h, edge_index_with_h, coords_with_h, num_heavy)."""
    heavy = build_heavy_rdkit_mol(atom_types, edge_index, cart_coords, smiles)
    num_heavy = heavy.GetNumAtoms()
    mol_h = Chem.AddHs(heavy, addCoords=True)
    conf = mol_h.GetConformer()
    n = mol_h.GetNumAtoms()
    zs = [int(mol_h.GetAtomWithIdx(i).GetAtomicNum()) for i in range(n)]

    coords = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i, :] = [p.x, p.y, p.z]

    # edge_index from RDKit bonds
    edges = []
    for b in mol_h.GetBonds():
        i = int(b.GetBeginAtomIdx())
        j = int(b.GetEndAtomIdx())
        edges.append((i, j))
        edges.append((j, i))
    edges = sorted(set(edges))
    edge_index_h = [ [e[0] for e in edges], [e[1] for e in edges] ]
    return zs, edge_index_h, coords, num_heavy


def unwrap_cart_coords_with_edge_index(cart_coords: np.ndarray, lattice_matrix: np.ndarray, edge_index: Any) -> np.ndarray:
    """
    Use packflow's PBC unwrapping (`make_molecules_whole`) to avoid long bond segments that cross the unit cell.
    Inputs:
      - cart_coords: (N, 3) cartesian coords (Å)
      - lattice_matrix: (3, 3) lattice matrix (Å)
      - edge_index: [2, E] style (list/tuple) describing bonds
    Returns:
      - cart coords (N, 3) with molecules made whole in a consistent image
    """
    # Import here (not at module import time) so our sys.path insertion above has taken effect.
    from packflow.utils.crystal_utils import cart_to_frac, frac_to_cart, make_molecules_whole

    bonds = [{"atom1_idx": int(i), "atom2_idx": int(j)} for i, j in _edge_index_to_undirected_bonds(edge_index)]
    frac = cart_to_frac(np.asarray(cart_coords, dtype=np.float64), np.asarray(lattice_matrix, dtype=np.float64))
    whole_frac = make_molecules_whole(frac, bonds)
    return frac_to_cart(whole_frac, np.asarray(lattice_matrix, dtype=np.float64))


def wrap_cart_coords_into_cell(cart_coords: np.ndarray, lattice_matrix: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(lattice_matrix)
    frac = cart_coords @ inv
    frac = frac - np.floor(frac)
    return frac @ lattice_matrix


def write_cif(atom_types: List[int], cart_coords: np.ndarray, lattice_params: List[float], out_path: str) -> None:
    """Write a CIF file in P1 symmetry."""
    lat = Lattice.from_parameters(*[float(x) for x in lattice_params])
    species = [Element.from_Z(int(z)) for z in atom_types]
    struct = Structure(lat, species, cart_coords, coords_are_cartesian=True)
    struct.to(filename=out_path)


def _lattice_matrix_to_params_list(cell: np.ndarray) -> List[float]:
    lat = Lattice(np.asarray(cell, dtype=np.float64))
    return [float(lat.a), float(lat.b), float(lat.c), float(lat.alpha), float(lat.beta), float(lat.gamma)]


def _build_jmol_adjacency(atom_types: np.ndarray, cart_coords: np.ndarray, lattice_matrix: np.ndarray) -> np.ndarray:
    """
    Build a covalent adjacency matrix using pymatgen's JmolNN, mirroring FastCSP's check.
    Includes all atoms present (including H).
    """
    atom_types = np.asarray(atom_types, dtype=np.int64)
    cart_coords = np.asarray(cart_coords, dtype=np.float64)
    lattice_matrix = np.asarray(lattice_matrix, dtype=np.float64)
    if cart_coords.ndim != 2 or cart_coords.shape[1] != 3:
        raise ValueError(f"cart_coords must be (N,3), got {cart_coords.shape}")
    if lattice_matrix.shape != (3, 3):
        raise ValueError(f"lattice_matrix must be (3,3), got {lattice_matrix.shape}")
    if atom_types.shape[0] != cart_coords.shape[0]:
        raise ValueError("atom_types and cart_coords length mismatch")

    # Wrap into unit cell for robust PBC neighbor finding
    inv = np.linalg.inv(lattice_matrix)
    frac = cart_coords @ inv
    frac = frac - np.floor(frac)

    lat = Lattice(lattice_matrix)
    species = [Element.from_Z(int(z)) for z in atom_types.tolist()]
    struct = Structure(lat, species, frac, coords_are_cartesian=False)

    nn_info = JmolNN().get_all_nn_info(struct)
    n = len(nn_info)
    mat = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(len(nn_info[i])):
            mat[i, int(nn_info[i][j]["site_index"])] = 1
    return mat


def compute_connectivity_changed_honly_vs_full(
    honly_row: Optional[Dict[str, Any]],
    full_metrics: Optional[Dict[str, Any]],
) -> Optional[bool]:
    """
    Returns:
      - True/False if both sides are available and comparable
      - None if missing data or atom count mismatch (e.g., ASE merged overlapping atoms)
    """
    if honly_row is None or full_metrics is None:
        return None
    # H-only relaxed reference
    atom_types = honly_row.get("atom_types")
    pos_ref = honly_row.get("positions")
    cell_ref = honly_row.get("cell")
    if atom_types is None or pos_ref is None or cell_ref is None:
        return None
    # Full relaxed target
    pos_full = full_metrics.get("relaxed_cart_coords")
    cell_full = full_metrics.get("relaxed_lattice_matrix")
    if pos_full is None or cell_full is None:
        return None
    # Check for atom count mismatch (can happen if ASE merged overlapping atoms when reading CIF)
    n_atoms_honly = len(atom_types)
    n_atoms_full = len(pos_full)
    if n_atoms_honly != n_atoms_full:
        crystal_id = honly_row.get("id", "unknown")
        print(
            f"WARNING: Atom count mismatch for {crystal_id}: H-only has {n_atoms_honly} atoms, "
            f"full-relax has {n_atoms_full}. This may indicate overlapping atoms in the structure. "
            f"Skipping connectivity check.",
            file=sys.stderr,
        )
        return None
    # Compute covalent adjacency and compare
    a0 = _build_jmol_adjacency(np.array(atom_types), np.array(pos_ref), np.array(cell_ref))
    a1 = _build_jmol_adjacency(np.array(atom_types), np.array(pos_full), np.array(cell_full))
    return (not np.array_equal(a0, a1))


def run_fairchem_honly_features(
    fairchem_python: str,
    project_root: str,
    manifest_entries: List[Dict[str, Any]],
    out_pt_path: str,
    device: str,
    chunk: bool = False,
) -> Dict[str, Any]:
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers", "uma_h_relax.py")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        manifest_path = f.name
        json.dump(manifest_entries, f)
    try:
        # Force offline HF behavior on compute nodes (avoid noisy HTTPS retries).
        # This makes fairchem/UMA load strictly from local cache and fail fast if missing.
        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

        cmd = [
            fairchem_python,
            script_path,
            "--batch_manifest",
            manifest_path,
            "--out_pt",
            out_pt_path,
            "--device",
            device,
            "--eta_log_every_sec",
            "60",
        ]
        if chunk:
            cmd.append("--chunk")
        # Stream stderr to parent (Slurm log) so progress/ETA is visible in real time.
        # Keep stdout captured so we can parse the final JSON line.
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, cwd=project_root, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"H-only UMA script failed: rc={res.returncode}\nSTDOUT:\n{res.stdout}")
        lines = (res.stdout or "").strip().splitlines()
        if not lines:
            raise RuntimeError("No stdout from H-only UMA script.")
        return json.loads(lines[-1])
    finally:
        try:
            os.remove(manifest_path)
        except OSError:
            pass


class _ShardedPtWriter:
    """
    Incrementally build a sharded `.pt` (index + part_*.pt shards) without holding all rows in memory.

    This is used to make the overall pipeline resilient to a single bad structure poisoning a CUDA process:
    we can run fairchem on smaller subsets (e.g. per-crystal) and append successful rows into one global
    sharded output for the eval directory.
    """

    def __init__(self, out_pt_path: str, chunk_size: int = 200):
        self.out_pt_path = out_pt_path
        self.chunk_size = int(chunk_size)
        self.chunk_dir = out_pt_path + ".chunks"
        os.makedirs(self.chunk_dir, exist_ok=True)

        self._parts: List[str] = []
        self._part_idx: int = 0
        self._rows: List[Dict[str, Any]] = []

    @property
    def chunks_dir_basename(self) -> str:
        return os.path.basename(self.chunk_dir)

    def append_row(self, row: Dict[str, Any]) -> Tuple[str, int]:
        if len(self._rows) >= self.chunk_size:
            self._flush()
        part_name = f"part_{self._part_idx:04d}.pt"
        idx_in_part = len(self._rows)
        self._rows.append(row)
        return part_name, idx_in_part

    def _flush(self) -> None:
        if not self._rows:
            return
        part_name = f"part_{self._part_idx:04d}.pt"
        torch.save(self._rows, os.path.join(self.chunk_dir, part_name))
        self._parts.append(part_name)
        self._rows = []
        self._part_idx += 1

    def finalize(self) -> None:
        self._flush()
        torch.save(
            {"sharded": True, "chunks_dir": self.chunks_dir_basename, "parts": self._parts},
            self.out_pt_path,
        )


def run_fairchem_full_relax_metrics(
    fairchem_python: str,
    project_root: str,
    manifest_entries: List[Dict[str, Any]],
    device: str,
) -> Dict[str, Any]:
    """Uses scripts/calculate_uma_metrics.py in batch mode (crystal-only entries allowed)."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers", "uma_metrics.py")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        manifest_path = f.name
        json.dump(manifest_entries, f)
    try:
        # Force offline HF behavior on compute nodes (avoid noisy HTTPS retries).
        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

        cmd = [fairchem_python, script_path, "--batch_manifest", manifest_path, "--device", device]
        # Stream stderr to parent (Slurm log) so existing per-10 progress prints are visible.
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, cwd=project_root, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"Full-relax UMA script failed: rc={res.returncode}\nSTDOUT:\n{res.stdout}")
        lines = (res.stdout or "").strip().splitlines()
        if not lines:
            raise RuntimeError("No stdout from full-relax UMA script.")
        return json.loads(lines[-1])
    finally:
        try:
            os.remove(manifest_path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="Directory containing all_seeds_data.json")
    ap.add_argument("--processed_data_dir", required=True, help="Directory containing test.pt for SMILES lookup")
    ap.add_argument("--fairchem_python", default=FAIRCHEM_PY_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--h_only_relax_fmax", type=float, default=0.05)
    ap.add_argument("--h_only_relax_max_steps", type=int, default=25)
    ap.add_argument("--full_relaxation_steps", type=int, default=0, help="0 disables full relaxation")
    ap.add_argument(
        "--chunk",
        action="store_true",
        help=(
            "If set, enable chunked UMA feature output on the fairchem side (smaller .pt shards + index file). "
            "Also avoids rewriting large .pt files in this postprocess step."
        ),
    )
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="If set, also write hydrogenated (pre-H-relax) comparison visualizations using edge_index_h and all atoms including H.",
    )
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    in_json = os.path.join(args.eval_dir, "all_seeds_data.json")
    out_json = os.path.join(args.eval_dir, "all_seeds_data_hrelaxed_uma.json")
    out_pt = os.path.join(args.eval_dir, "hrelaxed_uma_features.pt")
    out_full_pt = os.path.join(args.eval_dir, "full_relax_metrics.pt")
    out_readme = os.path.join(args.eval_dir, "README_hrelaxed_uma.md")

    smiles_by_ref = load_smiles_by_refcode(args.processed_data_dir)

    # Optional H-added visualization (pre-H-relax) setup
    viz_dir = None
    visualize_crystal_comparison = None
    if args.visualize:
        try:
            # Reuse the exact visualization helper used pre-hydrogenation.
            from packflow.evaluation.evaluator import visualize_crystal_comparison as _viz

            visualize_crystal_comparison = _viz
            viz_dir = os.path.join(args.eval_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
        except Exception as e:
            print(f"WARNING: --visualize requested but failed to import visualization helper: {e}")
            visualize_crystal_comparison = None
            viz_dir = None

    # We will create temporary CIFs for H-only relaxation inputs
    with tempfile.TemporaryDirectory() as tmp:
        honly_manifest: List[Dict[str, Any]] = []
        honly_manifest_by_refcode: Dict[str, List[Dict[str, Any]]] = {}
        # map id -> structural metadata to attach into output json
        id_to_struct_meta: Dict[str, Any] = {}
        # For preds only: True if this pred is from PackFlow (atom ordering matches GT; i.e. no genarris reordering)
        id_to_is_packflow_pred: Dict[str, bool] = {}

        # We stream input and also build a lightweight in-memory index of which seeds exist.
        # For output JSON we will stream again and write incrementally (to avoid huge RAM).
        t_build0 = None
        try:
            import time as _time

            t_build0 = _time.time()
        except Exception:
            t_build0 = None
        n_crystals = 0
        n_structs = 0
        for crystal in iter_top_level_json_array(in_json):
            n_crystals += 1
            bm = crystal.get("best_metrics", {})
            refcode = bm.get("refcode")
            if not refcode:
                continue
            refcode = str(refcode)
            smiles = smiles_by_ref.get(refcode)
            if smiles is None:
                continue
            representative_seed_idx = bm.get("representative_seed_idx", 0)
            try:
                representative_seed_idx = int(representative_seed_idx)
            except Exception:
                representative_seed_idx = 0

            # Identify GT from first seed (same for all seeds)
            seeds = crystal.get("all_seeds_data", [])
            if not seeds:
                continue
            first_seed = seeds[0]
            gt_atom_types = first_seed.get("gt_atom_types")
            gt_edge_index = first_seed.get("gt_edge_index")
            gt_coords = np.array(first_seed.get("gt_coords"), dtype=np.float64)
            gt_lat_params = first_seed.get("gt_lattice")
            gt_lat_matrix = np.array(first_seed.get("gt_lattice_matrix"), dtype=np.float64)
            if gt_atom_types is None or gt_edge_index is None or gt_lat_params is None:
                continue

            gt_atom_types_list = [int(x) for x in gt_atom_types]
            try:
                gt_h_zs, gt_h_edge_index, gt_h_coords, gt_num_heavy = hydrogenate_from_smiles_template(
                    gt_atom_types_list, gt_edge_index, gt_coords, smiles
                )
            except ValueError as e:
                print(f"WARNING: Skipping {refcode} - GT hydrogenation failed: {e}", file=sys.stderr)
                continue
            gt_h_coords = wrap_cart_coords_into_cell(gt_h_coords, gt_lat_matrix)
            gt_id = f"{refcode}_gt_hrelax"
            gt_cif = os.path.join(tmp, f"{gt_id}.cif")
            write_cif(gt_h_zs, gt_h_coords, [float(x) for x in gt_lat_params], gt_cif)

            gt_entry = {
                "id": gt_id,
                "crystal_path": gt_cif,
                "num_heavy": gt_num_heavy,
                "refcode": refcode,
                "seed_idx": None,
                "kind": "gt",
                "relax_fmax": args.h_only_relax_fmax,
                "relax_max_steps": args.h_only_relax_max_steps,
                # Pass through structural metadata so the fairchem-side writer can include it
                # without a second full rewrite of a potentially huge .pt file.
                "edge_index_h": gt_h_edge_index,
                "coords_h_pre_hrelax": gt_h_coords.tolist(),
                "lattice_params": [float(x) for x in gt_lat_params],
                "lattice_matrix": gt_lat_matrix.tolist(),
            }
            honly_manifest.append(gt_entry)
            honly_manifest_by_refcode.setdefault(refcode, []).append(gt_entry)
            n_structs += 1
            id_to_struct_meta[gt_id] = {
                "atom_types_h": gt_h_zs,
                "edge_index_h": gt_h_edge_index,
                "coords_h_pre_hrelax": gt_h_coords.tolist(),
                "lattice_params": [float(x) for x in gt_lat_params],
                "lattice_matrix": gt_lat_matrix.tolist(),
            }

            for seed in seeds:
                seed_idx = seed.get("seed_idx")
                if seed_idx is None:
                    continue
                try:
                    seed_idx_int = int(seed_idx)
                except Exception:
                    continue

                pred_coords = np.array(seed.get("pred_coords"), dtype=np.float64)
                pred_lat_params = seed.get("pred_lattice") or seed.get("gt_lattice")
                pred_lat_matrix = np.array(seed.get("pred_lattice_matrix") or seed.get("gt_lattice_matrix"), dtype=np.float64)

                pred_atom_types = seed.get("pred_atom_types") or seed.get("gt_atom_types")
                pred_edge_index = seed.get("pred_edge_index") or seed.get("gt_edge_index")
                if pred_atom_types is None or pred_edge_index is None or pred_lat_params is None:
                    continue

                pred_atom_types_list = [int(x) for x in pred_atom_types]
                try:
                    pred_h_zs, pred_h_edge_index, pred_h_coords, pred_num_heavy = hydrogenate_from_smiles_template(
                        pred_atom_types_list, pred_edge_index, pred_coords, smiles
                    )
                except ValueError as e:
                    print(f"WARNING: Skipping {refcode} seed {seed_idx} - prediction hydrogenation failed: {e}", file=sys.stderr)
                    continue
                pred_h_coords = wrap_cart_coords_into_cell(pred_h_coords, pred_lat_matrix)
                pred_id = f"{refcode}_seed_{int(seed_idx)}_pred_hrelax"
                # PackFlow preds maintain GT atom ordering (no pred_atom_types field); Genarris provides pred_atom_types and can reorder.
                id_to_is_packflow_pred[pred_id] = seed.get("pred_atom_types") is None
                pred_cif = os.path.join(tmp, f"{pred_id}.cif")
                write_cif(pred_h_zs, pred_h_coords, [float(x) for x in pred_lat_params], pred_cif)

                # Optional: visualize H-added (pre-H-relax) comparison for the representative seed only
                if (
                    visualize_crystal_comparison is not None
                    and viz_dir is not None
                    and seed_idx_int == representative_seed_idx
                ):
                    try:
                        viz_path = os.path.join(
                            viz_dir, f"{refcode}_comparison_seed{representative_seed_idx}_hadded_pre_relax.png"
                        )
                        # Important: unwrap GT hydrogenated coords so bonds don't draw across the unit cell.
                        gt_h_coords_whole = unwrap_cart_coords_with_edge_index(gt_h_coords, gt_lat_matrix, gt_h_edge_index)
                        gt_cd = {
                            "whole_cartesian_coords": torch.tensor(gt_h_coords_whole, dtype=torch.float32),
                            "lattice_1": torch.tensor([float(x) for x in gt_lat_params], dtype=torch.float32),
                            "atom_types": torch.tensor(gt_h_zs, dtype=torch.int64),
                            "edge_index": torch.tensor(gt_h_edge_index, dtype=torch.int64),
                        }
                        visualize_crystal_comparison(
                            gt_cd,
                            torch.tensor(pred_h_coords, dtype=torch.float32),
                            torch.tensor([float(x) for x in pred_lat_params], dtype=torch.float32),
                            f"{refcode} (H-added, pre-H-relax)",
                            viz_path,
                            draw_pred_bonds=True,
                            center_pred=True,
                            pred_atom_types=np.array(pred_h_zs, dtype=np.int64),
                            pred_edge_index=torch.tensor(pred_h_edge_index, dtype=torch.int64),
                        )
                    except Exception as e:
                        print(f"WARNING: Failed to write H-added visualization for {refcode} seed {seed_idx_int}: {e}")

                pred_entry = {
                    "id": pred_id,
                    "crystal_path": pred_cif,
                    "num_heavy": pred_num_heavy,
                    "refcode": refcode,
                    "seed_idx": seed_idx_int,
                    "kind": "pred",
                    "relax_fmax": args.h_only_relax_fmax,
                    "relax_max_steps": args.h_only_relax_max_steps,
                    # Pass through structural metadata so the fairchem-side writer can include it
                    # without a second full rewrite of a potentially huge .pt file.
                    "edge_index_h": pred_h_edge_index,
                    "coords_h_pre_hrelax": pred_h_coords.tolist(),
                    "lattice_params": [float(x) for x in pred_lat_params],
                    "lattice_matrix": pred_lat_matrix.tolist(),
                }
                honly_manifest.append(pred_entry)
                honly_manifest_by_refcode.setdefault(refcode, []).append(pred_entry)
                n_structs += 1
                id_to_struct_meta[pred_id] = {
                    "atom_types_h": pred_h_zs,
                    "edge_index_h": pred_h_edge_index,
                    "coords_h_pre_hrelax": pred_h_coords.tolist(),
                    "lattice_params": [float(x) for x in pred_lat_params],
                    "lattice_matrix": pred_lat_matrix.tolist(),
                }

            if n_crystals % 5 == 0:
                try:
                    import time as _time

                    if t_build0 is not None:
                        elapsed_min = (_time.time() - t_build0) / 60.0
                        print(
                            f"[postprocess] built manifest for {n_crystals} crystals "
                            f"({n_structs} structures) | elapsed={elapsed_min:.1f} min",
                            file=sys.stderr,
                            flush=True,
                        )
                except Exception:
                    pass

        if t_build0 is not None:
            try:
                import time as _time

                elapsed_min = (_time.time() - t_build0) / 60.0
                print(
                    f"[postprocess] manifest complete: {n_crystals} crystals, {n_structs} structures | elapsed={elapsed_min:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass

        # Run H-only relaxation + features and save .pt
        #
        # IMPORTANT: A single CUDA device-side assert inside fairchem/UMA can "poison" the entire process,
        # causing all subsequent structures in that invocation to fail. To make the run resilient, when
        # chunking is enabled we run fairchem per-crystal (refcode) and append successful rows into one
        # global sharded output. If a single crystal fails, we skip it and continue.
        if args.chunk:
            honly_results: Dict[str, Any] = {}
            writer = _ShardedPtWriter(out_pt, chunk_size=200)
            # Process refcodes in deterministic order (stable logs / reproducibility)
            for refcode in sorted(honly_manifest_by_refcode.keys()):
                entries = honly_manifest_by_refcode.get(refcode) or []
                if not entries:
                    continue

                tmp_pt = os.path.join(tmp, f"honly_{refcode}.pt")
                try:
                    # Run fairchem on a small, isolated subset (this refcode only).
                    # We do NOT enable fairchem-side chunking here; we chunk globally in this parent process.
                    local_results = run_fairchem_honly_features(
                        fairchem_python=args.fairchem_python,
                        project_root=project_root,
                        manifest_entries=entries,
                        out_pt_path=tmp_pt,
                        device=args.device,
                        chunk=False,
                    )
                except Exception as e:
                    print(f"WARNING: UMA H-only failed for {refcode}; skipping this crystal. Error: {e}", file=sys.stderr)
                    continue

                # Load successful rows produced for this crystal and append to the global sharded output.
                try:
                    local_rows = torch.load(tmp_pt, map_location="cpu", weights_only=False)
                except Exception as e:
                    print(
                        f"WARNING: Could not read UMA H-only output for {refcode} at {tmp_pt}; skipping. Error: {e}",
                        file=sys.stderr,
                    )
                    continue
                finally:
                    try:
                        if os.path.exists(tmp_pt):
                            os.remove(tmp_pt)
                    except OSError:
                        pass

                if not isinstance(local_rows, list):
                    print(
                        f"WARNING: Unexpected UMA H-only output type for {refcode}: {type(local_rows)}; skipping.",
                        file=sys.stderr,
                    )
                    continue

                for row in local_rows:
                    rid = row.get("id")
                    if not rid:
                        continue
                    rid = str(rid)
                    # Append row to global sharded output
                    part_name, idx_in_part = writer.append_row(row)
                    m = local_results.get(rid, {}) if isinstance(local_results, dict) else {}

                    # Mirror fairchem-side schema: pt_index is within the referenced part file
                    honly_results[rid] = {
                        "pt_index": int(idx_in_part),
                        "pt_path": f"{writer.chunks_dir_basename}/{part_name}",
                        "h_relax_converged": m.get("h_relax_converged", row.get("h_relax_converged")),
                        "h_relax_steps": m.get("h_relax_steps", row.get("h_relax_steps")),
                        "h_relax_trajectory_len": m.get("h_relax_trajectory_len", len(row.get("h_relax_trajectory") or [])),
                        "energy": m.get("energy", row.get("energy")),
                        "mean_force_norm": m.get("mean_force_norm", row.get("mean_force_norm")),
                        "max_force_norm": m.get("max_force_norm", row.get("max_force_norm")),
                    }

            writer.finalize()
        else:
            honly_results = run_fairchem_honly_features(
                fairchem_python=args.fairchem_python,
                project_root=project_root,
                manifest_entries=honly_manifest,
                out_pt_path=out_pt,
                device=args.device,
                chunk=args.chunk,
            )

        # Determine whether the H-relaxed features output is sharded (chunked) once, up front.
        hrelaxed_pt_is_sharded = False
        hrelaxed_pt_chunks_dir = None
        try:
            _obj = torch.load(out_pt, map_location="cpu", weights_only=False)
            if isinstance(_obj, dict) and _obj.get("sharded"):
                hrelaxed_pt_is_sharded = True
                hrelaxed_pt_chunks_dir = _obj.get("chunks_dir")
        except Exception:
            pass

        def _iter_hrelaxed_pt_rows(path: str):
            """Iterate H-only relaxed .pt rows regardless of sharded/non-sharded format."""
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, list):
                for row in obj:
                    yield row
                return
            if isinstance(obj, dict) and obj.get("sharded"):
                chunks_dir = obj.get("chunks_dir")
                parts = obj.get("parts") or []
                if chunks_dir is None:
                    raise ValueError("Sharded pt index missing 'chunks_dir'")
                base_dir = os.path.dirname(path)
                for rel in parts:
                    part_path = os.path.join(base_dir, chunks_dir, rel)
                    for row in torch.load(part_path, map_location="cpu", weights_only=False):
                        yield row
                return
            raise ValueError(f"Unknown hrelaxed pt format at {path}")

        # Build a compact lookup of H-only relaxed rows by id (needed for full-relax augmentation + connectivity check).
        # NOTE: This is a lightweight in-memory dict (arrays are stored in rows already); we keep references.
        honly_row_by_id: Dict[str, Dict[str, Any]] = {}
        for row in _iter_hrelaxed_pt_rows(out_pt):
            rid = row.get("id")
            if rid:
                honly_row_by_id[str(rid)] = row

        # Optionally run full relaxation (all atoms + lattice) starting from H-only-relaxed structures stored in out_pt
        full_results = None
        full_pt_rows: List[Dict[str, Any]] = []
        full_id_to_pt_index: Dict[str, int] = {}
        if args.full_relaxation_steps > 0:
            # Write CIFs from H-only-relaxed structures as starting points
            full_manifest = []
            for row in _iter_hrelaxed_pt_rows(out_pt):
                mid = row.get("id")
                if not mid:
                    continue
                atom_types = row.get("atom_types")
                positions = row.get("positions")
                lattice_params = row.get("lattice_params")
                cell = row.get("cell")
                if atom_types is None or positions is None or lattice_params is None or cell is None:
                    continue
                cif_path = os.path.join(tmp, f"{mid}_hrelaxed_start.cif")
                write_cif(
                    [int(x) for x in np.array(atom_types).tolist()],
                    np.array(positions, dtype=np.float64),
                    [float(x) for x in lattice_params],
                    cif_path,
                )
                full_manifest.append(
                    {
                        "id": mid,
                        "crystal_path": cif_path,
                        "relaxation_steps": int(args.full_relaxation_steps),
                    }
                )

            full_results = run_fairchem_full_relax_metrics(
                fairchem_python=args.fairchem_python,
                project_root=project_root,
                manifest_entries=full_manifest,
                device=args.device,
            )

            # Cache H-only adjacency matrices (JmolNN) per id to avoid recomputing for PackFlow GT match checks.
            honly_adj_cache: Dict[str, np.ndarray] = {}
            def _get_honly_adj(mid: str) -> Optional[np.ndarray]:
                if mid in honly_adj_cache:
                    return honly_adj_cache[mid]
                row = honly_row_by_id.get(mid)
                if row is None:
                    return None
                atom_types = row.get("atom_types")
                pos = row.get("positions")
                cell = row.get("cell")
                if atom_types is None or pos is None or cell is None:
                    return None
                mat = _build_jmol_adjacency(np.array(atom_types), np.array(pos), np.array(cell))
                honly_adj_cache[mid] = mat
                return mat

            # Convert to pt rows (heavy arrays)
            for mid, m in full_results.items():
                # Augment with atom_types (from H-only relaxed row; includes hydrogens and respects Genarris ordering)
                honly_row = honly_row_by_id.get(str(mid))
                atom_types = None
                edge_index_h = None
                refcode = None
                seed_idx = None
                kind = None
                if honly_row is not None:
                    atom_types = honly_row.get("atom_types")
                    edge_index_h = honly_row.get("edge_index_h")
                    refcode = honly_row.get("refcode")
                    seed_idx = honly_row.get("seed_idx")
                    kind = honly_row.get("kind")

                connectivity_changed = compute_connectivity_changed_honly_vs_full(honly_row, m)

                # Extra PackFlow-only check:
                # If this is a PackFlow prediction, also ensure its H-only covalent adjacency matches the GT H-only adjacency.
                # If it does not, treat as connectivity_changed=True (since topology already differs vs GT).
                try:
                    if (
                        kind == "pred"
                        and refcode is not None
                        and id_to_is_packflow_pred.get(str(mid), False)
                        and connectivity_changed is not None
                    ):
                        gt_id = f"{str(refcode)}_gt_hrelax"
                        a_pred = _get_honly_adj(str(mid))
                        a_gt = _get_honly_adj(gt_id)
                        if a_pred is not None and a_gt is not None:
                            if not np.array_equal(a_pred, a_gt):
                                connectivity_changed = True
                except Exception:
                    # Be conservative: if this check fails unexpectedly, do not override the primary value.
                    pass

                pt_row = {
                    "id": mid,
                    "refcode": refcode,
                    "seed_idx": seed_idx,
                    "kind": kind,
                    "atom_types": atom_types,
                    "edge_index_h": edge_index_h,
                    "connectivity_changed": connectivity_changed,
                    **m,
                }
                full_id_to_pt_index[str(mid)] = len(full_pt_rows)
                full_pt_rows.append(pt_row)

            # Save full-relax metrics (chunked if requested)
            if args.chunk:
                chunk_dir = out_full_pt + ".chunks"
                os.makedirs(chunk_dir, exist_ok=True)
                chunk_size = 200
                parts: List[str] = []
                part_idx = 0
                for i in range(0, len(full_pt_rows), chunk_size):
                    part_name = f"part_{part_idx:04d}.pt"
                    torch.save(full_pt_rows[i : i + chunk_size], os.path.join(chunk_dir, part_name))
                    parts.append(part_name)
                    part_idx += 1
                torch.save(
                    {"sharded": True, "chunks_dir": os.path.basename(chunk_dir), "parts": parts},
                    out_full_pt,
                )
            else:
                torch.save(full_pt_rows, out_full_pt)

            # Optional: visualize GT vs representative pred AFTER full relaxation
            if args.visualize and visualize_crystal_comparison is not None and viz_dir is not None:
                try:
                    # We'll stream input JSON again later; for viz we can do a quick pass over it here (small cost).
                    for crystal in iter_top_level_json_array(in_json):
                        bm = crystal.get("best_metrics", {}) or {}
                        refcode = bm.get("refcode")
                        if not refcode:
                            continue
                        refcode = str(refcode)
                        rep_seed = bm.get("representative_seed_idx", 0)
                        try:
                            rep_seed = int(rep_seed)
                        except Exception:
                            rep_seed = 0

                        gt_id = f"{refcode}_gt_hrelax"
                        pred_id = f"{refcode}_seed_{rep_seed}_pred_hrelax"
                        if gt_id not in full_results or pred_id not in full_results:
                            continue

                        gt_m = full_results.get(gt_id, {}) or {}
                        pr_m = full_results.get(pred_id, {}) or {}
                        gt_pos = gt_m.get("relaxed_cart_coords")
                        gt_cell = gt_m.get("relaxed_lattice_matrix")
                        pr_pos = pr_m.get("relaxed_cart_coords")
                        pr_cell = pr_m.get("relaxed_lattice_matrix")
                        if gt_pos is None or gt_cell is None or pr_pos is None or pr_cell is None:
                            continue

                        # Use edge_index_h and atom types from the H-only rows (assume preserved across full relax).
                        gt_row = honly_row_by_id.get(gt_id)
                        pr_row = honly_row_by_id.get(pred_id)
                        if gt_row is None or pr_row is None:
                            continue
                        gt_edge = gt_row.get("edge_index_h")
                        pr_edge = pr_row.get("edge_index_h")
                        gt_atom_types = gt_row.get("atom_types")
                        pr_atom_types = pr_row.get("atom_types")
                        if gt_edge is None or pr_edge is None or gt_atom_types is None or pr_atom_types is None:
                            continue

                        gt_cell_np = np.array(gt_cell, dtype=np.float64)
                        pr_cell_np = np.array(pr_cell, dtype=np.float64)
                        gt_pos_np = np.array(gt_pos, dtype=np.float64)
                        pr_pos_np = np.array(pr_pos, dtype=np.float64)

                        # Unwrap for nicer bond drawing
                        try:
                            gt_pos_whole = unwrap_cart_coords_with_edge_index(gt_pos_np, gt_cell_np, gt_edge)
                        except Exception:
                            gt_pos_whole = gt_pos_np
                        try:
                            pr_pos_whole = unwrap_cart_coords_with_edge_index(pr_pos_np, pr_cell_np, pr_edge)
                        except Exception:
                            pr_pos_whole = pr_pos_np

                        gt_lat_params = _lattice_matrix_to_params_list(gt_cell_np)
                        pr_lat_params = _lattice_matrix_to_params_list(pr_cell_np)

                        viz_path = os.path.join(
                            viz_dir, f"{refcode}_comparison_seed{rep_seed}_full_relax.png"
                        )
                        gt_cd = {
                            "whole_cartesian_coords": torch.tensor(gt_pos_whole, dtype=torch.float32),
                            "lattice_1": torch.tensor(gt_lat_params, dtype=torch.float32),
                            "atom_types": torch.tensor(np.array(gt_atom_types, dtype=np.int64), dtype=torch.int64),
                            "edge_index": torch.tensor(gt_edge, dtype=torch.int64),
                        }
                        visualize_crystal_comparison(
                            gt_cd,
                            torch.tensor(pr_pos_whole, dtype=torch.float32),
                            torch.tensor(pr_lat_params, dtype=torch.float32),
                            f"{refcode} (Full-relaxed)",
                            viz_path,
                            draw_pred_bonds=True,
                            center_pred=True,
                            pred_atom_types=np.array(pr_atom_types, dtype=np.int64),
                            pred_edge_index=torch.tensor(pr_edge, dtype=torch.int64),
                        )
                except Exception as e:
                    print(f"WARNING: Failed to write full-relax visualizations: {e}")

        # Write output JSON by streaming input and adding UMA scalar fields + pt indices
        with open(out_json, "w") as f_out:
            f_out.write("[\n")
            first = True

            for crystal in iter_top_level_json_array(in_json):
                bm = crystal.get("best_metrics", {})
                refcode = bm.get("refcode")
                if refcode:
                    refcode = str(refcode)
                    gt_id = f"{refcode}_gt_hrelax"
                    if gt_id in honly_results and "pt_index" in honly_results[gt_id]:
                        gt_pt_path = honly_results[gt_id].get("pt_path", os.path.basename(out_pt))
                        bm.setdefault("hrelaxed_uma", {})["gt_hrelaxed_id"] = gt_id
                        bm["hrelaxed_uma"]["gt_hrelaxed_pt_index"] = honly_results[gt_id].get("pt_index")
                        bm["hrelaxed_uma"]["gt_hrelaxed_pt_path"] = gt_pt_path
                        bm["hrelaxed_uma"]["gt_hrelaxed_energy"] = honly_results[gt_id].get("energy")
                        bm["hrelaxed_uma"]["gt_hrelaxed_mean_force_norm"] = honly_results[gt_id].get("mean_force_norm")
                        bm["hrelaxed_uma"]["gt_h_relax_converged"] = honly_results[gt_id].get("h_relax_converged")
                        bm["hrelaxed_uma"]["gt_h_relax_steps"] = honly_results[gt_id].get("h_relax_steps")
                        bm["hrelaxed_uma"]["hrelaxed_features_pt_path"] = os.path.basename(out_pt)
                        bm["hrelaxed_uma"]["full_relaxation_steps"] = int(args.full_relaxation_steps)
                        bm["hrelaxed_uma"]["full_relax_metrics_pt_path"] = (
                            os.path.basename(out_full_pt) if args.full_relaxation_steps > 0 else None
                        )
                        bm["hrelaxed_uma"]["hrelaxed_features_pt_is_sharded"] = bool(hrelaxed_pt_is_sharded)
                        bm["hrelaxed_uma"]["hrelaxed_features_pt_chunks_dir"] = hrelaxed_pt_chunks_dir
                        # Save Z from best_metrics if present
                        bm["hrelaxed_uma"]["Z_num_molecules"] = bm.get("num_molecules")
                        # Full relax GT pointer (if present)
                        if args.full_relaxation_steps > 0 and gt_id in full_id_to_pt_index:
                            bm["hrelaxed_uma"]["gt_full_relax_pt_index"] = full_id_to_pt_index[gt_id]

                # Per seed entries
                for seed in crystal.get("all_seeds_data", []) or []:
                    s_ref = seed.get("refcode")
                    s_idx = seed.get("seed_idx")
                    if s_ref is None or s_idx is None:
                        continue
                    sid = f"{s_ref}_seed_{int(s_idx)}_pred_hrelax"
                    if sid in honly_results and "pt_index" in honly_results[sid]:
                        sid_pt_path = honly_results[sid].get("pt_path", os.path.basename(out_pt))
                        seed.setdefault("hrelaxed_uma", {})["pred_hrelaxed_id"] = sid
                        seed["hrelaxed_uma"]["pred_hrelaxed_pt_index"] = honly_results[sid].get("pt_index")
                        seed["hrelaxed_uma"]["pred_hrelaxed_pt_path"] = sid_pt_path
                        seed["hrelaxed_uma"]["h_relax_converged"] = honly_results[sid].get("h_relax_converged")
                        seed["hrelaxed_uma"]["h_relax_steps"] = honly_results[sid].get("h_relax_steps")
                        seed["hrelaxed_uma"]["pred_hrelaxed_energy"] = honly_results[sid].get("energy")
                        seed["hrelaxed_uma"]["pred_hrelaxed_mean_force_norm"] = honly_results[sid].get("mean_force_norm")
                        seed["hrelaxed_uma"]["pred_hrelaxed_max_force_norm"] = honly_results[sid].get("max_force_norm")
                        seed["hrelaxed_uma"]["Z_num_molecules"] = bm.get("num_molecules")
                        if args.full_relaxation_steps > 0 and sid in full_id_to_pt_index:
                            seed["hrelaxed_uma"]["pred_full_relax_pt_index"] = full_id_to_pt_index[sid]

                # Add top-level pointers
                crystal.setdefault("hrelaxed_uma_meta", {})["hrelaxed_features_pt_path"] = os.path.basename(out_pt)
                crystal["hrelaxed_uma_meta"]["hrelaxed_features_pt_is_sharded"] = bool(hrelaxed_pt_is_sharded)
                crystal["hrelaxed_uma_meta"]["hrelaxed_features_pt_chunks_dir"] = hrelaxed_pt_chunks_dir
                if args.full_relaxation_steps > 0:
                    crystal["hrelaxed_uma_meta"]["full_relax_metrics_pt_path"] = os.path.basename(out_full_pt)
                    # Mirror sharding metadata for full-relax metrics
                    crystal["hrelaxed_uma_meta"]["full_relax_metrics_pt_is_sharded"] = bool(args.chunk)
                    crystal["hrelaxed_uma_meta"]["full_relax_metrics_pt_chunks_dir"] = (
                        os.path.basename(out_full_pt + ".chunks") if args.chunk else None
                    )
                crystal["best_metrics"] = bm

                if not first:
                    f_out.write(",\n")
                first = False
                json.dump(crystal, f_out)

            f_out.write("\n]\n")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_pt}")
    if args.full_relaxation_steps > 0:
        print(f"Wrote: {out_full_pt}")

    # --- Write README describing outputs and schemas ---
    readme_lines: List[str] = []
    readme_lines.append("# H-relaxed UMA postprocess outputs\n")
    readme_lines.append(
        "This folder contains evaluation results from `evaluate_test_set_metrics.py` plus additional UMA quantities computed **after**:\n"
    )
    readme_lines.append("1. RDKit hydrogenation (bond orders from SMILES template, then `AddHs(addCoords=True)`)\n")
    readme_lines.append(
        "2. UMA **H-only** relaxation under PBC (heavy atoms fixed; cell fixed; FIRE; fmax=0.05 eV/Å; max steps=25)\n"
    )
    if args.full_relaxation_steps > 0:
        readme_lines.append(
            f"3. UMA **full** relaxation (all atoms + lattice) starting from the H-only relaxed structure (FIRE + FrechetCellFilter; fmax=0.05 eV/Å; steps={int(args.full_relaxation_steps)})\n"
        )
    else:
        readme_lines.append("3. (optional) UMA full relaxation is disabled for this run (full_relaxation_steps=0).\n")

    readme_lines.append("\n## Files\n")
    readme_lines.append(
        "- `all_seeds_data.json`: original evaluation output (non-UMA metrics are computed **pre-hydrogenation**; structure snapshots include GT + pred pre-H).\n"
    )
    readme_lines.append(
        "- `all_seeds_data_hrelaxed_uma.json`: same overall structure as `all_seeds_data.json`, but augmented with **H-relaxed UMA scalar metrics** and pointers into `.pt` files (no large arrays inline).\n"
    )
    readme_lines.append(
        "- `hrelaxed_uma_features.pt`: torch-saved data for **GT and pred** after hydrogenation + H-only UMA relaxation.\n"
    )
    readme_lines.append(
        "  - Default: this is a single torch-saved **list of dicts** (one row per structure).\n"
    )
    readme_lines.append(
        "  - If `--chunk` was used: this is a small torch-saved **index dict** pointing to shard files under `hrelaxed_uma_features.pt.chunks/part_*.pt`.\n"
    )
    if args.visualize:
        readme_lines.append(
            "- `visualizations/*_hadded_pre_relax.png`: additional comparison visualizations **after hydrogenation but before UMA H-only relaxation**, using `edge_index_h` and all atoms including H.\n"
        )
    if args.full_relaxation_steps > 0:
        readme_lines.append(
            "- `full_relax_metrics.pt`: torch-saved data for UMA **full relaxation** metrics (stress/pressure/trajectory/etc) starting from the H-only-relaxed structures.\n"
        )
        readme_lines.append(
            "  - If `--chunk` was used: this is a small torch-saved **index dict** pointing to shard files under `full_relax_metrics.pt.chunks/`.\n"
        )

    readme_lines.append("\n## ID conventions\n")
    readme_lines.append("Each hydrogenated + H-only-relaxed structure written to the `.pt` has an `id`:\n")
    readme_lines.append("- `REFCODE_gt_hrelax` (GT for that refcode)\n")
    readme_lines.append("- `REFCODE_seed_<seed_idx>_pred_hrelax` (prediction for that seed)\n")

    readme_lines.append("\n## `all_seeds_data_hrelaxed_uma.json` schema additions\n")
    readme_lines.append(
        "This file is a JSON list of crystals (same top-level list structure as `all_seeds_data.json`). We add:\n"
    )
    readme_lines.append("\n### Crystal-level additions\n")
    readme_lines.append("Under `best_metrics.hrelaxed_uma` (dict):\n")
    readme_lines.append("- `gt_hrelaxed_id`: string id (see conventions above)\n")
    readme_lines.append("- `gt_hrelaxed_pt_index`: index into `hrelaxed_uma_features.pt`\n")
    readme_lines.append("- `gt_hrelaxed_energy`: UMA energy (eV) after H-only relaxation\n")
    readme_lines.append("- `gt_hrelaxed_mean_force_norm`: mean per-atom ||F|| (eV/Å) after H-only relaxation\n")
    readme_lines.append("- `gt_h_relax_converged`: bool\n")
    readme_lines.append("- `gt_h_relax_steps`: int (# recorded optimizer steps)\n")
    readme_lines.append("- `Z_num_molecules`: copied from `best_metrics.num_molecules` if present\n")
    readme_lines.append("- `hrelaxed_features_pt_path`: basename of `hrelaxed_uma_features.pt`\n")
    readme_lines.append("- `full_relaxation_steps`: int (0 disables)\n")
    readme_lines.append("- `full_relax_metrics_pt_path`: basename of `full_relax_metrics.pt` (or null)\n")
    if args.full_relaxation_steps > 0:
        readme_lines.append("- `gt_full_relax_pt_index`: index into `full_relax_metrics.pt` (GT id)\n")

    readme_lines.append("\n### Seed-level additions\n")
    readme_lines.append(
        "For each seed entry in `all_seeds_data[*].all_seeds_data[*]`, we add `hrelaxed_uma` (dict) when available:\n"
    )
    readme_lines.append("- `pred_hrelaxed_id`: string id\n")
    readme_lines.append("- `pred_hrelaxed_pt_index`: index into `hrelaxed_uma_features.pt`\n")
    readme_lines.append("- `h_relax_converged`: bool\n")
    readme_lines.append("- `h_relax_steps`: int\n")
    readme_lines.append("- `pred_hrelaxed_energy`: UMA energy (eV) after H-only relaxation\n")
    readme_lines.append("- `pred_hrelaxed_mean_force_norm`: mean per-atom ||F|| (eV/Å) after H-only relaxation\n")
    readme_lines.append("- `pred_hrelaxed_max_force_norm`: max per-atom ||F|| (eV/Å) after H-only relaxation\n")
    readme_lines.append("- `Z_num_molecules`: copied from `best_metrics.num_molecules` if present\n")
    if args.full_relaxation_steps > 0:
        readme_lines.append("- `pred_full_relax_pt_index`: index into `full_relax_metrics.pt` (pred id)\n")
        readme_lines.append("\n### `full_relax_metrics.pt` row schema (per id)\n")
        readme_lines.append("Each row is a dict containing `id` plus UMA outputs from `scripts/calculate_uma_metrics.py`.\n")
        readme_lines.append("Additional fields added by this postprocess step:\n")
        readme_lines.append("- `atom_types`: atomic numbers (includes H) copied from the H-only-relaxed `.pt` row (assumes atom ordering preserved)\n")
        readme_lines.append("- `edge_index_h`: bond edge_index for the hydrogenated structure (for visualization and connectivity tests)\n")
        readme_lines.append("- `connectivity_changed`: bool or null; FastCSP-style JmolNN covalent adjacency differs between H-only-relaxed and full-relaxed\n")
        readme_lines.append("\nKey UMA-provided fields (subset):\n")
        readme_lines.append("- `relaxation_converged`: bool\n")
        readme_lines.append("- `relaxation_trajectory`: list[float] (energy per optimizer step; full trajectory)\n")
        readme_lines.append("- `relaxed_cart_coords`, `relaxed_lattice_matrix`: relaxed geometry\n")
        readme_lines.append("- `unrelaxed_cart_coords`, `unrelaxed_lattice_matrix`: starting geometry read from the input CIF\n")

    readme_lines.append("\n### Folder-level additions\n")
    readme_lines.append(
        "At the crystal dict top-level we also add `hrelaxed_uma_meta` containing basenames of the `.pt` files:\n"
    )
    readme_lines.append("- `hrelaxed_features_pt_path`\n")
    if args.full_relaxation_steps > 0:
        readme_lines.append("- `full_relax_metrics_pt_path`\n")

    readme_lines.append("\n## `hrelaxed_uma_features.pt` schema\n")
    readme_lines.append(
        "This is a `torch.save(list_of_dicts)` file. Each dict corresponds to one structure (GT or pred) after H-only relaxation and contains:\n"
    )
    readme_lines.append("- `id`, `refcode`, `seed_idx`, `kind` (`gt`/`pred`)\n")
    readme_lines.append("- `num_heavy`: number of heavy atoms (H-only relaxation keeps these fixed)\n")
    readme_lines.append("- `num_atoms`: total atoms including hydrogens\n")
    readme_lines.append("- `atom_types`: atomic numbers (includes H)\n")
    readme_lines.append(
        "- `edge_index_h`: bonded graph edge_index for the hydrogenated structure (includes H bonds), shape [2, E]\n"
    )
    readme_lines.append(
        "- `coords_h_pre_hrelax`: Cartesian coords (Å) right after hydrogenation (before UMA H-only relaxation)\n"
    )
    readme_lines.append("- `positions`: Cartesian coords (Å) after UMA H-only relaxation\n")
    readme_lines.append("- `cell`: 3x3 lattice matrix, PBC on\n")
    readme_lines.append("- `lattice_params`: [a,b,c,alpha,beta,gamma]\n")
    readme_lines.append("- `h_relax_converged`, `h_relax_steps`, `h_relax_trajectory`\n")
    readme_lines.append("- UMA outputs on final H-only-relaxed structure:\n")
    readme_lines.append(
        "  - `energy` (eV), `forces` (N×3), `per_atom_force_norms`, `mean_force_norm`, `max_force_norm`\n"
    )
    readme_lines.append("  - `per_atom_energies` (N)\n")
    readme_lines.append("  - `node_embeddings` (N×sph_feature_size×C) and `node_embeddings_scalar` (N×C)\n")

    if args.full_relaxation_steps > 0:
        readme_lines.append("\n## `full_relax_metrics.pt` schema\n")
        readme_lines.append(
            "This is a `torch.save(list_of_dicts)` file. Each dict has `id` plus the JSON payload returned by `scripts/calculate_uma_metrics.py`, including:\n"
        )
        readme_lines.append("- `unrelaxed_energy`, `unrelaxed_forces`, `force_norms`, stress/pressure, coords, lattice\n")
        readme_lines.append(
            "- `relaxed_energy`, `relaxed_forces`, `relaxed_force_norms`, stress/pressure, coords, lattice\n"
        )
        readme_lines.append("- `relaxation_trajectory`, `relaxation_converged`\n")
        readme_lines.append(
            "- `z_value` may be null (we do not pass an isolated-molecule file in this postprocess)\n"
        )

    with open(out_readme, "w") as f:
        f.write("".join(readme_lines))
    print(f"Wrote: {out_readme}")


if __name__ == "__main__":
    main()

