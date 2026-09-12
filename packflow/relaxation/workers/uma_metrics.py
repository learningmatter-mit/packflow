#!/usr/bin/env python3
"""
Script to calculate UMA relaxation metrics (steps and lattice energy).
Designed to be run in the `fairchem` environment.

Usage:
    python calculate_uma_metrics.py --batch_manifest <path_to_json> --device <cpu|cuda> [--skip_relaxation]
    OR
    python calculate_uma_metrics.py --crystal <path> --molecule <path> ...

Output (JSON to stdout):
    {
        "id1": {metrics...},
        "id2": {metrics...}
    }
"""

import sys
import os
import argparse
import json
import torch
import numpy as np
from ase.io import read
from ase.optimize import FIRE
try:
    from ase.filters import FrechetCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter as FrechetCellFilter
import traceback


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

# AMD imports for descriptor computation
try:
    from amd import PeriodicSet, AMD
    HAS_AMD = True
except ImportError:
    HAS_AMD = False
    print("Warning: 'amd' not found. AMD descriptors will not be computed.", file=sys.stderr)


def compute_amd_descriptor(positions, cell, atomic_numbers, k_neighbors=100):
    """
    Compute AMD descriptor for a crystal structure.
    
    Args:
        positions: [N, 3] Cartesian positions
        cell: [3, 3] lattice matrix
        atomic_numbers: [N] atomic numbers
        k_neighbors: number of neighbors for AMD
    
    Returns:
        AMD descriptor array or None if computation fails
    """
    if not HAS_AMD:
        return None
    
    try:
        ps = PeriodicSet(positions, cell, atomic_numbers)
        amd_vec = AMD(ps, k_neighbors)
        return amd_vec.tolist()  # Convert to list for JSON serialization
    except Exception as e:
        print(f"   WARNING: AMD descriptor computation failed: {e}", file=sys.stderr)
        return None


def compute_amd_linf(amd1, amd2):
    """Compute L_inf distance between two AMD descriptors."""
    if amd1 is None or amd2 is None:
        return None
    try:
        amd1_arr = np.array(amd1)
        amd2_arr = np.array(amd2)
        return float(np.max(np.abs(amd1_arr - amd2_arr)))
    except Exception as e:
        print(f"   WARNING: AMD L_inf computation failed: {e}", file=sys.stderr)
        return None


def calculate_metrics_for_entry(predictor, task_name, entry, device='cuda'):
    crystal_path = entry['crystal_path']
    molecule_path = entry.get('molecule_path')
    # output_relaxed_crystal_path not needed anymore
    uma_relaxation_steps = entry.get('relaxation_steps', 0)
    
    metrics = {
        "relaxation_steps": uma_relaxation_steps,
        "lattice_energy": None,
        "crystal_energy_total": None,
        "molecule_energy": None,
        "z_value": None,
        "error": None,
        # Relaxation metrics
        "relaxed_energy": None, 
        "relaxed_max_force": None,
        "relaxed_mean_force": None,
        "relaxation_trajectory": [],
        "relaxed_lattice_matrix": None,
        "relaxed_frac_coords": None,
        "relaxed_cart_coords": None,  # Cartesian coords after relaxation
        "relaxed_force_norms": None,  # Per-atom force norms after relaxation
        "relaxed_amd_descriptor": None,  # AMD descriptor of relaxed crystal
        "relaxed_stress_voigt": None,  # Stress tensor in Voigt notation (6 components) after relaxation
        "relaxed_stress_matrix": None,  # Stress tensor as 3x3 matrix after relaxation
        "relaxed_pressure": None,  # Hydrostatic pressure after relaxation (eV/Å³)
        "relaxation_converged": None,  # True if relaxation converged (max force < fmax), False if hit step limit
        "amd_linf_unrelaxed_vs_relaxed": None,  # AMD L_inf between unrelaxed and relaxed
        # Unrelaxed metrics
        "unrelaxed_energy": None,
        "unrelaxed_max_force": None,
        "unrelaxed_mean_force": None,
        "unrelaxed_lattice_matrix": None,  # Lattice matrix (unrelaxed)
        "unrelaxed_frac_coords": None,  # Fractional coords (unrelaxed)
        "unrelaxed_cart_coords": None,  # Cartesian coords (unrelaxed)
        "unrelaxed_stress_voigt": None,  # Stress tensor in Voigt notation (6 components)
        "unrelaxed_stress_matrix": None,  # Stress tensor as 3x3 matrix
        "unrelaxed_pressure": None,  # Hydrostatic pressure (eV/Å³)
        "force_norms": None,  # Per-atom force norms (unrelaxed)
        "amd_descriptor": None,  # AMD descriptor of unrelaxed crystal
    }

    try:
        # 1. Load Structures
        crystal = read(crystal_path)
        molecule = None
        if molecule_path:
            molecule = read(molecule_path)
        
        # Calculate Z (molecules per unit cell) if molecule provided
        if molecule is not None:
            n_atoms_crystal = len(crystal)
            n_atoms_molecule = len(molecule)
            if n_atoms_molecule == 0:
                raise ValueError("Molecule file is empty")
            z_value = n_atoms_crystal / n_atoms_molecule
            metrics["z_value"] = z_value
        else:
            metrics["z_value"] = None

        # 2. Attach Calculator
        calc = FAIRChemCalculator(predictor, task_name=task_name)
        crystal.calc = calc

        # 3. Unrelaxed Metrics (Always Compute)
        e_unrelaxed = crystal.get_potential_energy()
        forces_unrelaxed = crystal.get_forces()
        force_norms_unrelaxed = np.linalg.norm(forces_unrelaxed, axis=1)
        max_force = force_norms_unrelaxed.max()
        
        # Check for CUDA errors after unrelaxed computation
        if not cuda_sync_and_check():
            raise RuntimeError("CUDA error detected after unrelaxed energy/force computation")
        
        # Get stress tensor (Voigt notation: xx, yy, zz, yz, xz, xy) in eV/Å³
        stress_voigt_unrelaxed = crystal.get_stress(voigt=True)
        stress_matrix_unrelaxed = crystal.get_stress(voigt=False)  # 3x3 matrix
        # Hydrostatic pressure: P = -trace(stress)/3
        pressure_unrelaxed = -np.trace(stress_matrix_unrelaxed) / 3.0
        
        # Store unrelaxed structural data
        unrelaxed_positions = crystal.get_positions()
        unrelaxed_cell = crystal.cell.array
        atomic_nums = crystal.get_atomic_numbers()
        
        metrics["unrelaxed_energy"] = e_unrelaxed
        metrics["unrelaxed_max_force"] = float(max_force)
        metrics["unrelaxed_mean_force"] = float(np.mean(force_norms_unrelaxed))
        metrics["force_norms"] = force_norms_unrelaxed.tolist()  # Per-atom force norms
        metrics["unrelaxed_forces"] = forces_unrelaxed.tolist()  # Full force vectors [N, 3]
        metrics["unrelaxed_lattice_matrix"] = unrelaxed_cell.tolist()
        metrics["unrelaxed_frac_coords"] = crystal.get_scaled_positions().tolist()
        metrics["unrelaxed_cart_coords"] = unrelaxed_positions.tolist()
        metrics["unrelaxed_stress_voigt"] = stress_voigt_unrelaxed.tolist()
        metrics["unrelaxed_stress_matrix"] = stress_matrix_unrelaxed.tolist()
        metrics["unrelaxed_pressure"] = float(pressure_unrelaxed)
        
        # Compute AMD descriptor for unrelaxed structure
        amd_unrelaxed = compute_amd_descriptor(unrelaxed_positions, unrelaxed_cell, atomic_nums)
        metrics["amd_descriptor"] = amd_unrelaxed

        print(f"[DEBUG] UMA Relaxation Steps received: {uma_relaxation_steps}", file=sys.stderr)
        
        # 4. Relaxation (Optional)
        if uma_relaxation_steps > 0:
            # Setup Optimizer
            # Use FrechetCellFilter to relax both atomic positions and lattice
            ecf = FrechetCellFilter(crystal)
            opt = FIRE(ecf)
            
            trajectory = []
            def log_step():
                # We need to compute potential energy. 
                # Note: opt.atoms is the Filter object, opt.atoms.atoms is the Crystal
                e = crystal.get_potential_energy()
                # Cast to standard float for JSON serialization
                trajectory.append(float(e))
                
            # Log every step
            opt.attach(log_step, interval=1)
            
            # Run relaxation - returns True if converged, False if hit step limit
            converged = opt.run(fmax=0.05, steps=uma_relaxation_steps)
            metrics["relaxation_converged"] = bool(converged)
            
            # Check for CUDA errors after relaxation
            if not cuda_sync_and_check():
                raise RuntimeError("CUDA error detected after relaxation")
            
            print(f"[DEBUG] Trajectory length: {len(trajectory)}, converged: {converged}", file=sys.stderr)
            
            # Collect Final Metrics
            e_relaxed = crystal.get_potential_energy()
            forces_relaxed = crystal.get_forces()
            force_norms_relaxed = np.linalg.norm(forces_relaxed, axis=1)
            max_force_relaxed = force_norms_relaxed.max()
            
            # Get stress tensor for relaxed structure
            stress_voigt_relaxed = crystal.get_stress(voigt=True)
            stress_matrix_relaxed = crystal.get_stress(voigt=False)  # 3x3 matrix
            # Hydrostatic pressure: P = -trace(stress)/3
            pressure_relaxed = -np.trace(stress_matrix_relaxed) / 3.0
            
            # Store relaxed structural data
            relaxed_positions = crystal.get_positions()
            relaxed_cell = crystal.cell.array
            
            metrics["relaxed_energy"] = e_relaxed
            metrics["relaxed_max_force"] = float(max_force_relaxed)
            metrics["relaxed_mean_force"] = float(np.mean(force_norms_relaxed))
            metrics["relaxation_trajectory"] = trajectory
            metrics["relaxed_lattice_matrix"] = relaxed_cell.tolist()
            metrics["relaxed_frac_coords"] = crystal.get_scaled_positions().tolist()
            metrics["relaxed_cart_coords"] = relaxed_positions.tolist()
            metrics["relaxed_force_norms"] = force_norms_relaxed.tolist()  # Per-atom force norms after relaxation
            metrics["relaxed_forces"] = forces_relaxed.tolist()  # Full force vectors after relaxation [N, 3]
            metrics["relaxed_stress_voigt"] = stress_voigt_relaxed.tolist()
            metrics["relaxed_stress_matrix"] = stress_matrix_relaxed.tolist()
            metrics["relaxed_pressure"] = float(pressure_relaxed)
            
            # Compute AMD descriptor for relaxed structure
            amd_relaxed = compute_amd_descriptor(relaxed_positions, relaxed_cell, atomic_nums)
            metrics["relaxed_amd_descriptor"] = amd_relaxed
            
            # Compute AMD L_inf between unrelaxed and relaxed
            amd_linf = compute_amd_linf(amd_unrelaxed, amd_relaxed)
            metrics["amd_linf_unrelaxed_vs_relaxed"] = amd_linf
            
            # Final CUDA check after all relaxed metrics
            if not cuda_sync_and_check():
                raise RuntimeError("CUDA error detected after collecting relaxed metrics")

    except Exception as e:
        metrics["error"] = str(e)
        # print(f"Error in UMA script ({crystal_path}): {e}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr)

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_manifest", help="Path to JSON file with list of entries")
    parser.add_argument("--crystal", help="Path to single crystal CIF")
    parser.add_argument("--molecule", help="Path to single molecule XYZ (optional)")
    parser.add_argument("--relaxation_steps", type=int, default=0, help="Number of relaxation steps (0 to disable)")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # Check inputs
    entries = []
    if args.batch_manifest:
        with open(args.batch_manifest, 'r') as f:
            entries = json.load(f)
    elif args.crystal:
        entry = {
            "id": "single_entry",
            "crystal_path": args.crystal,
            "relaxation_steps": args.relaxation_steps
        }
        if args.molecule:
            entry["molecule_path"] = args.molecule
        entries = [entry]
    else:
        print("Error: Must provide either --batch_manifest or --crystal", file=sys.stderr)
        sys.exit(1)
        
    device = args.device
    
    # Initialize Predictor ONCE
    try:
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
    except Exception as e:
         print(f"Error loading model: {e}", file=sys.stderr)
         sys.exit(1)
         
    results = {}
    
    total_entries = len(entries)
    print(f"Processing {total_entries} entries on {device}...", file=sys.stderr)
    
    for i, entry in enumerate(entries):
        if (i+1) % 10 == 0:
            print(f"Processing {i+1}/{total_entries}...", file=sys.stderr)
            
        entry_id = entry.get("id", "unknown")
        # Ensure relaxation steps is propagated from args if not in entry (for batch mode overrides if we ever added that)
        # But for now, we assume batch manifest has it OR we don't support batch-wide override effectively
        # Actually let's just use what's in entry, but for command line single mode we added it.
        # If running via batch manifest, we might want to inject it if missing?
        # Let's rely on the caller to put it in the manifest or the single entry dict.
        # UPDATE: evaluate_test_set_metrics writes the manifest/calls this.
        # Since I'm not changing the manifest format in evaluate yet, I should probably handle the case where
        # it's passed as an arg but we are in batch mode?
        # Actually, evaluate_test_set_metrics calls this script with --crystal and --molecule mostly (single mode).
        # Wait, let's check evaluate_test_set_metrics usage.
        # It calls via subprocess: `python calculate_uma_metrics.py --crystal ... --molecule ...`
        # So it uses the single entry mode.
        
        metrics = calculate_metrics_for_entry(predictor, "omc", entry, device=device)
        results[entry_id] = metrics
        
        # If there was an error, attempt to clear CUDA state for subsequent entries
        if metrics.get("error"):
            cuda_clear_error_state()
        
        # Cleanup input files immediately to save space
        try:
            if "crystal_path" in entry and os.path.exists(entry["crystal_path"]):
                os.remove(entry["crystal_path"])
            if "molecule_path" in entry and entry.get("molecule_path") and os.path.exists(entry["molecule_path"]):
                os.remove(entry["molecule_path"])
        except OSError as e:
            print(f"Warning: Failed to delete input files for {entry_id}: {e}", file=sys.stderr)
        
    # Output all results as JSON
    print(json.dumps(results))

if __name__ == "__main__":
    main()
