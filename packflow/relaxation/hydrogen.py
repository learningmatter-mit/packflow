#!/usr/bin/env python3
"""
Extract ground truth structures WITH hydrogens from CCDC for CSP blind test refcodes.

Requires the CSD Python API. It queries CCDC for the CSP blind-test
refcodes and saves the complete
structures (including hydrogen positions from the database) to a pickle file.

The pickle file can later be loaded by the relaxation script (which uses torch).

Usage:
    python extract_ccdc_gt_hydrogens.py

Output:
    csd_blind_test_ground_truths_with_hydrogens/ccdc_gt_structures.pkl
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# CSD Python API imports
try:
    from ccdc import io
    from ccdc.crystal import Crystal
    from ccdc.molecule import Molecule
except ImportError:
    print("ERROR: CSD Python API not available.")
    print("Install the CCDC CSD Python API in this environment and retry.")
    sys.exit(1)

# Explicit CSD database path (workaround for database path resolution issues)
CSD_DB_PATH = os.environ.get('CSD_DATABASE_PATH', 'CSD')  # set to your local CSD .sqlite


def get_csd_reader():
    """
    Get CSD EntryReader with explicit database path.
    This function provides a workaround for CSD database path resolution issues.
    """
    return io.EntryReader(CSD_DB_PATH)


# CSP Blind Test refcodes (Figure 5 case studies). Same list as
# ``data/refcodes/blind_test.txt``.
from packflow.config import load_refcodes as _load_refcodes
CSP_BLIND_TEST_REFCODES = _load_refcodes("blind_test.txt") or [
    "XAFPAY01",
    "OBEQOD",
]

# Atomic number lookup
ELEMENT_TO_Z = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
    'Si': 14, 'B': 5, 'Se': 34, 'Te': 52, 'As': 33, 'Sb': 51, 'Bi': 83,
}


def get_atomic_number(element_symbol: str) -> int:
    """Convert element symbol to atomic number."""
    # Handle potential variations in capitalization
    sym = element_symbol.strip().capitalize()
    if sym in ELEMENT_TO_Z:
        return ELEMENT_TO_Z[sym]
    # Try uppercase for two-letter symbols like Cl, Br
    sym_upper = element_symbol.strip()
    if len(sym_upper) == 2:
        sym_upper = sym_upper[0].upper() + sym_upper[1].lower()
        if sym_upper in ELEMENT_TO_Z:
            return ELEMENT_TO_Z[sym_upper]
    raise ValueError(f"Unknown element symbol: {element_symbol}")


def extract_structure_from_ccdc(refcode: str) -> Optional[Dict[str, Any]]:
    """
    Extract crystal structure with hydrogens from CCDC.
    
    Args:
        refcode: CSD refcode (e.g., 'XAFPAY')
    
    Returns:
        Dict with structure data, or None if extraction fails
    """
    try:
        # Read entry from CSD using explicit database path
        csd_reader = get_csd_reader()
        entry = csd_reader.entry(refcode)
        
        if entry is None:
            print(f"  ERROR: Entry {refcode} not found in CSD")
            return None
        
        crystal = entry.crystal
        if crystal is None:
            print(f"  ERROR: No crystal data for {refcode}")
            return None
        
        molecule = entry.molecule
        
        # Get cell parameters
        cell = crystal.cell_lengths + crystal.cell_angles  # (a, b, c, alpha, beta, gamma)
        
        # Convert to lattice matrix (3x3)
        # Using the standard crystallographic convention
        a, b, c = cell[0], cell[1], cell[2]
        alpha, beta, gamma = np.radians(cell[3]), np.radians(cell[4]), np.radians(cell[5])
        
        # Compute lattice vectors
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        # Volume factor
        val = 1.0 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2.0*cos_alpha*cos_beta*cos_gamma
        if val < 0:
            val = 0.0
        vol_factor = np.sqrt(val)
        
        # Lattice matrix (row vectors)
        lattice_matrix = np.array([
            [a, 0.0, 0.0],
            [b * cos_gamma, b * sin_gamma, 0.0],
            [c * cos_beta, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma, c * vol_factor / sin_gamma]
        ])
        
        # Get atoms from the crystal (asymmetric unit expanded to full cell)
        # crystal.molecule gives the asymmetric unit, we need the full crystal
        
        # Check if structure has hydrogens
        atom_symbols = [atom.atomic_symbol for atom in crystal.molecule.atoms]
        has_hydrogens = 'H' in atom_symbols
        
        # Count expected hydrogens based on molecule composition
        molecule_atoms = molecule.atoms if molecule else crystal.molecule.atoms
        molecule_has_h = 'H' in [a.atomic_symbol for a in molecule_atoms]
        
        print(f"  {refcode}: has_hydrogens={has_hydrogens}, molecule_has_h={molecule_has_h}")
        
        # If molecule should have H but crystal doesn't, try to add them
        crystal_mol = crystal.molecule
        if molecule_has_h and not has_hydrogens:
            print(f"    Adding hydrogens to {refcode}...")
            crystal_mol.add_hydrogens()
            has_hydrogens = True
        
        # Extract atom data
        atom_types = []
        frac_coords = []
        cart_coords = []
        
        for atom in crystal_mol.atoms:
            try:
                z = get_atomic_number(atom.atomic_symbol)
            except ValueError as e:
                print(f"    WARNING: {e}, skipping atom")
                continue
            
            atom_types.append(z)
            
            # Get fractional coordinates
            frac = atom.fractional_coordinates
            if frac is None:
                # Try to compute from Cartesian if available
                cart = atom.coordinates
                if cart is None:
                    print(f"    WARNING: No coordinates for atom {atom.label}")
                    continue
                # Convert Cartesian to fractional
                cart_np = np.array([cart.x, cart.y, cart.z])
                inv_lattice = np.linalg.inv(lattice_matrix)
                frac_np = cart_np @ inv_lattice
                frac_coords.append(frac_np.tolist())
                cart_coords.append(cart_np.tolist())
            else:
                frac_coords.append([frac.x, frac.y, frac.z])
                # Compute Cartesian from fractional
                frac_np = np.array([frac.x, frac.y, frac.z])
                cart_np = frac_np @ lattice_matrix
                cart_coords.append(cart_np.tolist())
        
        # Get Z value (number of molecules in unit cell)
        z_value = crystal.z_value if hasattr(crystal, 'z_value') and crystal.z_value else None
        if z_value is None:
            # Try to compute from asymmetric unit
            z_prime = crystal.z_prime if hasattr(crystal, 'z_prime') and crystal.z_prime else 1
            # Z' is molecules per asymmetric unit
            # For most cases, Z = Z' * (number of symmetry operations)
            # This is approximate; exact value depends on space group
            z_value = None  # Will be filled from evaluation data later
        
        result = {
            "refcode": refcode,
            "atom_types": np.array(atom_types, dtype=np.int64),
            "frac_coords": np.array(frac_coords, dtype=np.float64),
            "cart_coords": np.array(cart_coords, dtype=np.float64),
            "lattice_matrix": lattice_matrix.astype(np.float64),
            "cell_lengths": np.array([a, b, c], dtype=np.float64),
            "cell_angles": np.array([np.degrees(alpha), np.degrees(beta), np.degrees(gamma)], dtype=np.float64),
            "has_experimental_hydrogens": has_hydrogens and not molecule_has_h or has_hydrogens,  # True if H from CCDC
            "num_atoms": len(atom_types),
            "z_value": z_value,
        }
        
        print(f"    Extracted {len(atom_types)} atoms (including {sum(1 for z in atom_types if z == 1)} H)")
        
        return result
        
    except Exception as e:
        print(f"  ERROR extracting {refcode}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    output_dir = Path(os.environ.get("PACKFLOW_GT_H_DIR", str(Path(__file__).resolve().parents[2] / "data" / "csd_blind_test_ground_truths_with_hydrogens")))
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "ccdc_gt_structures.pkl"
    
    print("=" * 60)
    print("Extracting CCDC ground truth structures with hydrogens")
    print("=" * 60)
    print(f"Refcodes: {CSP_BLIND_TEST_REFCODES}")
    print(f"Output: {output_file}")
    print()
    
    results = {}
    success_count = 0
    
    for refcode in CSP_BLIND_TEST_REFCODES:
        print(f"Processing {refcode}...")
        data = extract_structure_from_ccdc(refcode)
        if data is not None:
            results[refcode] = data
            success_count += 1
        print()
    
    # Save results using pickle (available in ccdc_new env)
    print("=" * 60)
    print(f"Successfully extracted {success_count}/{len(CSP_BLIND_TEST_REFCODES)} structures")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print("Done!")
    print()
    
    # Print summary
    print("Summary:")
    for refcode, data in results.items():
        n_atoms = data["num_atoms"]
        n_h = sum(1 for z in data["atom_types"] if z == 1)
        print(f"  {refcode}: {n_atoms} atoms ({n_h} H), lattice: {data['cell_lengths']}")


if __name__ == "__main__":
    main()
