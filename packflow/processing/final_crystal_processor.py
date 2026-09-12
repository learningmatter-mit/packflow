#!/usr/bin/env python3
"""
Crystal Structure Processing Module

This module provides functionality for processing crystallographic structures
from mmCIF files. It converts mmCIF data to the expected dictionary format,
applies symmetry operations, and processes crystal structures into graph features.

Refactored to:
- Remove redundant Niggli reduction (pass-through logic).
- Remove unused visualization and geometric transformation code.
- Ensure strict consistency with crystal_utils conventions.
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import gemmi
from joblib import Parallel, delayed

from pymatgen.core import Lattice

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import processing utilities from crystal utils
from packflow.utils.crystal_utils import (
    get_crystal_info, 
    get_molecule_ids,
    error_dict,
    lattice_matrix_to_k_basis
)

# Constants
DEFAULT_SMILES = 'CC(=O)C=C(C)Nc1cncnc1N' 


def parse_mmcif_to_crystal_dict(mmcif_file_path: str) -> Dict[str, Any]:
    """
    Parse an mmCIF file and convert it to the expected dictionary format.
    """
    if not os.path.exists(mmcif_file_path):
        raise FileNotFoundError(f"mmCIF file not found: {mmcif_file_path}")
    
    # Initialize data containers
    unit_cell_parameters = {}
    atom_list = []
    bond_list = []
    molecular_smiles = None
    crystal_refcode = os.path.splitext(os.path.basename(mmcif_file_path))[0]
    
    try:
        with open(mmcif_file_path, 'r', encoding='utf-8') as file:
            file_lines = file.readlines()
    except (IOError, UnicodeDecodeError) as e:
        raise ValueError(f"Error reading mmCIF file {mmcif_file_path}: {e}")
    
    atom_site_column_mapping = {}
    struct_conn_column_mapping = {}
    is_parsing_connectivity = False
    atom_label_to_index_map = {}
    
    for line in file_lines:
        line_content = line.strip()
        
        if not line_content or line_content.startswith('#'):
            continue
        
        molecular_smiles = _extract_smiles_from_line(line_content, molecular_smiles)
        unit_cell_parameters = _extract_cell_parameters(line_content, unit_cell_parameters)
        
        if line_content.startswith('_atom_site.'):
            column_name = line_content.split()[0].replace('_atom_site.', '')
            atom_site_column_mapping[column_name] = len(atom_site_column_mapping)
            is_parsing_connectivity = False
        elif line_content.startswith('_struct_conn.'):
            column_name = line_content.split()[0].replace('_struct_conn.', '')
            struct_conn_column_mapping[column_name] = len(struct_conn_column_mapping)
            is_parsing_connectivity = True
        
        elif not line_content.startswith(('_', 'loop_', 'data_')):
            if not is_parsing_connectivity:
                _parse_atom_data_line(line_content, atom_site_column_mapping, 
                                    atom_list, atom_label_to_index_map)
            else:
                _parse_bond_data_line(line_content, struct_conn_column_mapping, 
                                    bond_list, atom_label_to_index_map)
    
    if molecular_smiles is None:
        molecular_smiles = DEFAULT_SMILES
    
    if not unit_cell_parameters or len(atom_list) == 0:
        raise ValueError(f"Insufficient data parsed from mmCIF file: {mmcif_file_path}")
    
    return {
        'refcode': crystal_refcode,
        'smiles': molecular_smiles,
        'cell': unit_cell_parameters,
        'atoms': atom_list,
        'bonds': bond_list
    }


def apply_symmetry_operations(crystal_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply symmetry operations to expand the asymmetric unit to the full unit cell using Gemmi.
    """
    try:
        refcode = crystal_dict['refcode']
        
        # Search for the original mmCIF file based on refcode
        original_mmcif_path = None
        search_dirs = ["data/test_cifs", "data/train", "data/val", "data/test"]
        for search_dir in search_dirs:
            p = os.path.join(search_dir, f"{refcode}.mmcif")
            if os.path.exists(p):
                original_mmcif_path = p
                break
        
        if not original_mmcif_path:
            # If we can't find the file to read symmetry ops, return the dict as is
            return crystal_dict

        doc = gemmi.cif.read(original_mmcif_path)
        block = doc.sole_block()
        
        # Extract symmetry operations from file
        symmetry_ops = []
        try:
            with open(original_mmcif_path, 'r') as f:
                lines = f.readlines()
            in_oper = False
            for line in lines:
                line = line.strip()
                if line.startswith('_pdbx_struct_oper_list.id'): 
                    in_oper = True
                    continue
                elif line.startswith('_') and not line.startswith('_pdbx_struct_oper_list.'): 
                    in_oper = False
                    continue
                
                if in_oper and line and not line.startswith('_'):
                    parts = line.split()
                    if len(parts) >= 2:
                        symmetry_ops.append((parts[0], ' '.join(parts[1:])))
        except:
            return error_dict(crystal_dict, "Failed parsing symmetry ops")

        original_atoms = crystal_dict['atoms']
        original_count = len(original_atoms)
        original_smiles = crystal_dict['smiles']
        expanded_smiles_parts = [original_smiles]
        
        all_atoms = []
        all_bonds = []
        
        for op_id, op_sym in symmetry_ops:
            try:
                transformed = _apply_symmetry_operation(original_atoms, op_sym)
                # Wrap coords
                for atom in transformed:
                    atom['fract_x'] = (atom['fract_x'] % 1.0)
                    atom['fract_y'] = (atom['fract_y'] % 1.0)
                    atom['fract_z'] = (atom['fract_z'] % 1.0)
                
                all_atoms.extend(transformed)
                
                if not _is_identity_operation(op_sym):
                    expanded_smiles_parts.append(original_smiles)
            except:
                continue

        # Replicate bonds
        original_bonds = crystal_dict['bonds']
        for op_idx, _ in enumerate(symmetry_ops):
            if len(all_atoms) >= original_count * (op_idx + 1):
                offset = op_idx * original_count
                for bond in original_bonds:
                    all_bonds.append({
                        'atom1_idx': bond['atom1_idx'] + offset,
                        'atom2_idx': bond['atom2_idx'] + offset
                    })

        crystal_dict['atoms'] = all_atoms
        crystal_dict['bonds'] = all_bonds
        crystal_dict['smiles'] = ".".join(expanded_smiles_parts)
        
        return crystal_dict
        
    except Exception as e:
        return error_dict(crystal_dict, str(e))


def _is_identity_operation(operation: str) -> bool:
    try:
        x, y, z = _parse_symmetry_expression(operation)
        return (x.strip() == "x" and y.strip() == "y" and z.strip() == "z")
    except:
        return False


def _apply_symmetry_operation(atoms: List[Dict], symmetry_operation: str) -> List[Dict]:
    x_expr, y_expr, z_expr = _parse_symmetry_expression(symmetry_operation)
    transformed_atoms = []
    for atom in atoms:
        x, y, z = atom['fract_x'], atom['fract_y'], atom['fract_z']
        new_atom = {
            'element': atom['element'],
            'fract_x': _evaluate_coordinate_expression(x_expr, x, y, z),
            'fract_y': _evaluate_coordinate_expression(y_expr, x, y, z),
            'fract_z': _evaluate_coordinate_expression(z_expr, x, y, z)
        }
        transformed_atoms.append(new_atom)
    return transformed_atoms


def _parse_symmetry_expression(operation: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in operation.split(',')]
    if len(parts) != 3: 
        raise ValueError("Invalid symmetry operation")
    return parts[0], parts[1], parts[2]


def _evaluate_coordinate_expression(expr: str, x: float, y: float, z: float) -> float:
    expr = expr.strip()
    if expr == 'x': return x
    if expr == 'y': return y
    if expr == 'z': return z
    if expr == '-x': return -x
    if expr == '-y': return -y
    if expr == '-z': return -z
    
    result = 0.0
    terms = []
    current_term = ""
    sign = 1
    i = 0
    while i < len(expr):
        char = expr[i]
        if char == '+':
            if current_term: 
                terms.append((sign, current_term.strip()))
                current_term = ""
            sign = 1
        elif char == '-':
            if current_term: 
                terms.append((sign, current_term.strip()))
                current_term = ""
            sign = -1
        else:
            current_term += char
        i += 1
    if current_term: 
        terms.append((sign, current_term.strip()))
    
    for s, t in terms:
        if not t: 
            continue
        result += s * _evaluate_single_term(t, x, y, z)
    return result


def _evaluate_single_term(term: str, x: float, y: float, z: float) -> float:
    term = term.strip()
    if term == 'x': return x
    if term == 'y': return y
    if term == 'z': return z
    if term == '-x': return -x
    if term == '-y': return -y
    if term == '-z': return -z
    if '/' in term:
        try:
            if '+' in term:
                p = term.split('+')
                return _evaluate_fraction(p[0]) + _evaluate_coordinate_expression(p[1], x, y, z)
            if '-' in term and not term.startswith('-'):
                p = term.split('-')
                return _evaluate_fraction(p[0]) - _evaluate_coordinate_expression(p[1], x, y, z)
            return _evaluate_fraction(term)
        except: 
            pass
    try: 
        return float(term)
    except: 
        return 0.0


def _evaluate_fraction(frac_str: str) -> float:
    try:
        if '/' in frac_str:
            n, d = frac_str.split('/')
            return float(n) / float(d)
        return float(frac_str)
    except: 
        return 0.0


def _extract_smiles_from_line(line: str, current: Optional[str]) -> Optional[str]:
    if '_chemical_identifier_smiles' in line and current is None:
        parts = line.split("'")
        if len(parts) >= 2: 
            return parts[1]
    return current


def _extract_cell_parameters(line: str, params: Dict[str, float]) -> Dict[str, float]:
    mappings = {
        '_cell.length_a': 'a', 
        '_cell.length_b': 'b', 
        '_cell.length_c': 'c',
        '_cell.angle_alpha': 'alpha', 
        '_cell.angle_beta': 'beta', 
        '_cell.angle_gamma': 'gamma'
    }
    for k, v in mappings.items():
        if line.startswith(k):
            try: 
                params[v] = float(line.split()[-1].split('(')[0])
            except: 
                continue
    return params


def _parse_atom_data_line(line: str, col_map: Dict[str, int], atom_list: List[Dict], label_map: Dict[str, int]):
    if 'type_symbol' not in col_map: 
        return
    parts = line.split()
    if len(parts) <= max(col_map.values()): 
        return
    
    req = {
        'element': col_map.get('type_symbol', -1), 
        'label': col_map.get('label_atom_id', -1),
        'fx': col_map.get('fract_x', -1), 
        'fy': col_map.get('fract_y', -1), 
        'fz': col_map.get('fract_z', -1)
    }
    
    if any(req[k] < 0 for k in ['element', 'fx', 'fy', 'fz']): 
        return
    
    try:
        idx = len(atom_list)
        atom_list.append({
            'element': parts[req['element']],
            'fract_x': float(parts[req['fx']]), 
            'fract_y': float(parts[req['fy']]), 
            'fract_z': float(parts[req['fz']])
        })
        if req['label'] >= 0: 
            label_map[parts[req['label']]] = idx
    except: 
        pass


def _parse_bond_data_line(line: str, col_map: Dict[str, int], bond_list: List[Dict], label_map: Dict[str, int]):
    if not col_map: 
        return
    parts = line.split()
    if len(parts) <= max(col_map.values()): 
        return
    
    req = {
        'type': col_map.get('conn_type_id', -1), 
        'a1': col_map.get('ptnr1_label_atom_id', -1), 
        'a2': col_map.get('ptnr2_label_atom_id', -1)
    }
    if any(v < 0 for v in req.values()): 
        return
    
    try:
        if parts[req['type']] == 'covale':
            l1, l2 = parts[req['a1']], parts[req['a2']]
            if l1 in label_map and l2 in label_map:
                bond_list.append({'atom1_idx': label_map[l1], 'atom2_idx': label_map[l2]})
    except: 
        pass


def process_single_crystal(datapoint, RemoveHs=False, conformer_timeout=60, skip_rdkit=False, tag_hydrogen_bonding=False, find_aromatic_rings=False):
    """
    Process a single crystal datapoint.
    
    Changes:
    - Removed Niggli reduction logic.
    - Directly passes transformed fractional coordinates from graph_arrays (from crystal_utils).
    """
    print(f"Processing crystal: {datapoint['refcode']}")
    
    # Build crystal from dict
    result = get_crystal_info(datapoint, RemoveHs=RemoveHs, tag_hydrogen_bonding=tag_hydrogen_bonding, find_aromatic_rings=find_aromatic_rings)
    if isinstance(result, dict) and 'error' in result:
        return result
    
    crystal, datapoint, graph_arrays = result
    
    # Get molecule ids
    molecule_ids, molecule_lists = get_molecule_ids(edge_index=graph_arrays[6], num_atoms=graph_arrays[8])
    
    print(f"Number of atoms: {graph_arrays[8]}")
    print(f"Number of molecules: {len(molecule_lists)}")
    
    atom_types_torch = torch.from_numpy(graph_arrays[2])

    # Convert Lattice Matrix to k-basis
    # crystal.lattice.matrix is a numpy array (3x3)
    k_basis = lattice_matrix_to_k_basis(crystal.lattice.matrix)

    # Extract donor and acceptor masks if available
    result_dict = {
        'refcode': datapoint['refcode'],
        'smiles': datapoint['smiles'],

        'atom_types': atom_types_torch,
        'edge_index': graph_arrays[6],
        'node_features': graph_arrays[5],
        'bond_features': graph_arrays[7],
        'molecule_ids': molecule_ids,

        # Pass lattice parameters directly (Niggli reduction removed as requested)
        'lattice_1': torch.tensor(crystal.lattice.parameters, dtype=torch.float32),
        
        # Pymatgen matrix (c || z)
        'cell_1': torch.tensor(crystal.lattice.matrix, dtype=torch.float32),

        # Invariant k-basis representation
        'k_basis': torch.from_numpy(k_basis).float(),

        # Use fractional coords directly from graph_arrays
        # These correspond to the built structure in crystal_utils
        'frac_coords': torch.from_numpy(graph_arrays[0]).float(),
    }
    
    # Add hydrogen bonding masks if available
    if tag_hydrogen_bonding and RemoveHs and len(graph_arrays) > 9:
        donor_mask = graph_arrays[9]
        acceptor_mask = graph_arrays[10]
        if donor_mask is not None and acceptor_mask is not None:
            result_dict['hbond_donor_mask'] = torch.from_numpy(donor_mask).bool()
            result_dict['hbond_acceptor_mask'] = torch.from_numpy(acceptor_mask).bool()
    
    # Add aromatic ring data if available
    if find_aromatic_rings and len(graph_arrays) > 11:
        ring_atom_pairs = graph_arrays[11]
        ring_centroids = graph_arrays[12]
        ring_molecule_ids = graph_arrays[13]
        ring_sizes = graph_arrays[14]
        num_rings = graph_arrays[15]
        
        if ring_atom_pairs is not None and num_rings > 0:
            result_dict['ring_atom_pairs'] = ring_atom_pairs if torch.is_tensor(ring_atom_pairs) else torch.from_numpy(ring_atom_pairs).long()
            result_dict['ring_centroids'] = ring_centroids if torch.is_tensor(ring_centroids) else torch.from_numpy(ring_centroids).float()
            result_dict['ring_molecule_ids'] = ring_molecule_ids if torch.is_tensor(ring_molecule_ids) else torch.from_numpy(ring_molecule_ids).long()
            result_dict['ring_sizes'] = ring_sizes if torch.is_tensor(ring_sizes) else torch.from_numpy(ring_sizes).long()
            result_dict['num_rings'] = int(num_rings)
    
    return result_dict


def process_single_mmcif_file(mmcif_file_path: str, 
                             output_directory: str = None, # Unused but kept for signature compatibility
                             skip_rdkit: bool = False) -> Dict[str, Any]:
    """
    Process a single mmCIF file through the pipeline.
    """
    file_basename = os.path.basename(mmcif_file_path)
    print(f"Processing crystal structure file: {file_basename}")
    print("-" * 50)
    
    try:
        # Stage 1: Parse mmCIF
        print("  1. Parsing mmCIF file format...")
        crystal_datapoint = parse_mmcif_to_crystal_dict(mmcif_file_path)
        
        # Stage 1.5: Apply symmetry
        print("  1.5. Applying symmetry operations...")
        crystal_datapoint = apply_symmetry_operations(crystal_datapoint)
        
        if isinstance(crystal_datapoint, dict) and 'error' in crystal_datapoint:
            return {'status': 'error', 'refcode': crystal_datapoint['refcode'], 'error': crystal_datapoint['error']}
        
        # Stage 2: Process structure
        print("  2. Processing crystal structure...")
        processed_crystal_data = process_single_crystal(
            crystal_datapoint, 
            RemoveHs=False, 
            conformer_timeout=10, 
            skip_rdkit=skip_rdkit
        )
        
        if isinstance(processed_crystal_data, dict) and 'error' in processed_crystal_data:
            return {'status': 'error', 'refcode': crystal_datapoint['refcode'], 'error': processed_crystal_data['error']}
        
        print("  ✓ Processing completed successfully!")
        
        return {
            'status': 'success', 
            'refcode': crystal_datapoint['refcode'], 
            'data': processed_crystal_data
        }
        
    except Exception as processing_error:
        error_message = str(processing_error)
        print(f"  ✗ Processing failed: {error_message}")
        return {
            'status': 'error', 
            'refcode': os.path.splitext(file_basename)[0], 
            'error': error_message
        }


def process_directory(input_dir, output_dir='visualizations', n_jobs=-1):
    """Process all mmCIF files in a directory."""
    mmcif_files = []
    if os.path.isdir(input_dir):
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith('.mmcif'):
                mmcif_files.append(os.path.join(input_dir, filename))
    else:
        print(f"Error: Directory not found: {input_dir}")
        return
    
    if not mmcif_files: 
        print(f"No mmCIF files found in {input_dir}")
        return
    
    print(f"Found {len(mmcif_files)} mmCIF files to process")
    
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_mmcif_file)(mmcif_file, output_dir) 
        for mmcif_file in mmcif_files
    )
    
    successful = sum(1 for result in results if result['status'] == 'success')
    print(f"Successful: {successful}, Failed: {len(results) - successful}")
    return results


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("Usage: python final_crystal_processor.py <mmcif_file_or_directory>")
        return

    if os.path.isfile(input_path):
        process_single_mmcif_file(input_path)
    elif os.path.isdir(input_path):
        process_directory(input_path)
    else:
        print(f"Error: Path not found: {input_path}")


if __name__ == "__main__":
    main()
