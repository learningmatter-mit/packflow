#!/usr/bin/env python3
"""
Batch Processing Utilities for Crystal Data

This module provides utilities for batch processing crystal structures
and preparing data in the format expected by CrystalDataset.

Refactored to use centralized crystal_utils for consistency in:
- Lattice matrix conventions (c || z)
- Molecule unwrapping logic
"""

import os
import torch
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union
from joblib import Parallel, delayed
from pathlib import Path

# Import standardized utilities from crystal_utils
# This ensures we use the same lattice convention (Pymatgen default) everywhere
from packflow.utils.crystal_utils import lattice_params_to_matrix, make_molecules_whole, frac_to_cart, cart_to_frac


def edge_index_to_bonds(edge_index):
    """Convert edge_index tensor to bonds list format."""
    bonds = []
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    
    # edge_index is shape (2, num_edges)
    for i in range(edge_index.shape[1]):
        atom1_idx, atom2_idx = edge_index[:, i]
        bonds.append({'atom1_idx': int(atom1_idx), 'atom2_idx': int(atom2_idx)})
    
    return bonds


def add_whole_cartesian_coords(processed_data):
    """Add whole centered cartesian coordinates to processed crystal data.

    This function:
    1. Makes molecules whole across periodic boundaries using fractional coordinates
    2. Converts the whole fractional coordinates to cartesian coordinates  
    3. Centers the cartesian coordinates by subtracting the mean position
    4. Stores the result as 'whole_cartesian_coords'

    Args:
        processed_data: Dictionary containing processed crystal data
        
    Returns:
        processed_data: Updated dictionary with whole_cartesian_coords field
    """
    try:
        # Extract necessary data
        frac_coords = processed_data['frac_coords']
        lattice_params = processed_data['lattice_1']
        edge_index = processed_data['edge_index']
        
        # Convert to numpy arrays
        if torch.is_tensor(frac_coords):
            frac_coords_np = frac_coords.cpu().numpy()
        else:
            frac_coords_np = np.array(frac_coords)
            
        if torch.is_tensor(lattice_params):
            lattice_params_np = lattice_params.cpu().numpy()
        else:
            lattice_params_np = np.array(lattice_params)
        
        # Convert edge_index to bonds format
        bonds = edge_index_to_bonds(edge_index)
        
        # Make molecules whole in fractional coordinates
        # Uses the centralized function from crystal_utils
        whole_frac_coords = make_molecules_whole(frac_coords_np, bonds)
        
        # Convert to cartesian coordinates
        # 1. Get standardized 3x3 matrix (c || z) from parameters using crystal_utils
        lattice_matrix = lattice_params_to_matrix(*lattice_params_np)
        
        # 2. Apply transformation using utility function
        whole_cart_coords = frac_to_cart(whole_frac_coords, lattice_matrix)
        
        # Center the cartesian coordinates by subtracting the mean
        cart_coords_mean = whole_cart_coords.mean(axis=0)
        centered_cart_coords = whole_cart_coords - cart_coords_mean
        
        # Convert back to tensor and add to processed data
        processed_data['whole_cartesian_coords'] = torch.from_numpy(centered_cart_coords).float()
        
        print(f"    ✓ Added whole centered cartesian coordinates for {processed_data.get('refcode', 'unknown')}")
        
    except Exception as e:
        print(f"    ⚠️  Warning: Failed to compute whole cartesian coordinates for {processed_data.get('refcode', 'unknown')}: {e}")
        # Fallback: just convert fractional to cartesian without making whole
        try:
            frac_coords = processed_data['frac_coords']
            lattice_params = processed_data['lattice_1']
            
            if torch.is_tensor(frac_coords):
                frac_coords_np = frac_coords.cpu().numpy()
            else:
                frac_coords_np = np.array(frac_coords)
                
            if torch.is_tensor(lattice_params):
                lattice_params_np = lattice_params.cpu().numpy()
            else:
                lattice_params_np = np.array(lattice_params)
            
            # Use same standardized conversion for fallback
            lattice_matrix = lattice_params_to_matrix(*lattice_params_np)
            cart_coords = frac_to_cart(frac_coords_np, lattice_matrix)
            
            # Center the coordinates even in fallback case
            cart_coords_mean = cart_coords.mean(axis=0)
            centered_cart_coords = cart_coords - cart_coords_mean
            
            processed_data['whole_cartesian_coords'] = torch.from_numpy(centered_cart_coords).float()
            print(f"    ✓ Added fallback centered cartesian coordinates for {processed_data.get('refcode', 'unknown')}")
        except Exception as e2:
            print(f"    ✗ Failed to add any cartesian coordinates for {processed_data.get('refcode', 'unknown')}: {e2}")
            # Create dummy coordinates as last resort
            num_atoms = len(processed_data['atom_types'])
            processed_data['whole_cartesian_coords'] = torch.zeros(num_atoms, 3).float()
    
    return processed_data


def batch_process_crystals(
    input_path: Union[str, List[str]], 
    output_path: str,
    n_jobs: int = -1,
    remove_hydrogens: bool = False,
    max_files: Optional[int] = None,
    timeout_per_file: int = 300,
    skip_rdkit: bool = False,
    tag_hydrogen_bonding: bool = False,
    find_aromatic_rings: bool = False
) -> Dict[str, Any]:
    """
    Batch process crystal structures and save them in a format suitable for CrystalDataset.
    
    Args:
        input_path (Union[str, List[str]]): Path to directory containing mmCIF files, 
                                          or list of mmCIF file paths.
        output_path (str): Path where the processed data will be saved (.pt or .pkl).
        n_jobs (int): Number of parallel jobs. Default -1 uses all available cores.
        remove_hydrogens (bool): Whether to remove hydrogen atoms from structures.
        max_files (int, optional): Maximum number of files to process.
        timeout_per_file (int): Timeout in seconds for processing each file.
        skip_rdkit (bool): If True, skip RDKit processing for faster execution.
        tag_hydrogen_bonding (bool): If True and remove_hydrogens=True, tag atoms as H-bond donors/acceptors.
        find_aromatic_rings (bool): If True, identify aromatic rings and return ring data.
        
    Returns:
        Dict[str, Any]: Processing summary with statistics and results.
    """
    print(f"Starting batch processing of crystal structures...")
    print(f"Output will be saved to: {output_path}")
    
    # Collect input files
    if isinstance(input_path, str):
        if os.path.isdir(input_path):
            mmcif_files = []
            for ext in ['*.mmcif', '*.cif']:
                mmcif_files.extend(Path(input_path).glob(ext))
            mmcif_files = [str(f) for f in mmcif_files]
        elif os.path.isfile(input_path):
            mmcif_files = [input_path]
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
    else:
        mmcif_files = input_path
    
    # Limit number of files if specified
    if max_files is not None:
        mmcif_files = mmcif_files[:max_files]
    
    print(f"Found {len(mmcif_files)} crystal structure files to process")
    
    if len(mmcif_files) == 0:
        raise ValueError("No crystal structure files found to process")
    
    # Process files in parallel
    print(f"Processing files using {n_jobs} parallel jobs...")
    
    def process_single_file_wrapper(file_path):
        """Wrapper function for parallel processing."""
        try:
            # Import processing functions locally to avoid circular imports
            from ..processing import process_single_mmcif_file, process_single_crystal, parse_mmcif_to_crystal_dict
            
            print(f"Processing: {os.path.basename(file_path)}")
            
            # Parse mmCIF file
            crystal_dict = parse_mmcif_to_crystal_dict(file_path)
            
            # Apply symmetry operations to generate full unit cell
            from ..processing.final_crystal_processor import apply_symmetry_operations
            crystal_dict = apply_symmetry_operations(crystal_dict)
            
            # Check if symmetry operations failed
            if isinstance(crystal_dict, dict) and 'error' in crystal_dict:
                return {
                    'status': 'error',
                    'refcode': crystal_dict.get('refcode', 'unknown'),
                    'file_path': file_path,
                    'error': crystal_dict['error']
                }
            
            # Process crystal structure
            processed_data = process_single_crystal(
                crystal_dict, 
                RemoveHs=remove_hydrogens,
                conformer_timeout=30,
                skip_rdkit=skip_rdkit,
                tag_hydrogen_bonding=tag_hydrogen_bonding,
                find_aromatic_rings=find_aromatic_rings  # Captured from outer scope
            )
            
            # Check if processing was successful
            if isinstance(processed_data, dict) and 'error' in processed_data:
                return {
                    'status': 'error',
                    'refcode': crystal_dict.get('refcode', 'unknown'),
                    'file_path': file_path,
                    'error': processed_data['error']
                }
            
            # Add file path for reference
            processed_data['file_path'] = file_path
            
            # Compute whole cartesian coordinates
            processed_data = add_whole_cartesian_coords(processed_data)
            
            return {
                'status': 'success',
                'refcode': processed_data['refcode'],
                'file_path': file_path,
                'data': processed_data
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'refcode': os.path.splitext(os.path.basename(file_path))[0],
                'file_path': file_path,
                'error': str(e)
            }
    
    # Execute parallel processing
    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout_per_file)(
        delayed(process_single_file_wrapper)(file_path) 
        for file_path in mmcif_files
    )
    # results = [process_single_file_wrapper(file_path) for file_path in mmcif_files]
    
    # Separate successful and failed results
    successful_data = []
    failed_results = []
    
    for result in results:
        if result['status'] == 'success':
            successful_data.append(result['data'])
        else:
            failed_results.append(result)
    
    # Save successful results
    if len(successful_data) > 0:
        save_processed_data(successful_data, output_path)
        print(f"Successfully saved {len(successful_data)} processed crystal structures to {output_path}")
    else:
        print("No crystal structures were successfully processed!")
    
    # Print summary
    total_files = len(mmcif_files)
    successful_count = len(successful_data)
    failed_count = len(failed_results)
    
    print(f"\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_files}")
    print(f"Successful: {successful_count} ({successful_count/total_files*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_files*100:.1f}%)")
    
    if successful_count > 0:
        print(f"\nSuccessfully processed crystals:")
        for data in successful_data[:10]:  # Show first 10
            print(f"  ✓ {data['refcode']}")
        if len(successful_data) > 10:
            print(f"  ... and {len(successful_data) - 10} more")
    
    if failed_count > 0:
        print(f"\nFailed to process:")
        for result in failed_results[:10]:  # Show first 10 failures
            print(f"  ✗ {result['refcode']}: {result['error'][:100]}...")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more failures")
    
    return {
        'total_files': total_files,
        'successful_count': successful_count,
        'failed_count': failed_count,
        'success_rate': successful_count / total_files if total_files > 0 else 0,
        'successful_data': successful_data,
        'failed_results': failed_results,
        'output_path': output_path if successful_count > 0 else None
    }


def save_processed_data(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save processed crystal data to file.
    
    Args:
        data (List[Dict[str, Any]]): List of processed crystal data dictionaries.
        output_path (str): Path to save the data.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path is not empty
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays to tensors for better PyTorch compatibility
    converted_data = []
    
    for item in data:
        converted_item = {}
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                converted_item[key] = torch.from_numpy(value)
            elif torch.is_tensor(value):
                converted_item[key] = value
            else:
                converted_item[key] = value
        converted_data.append(converted_item)
    
    # Save based on file extension
    if output_path.endswith('.pt'):
        torch.save(converted_data, output_path)
    elif output_path.endswith('.pkl'):
        with open(output_path, 'wb') as f:
            pickle.dump(converted_data, f)
    else:
        # Default to .pt format
        torch.save(converted_data, output_path)
    
    print(f"Saved {len(converted_data)} processed crystal structures to {output_path}")


def load_processed_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load processed crystal data from file.
    
    Args:
        file_path (str): Path to the processed data file.
        
    Returns:
        List[Dict[str, Any]]: List of processed crystal data dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    
    if file_path.endswith('.pt'):
        data = torch.load(file_path, map_location='cpu')
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # Try to load as torch file first
        try:
            data = torch.load(file_path, map_location='cpu')
        except:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
    
    print(f"Loaded {len(data)} processed crystal structures from {file_path}")
    return data


def split_processed_data(
    input_path: str,
    output_dir: str,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: Optional[float] = None,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Split processed crystal data into train/validation/test sets.
    
    Args:
        input_path (str): Path to the processed data file.
        output_dir (str): Directory to save the split datasets.
        train_split (float): Fraction for training set.
        val_split (float): Fraction for validation set.
        test_split (float, optional): Fraction for test set. If None, uses remainder.
        random_seed (int): Random seed for reproducible splits.
        
    Returns:
        Dict[str, str]: Paths to the created split files.
    """
    # Load data
    data = load_processed_data(input_path)
    total_samples = len(data)
    
    # Calculate split sizes
    if test_split is None:
        test_split = 1.0 - train_split - val_split
    
    train_size = int(train_split * total_samples)
    val_size = int(val_split * total_samples)
    test_size = total_samples - train_size - val_size
    
    print(f"Splitting {total_samples} samples:")
    print(f"  Train: {train_size} ({train_split:.1%})")
    print(f"  Validation: {val_size} ({val_split:.1%})")
    print(f"  Test: {test_size} ({test_split:.1%})")
    
    # Create random permutation for splitting
    np.random.seed(random_seed)
    indices = np.random.permutation(total_samples)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create split datasets
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save split datasets
    train_path = os.path.join(output_dir, 'train.pt')
    val_path = os.path.join(output_dir, 'val.pt')
    test_path = os.path.join(output_dir, 'test.pt')
    
    save_processed_data(train_data, train_path)
    save_processed_data(val_data, val_path)
    save_processed_data(test_data, test_path)
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path
    }


def create_dataset_info(processed_data_path: str) -> Dict[str, Any]:
    """
    Create summary information about a processed dataset.
    
    Args:
        processed_data_path (str): Path to processed data file.
        
    Returns:
        Dict[str, Any]: Dataset information and statistics.
    """
    data = load_processed_data(processed_data_path)
    
    if len(data) == 0:
        return {'error': 'Empty dataset'}
    
    # Collect statistics
    num_samples = len(data)
    refcodes = [item.get('refcode', 'unknown') for item in data]
    smiles = [item.get('smiles', 'unknown') for item in data]
    
    # Get tensor statistics for first valid sample
    sample_stats = {}
    for item in data:
        if 'atom_types' in item:
            sample_stats = {
                'num_atoms_range': (
                    min(len(item['atom_types']) for item in data if 'atom_types' in item),
                    max(len(item['atom_types']) for item in data if 'atom_types' in item)
                ),
                'avg_num_atoms': np.mean([len(item['atom_types']) for item in data if 'atom_types' in item]),
                'num_edges_range': (
                    min(item['edge_index'].shape[1] for item in data if 'edge_index' in item),
                    max(item['edge_index'].shape[1] for item in data if 'edge_index' in item)
                )
            }
            break
    
    unique_smiles = list(set(smiles))
    
    info = {
        'num_samples': num_samples,
        'unique_structures': len(set(refcodes)),
        'unique_molecules': len(unique_smiles),
        'sample_refcodes': refcodes[:10],  # First 10 refcodes
        'sample_smiles': unique_smiles[:10],  # First 10 unique SMILES
        **sample_stats
    }
    
    return info


if __name__ == '__main__':
    """
    Command-line interface for batch processing crystal structures.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process crystal structures")
    parser.add_argument('input_path', help='Path to directory containing mmCIF files or single mmCIF file')
    parser.add_argument('output_path', help='Path to save processed data (.pt or .pkl)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--remove_hydrogens', action='store_true', help='Remove hydrogen atoms')
    parser.add_argument('--tag_hydrogen_bonding', action='store_true', help='Tag H-bond donors/acceptors (requires --remove_hydrogens)')
    parser.add_argument('--find_aromatic_rings', action='store_true', help='Identify aromatic rings and return ring data')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    parser.add_argument('--split', action='store_true', help='Split data into train/val/test sets')
    parser.add_argument('--split_dir', help='Directory to save split datasets')
    
    args = parser.parse_args()
    
    # Batch process
    results = batch_process_crystals(
        input_path=args.input_path,
        output_path=args.output_path,
        n_jobs=args.n_jobs,
        remove_hydrogens=args.remove_hydrogens,
        max_files=args.max_files,
        tag_hydrogen_bonding=args.tag_hydrogen_bonding,
        find_aromatic_rings=args.find_aromatic_rings
    )
    
    # Split if requested
    if args.split and results['successful_count'] > 0:
        split_dir = args.split_dir or os.path.dirname(args.output_path)
        split_paths = split_processed_data(args.output_path, split_dir)
        print(f"\nDataset split saved to:")
        for split_name, split_path in split_paths.items():
            print(f"  {split_name}: {split_path}")
    
    # Print dataset info
    if results['successful_count'] > 0:
        info = create_dataset_info(args.output_path)
        print(f"\nDataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}") 