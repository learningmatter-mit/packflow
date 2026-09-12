#!/usr/bin/env python3
"""
Process Pre-made Split Data (Skip RDKit)

This script processes pre-made train/val/test splits using the batch_processor.py functionality
with skip_rdkit=True for faster processing. Each split directory contains mmCIF files that will 
be processed and saved as .pt files in the format expected by crystal_flow_matching.py and 
test_crystal_flow_matching.py.
"""

import os
import sys
from pathlib import Path
from packflow.utils.batch_processor import batch_process_crystals

def process_premade_splits_skip_rdkit(
    data_dir: str = "data",
    output_dir: str = "processed_data_lt200_symmetrized_skip_rdkit_niggli",
    n_jobs: int = -1,
    remove_hydrogens: bool = True,
    max_files: int = None,
    timeout_per_file: int = 300,
    skip_rdkit: bool = True,
    tag_hydrogen_bonding: bool = False,
    find_aromatic_rings: bool = False
):
    """
    Process pre-made train/val/test splits with RDKit processing skipped for faster execution.
    
    Args:
        data_dir (str): Directory containing train/, val/, and test/ subdirectories
        output_dir (str): Directory to save processed .pt files
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        remove_hydrogens (bool): Whether to remove hydrogen atoms
        max_files (int): Maximum files per split to process (None for all)
        timeout_per_file (int): Timeout per file in seconds
        skip_rdkit (bool): Whether to skip RDKit processing (default: True for faster execution)
        tag_hydrogen_bonding (bool): If True and remove_hydrogens=True, tag atoms as H-bond donors/acceptors
        find_aromatic_rings (bool): If True, identify aromatic rings and return ring data
    """
    
    # Define splits and their paths
    splits = ['train', 'val', 'test']
    
    # Verify input directories exist
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
        else:
            # Count mmCIF files in the directory
            mmcif_files = list(Path(split_dir).glob("*.mmcif")) + list(Path(split_dir).glob("*.cif"))
            print(f"Found {len(mmcif_files)} mmCIF files in {split_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    results = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"PROCESSING {split.upper()} SPLIT")
        print(f"{'='*60}")
        
        split_dir = os.path.join(data_dir, split)
        output_path = os.path.join(output_dir, f"{split}.pt")
        
        if not os.path.exists(split_dir):
            print(f"Skipping {split} - directory not found: {split_dir}")
            continue
        
        # Check if there are any mmCIF files
        mmcif_files = list(Path(split_dir).glob("*.mmcif")) + list(Path(split_dir).glob("*.cif"))
        if not mmcif_files:
            print(f"Skipping {split} - no mmCIF files found in: {split_dir}")
            continue
        
        print(f"Input directory: {split_dir}")
        print(f"Output file: {output_path}")
        print(f"Remove hydrogens: {remove_hydrogens}")
        print(f"Tag hydrogen bonding: {tag_hydrogen_bonding}")
        print(f"Find aromatic rings: {find_aromatic_rings}")
        print(f"Skip RDKit: {skip_rdkit}")
        
        try:
            # Process the split using batch_process_crystals with skip_rdkit=True
            result = batch_process_crystals(
                input_path=split_dir,
                output_path=output_path,
                n_jobs=n_jobs,
                remove_hydrogens=remove_hydrogens,
                max_files=max_files,
                timeout_per_file=timeout_per_file,
                skip_rdkit=skip_rdkit,
                tag_hydrogen_bonding=tag_hydrogen_bonding,
                find_aromatic_rings=find_aromatic_rings
            )
            
            results[split] = result
            
            if result['successful_count'] > 0:
                print(f"✓ Successfully processed {split} split: {result['successful_count']}/{result['total_files']} files")
            else:
                print(f"✗ Failed to process any files in {split} split")
                
        except Exception as e:
            print(f"✗ Error processing {split} split: {e}")
            results[split] = {'error': str(e)}
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    total_processed = 0
    total_files = 0
    
    for split in splits:
        if split in results and 'successful_count' in results[split]:
            successful = results[split]['successful_count']
            total = results[split]['total_files']
            success_rate = results[split]['success_rate'] * 100
            
            print(f"{split.capitalize()}: {successful}/{total} files ({success_rate:.1f}% success)")
            total_processed += successful
            total_files += total
        elif split in results and 'error' in results[split]:
            print(f"{split.capitalize()}: Error - {results[split]['error']}")
        else:
            print(f"{split.capitalize()}: Skipped")
    
    if total_files > 0:
        overall_success_rate = (total_processed / total_files) * 100
        print(f"\nOverall: {total_processed}/{total_files} files ({overall_success_rate:.1f}% success)")
    
    print(f"\nProcessed files saved to: {output_dir}")
    print("Files are ready for use with crystal_flow_matching.py and test_crystal_flow_matching.py")
    print("Note: RDKit processing was skipped for faster execution")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Process pre-made train/val/test splits with RDKit processing skipped")
    parser.add_argument('--data_dir', default='data', 
                       help='Directory containing train/, val/, test/ subdirectories (default: data)')
    parser.add_argument('--output_dir', default='processed_data_lt200_symmetrized_skip_rdkit_niggli',
                       help='Directory to save processed .pt files (default: processed_data_lt200_symmetrized_skip_rdkit_niggli)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--remove_hydrogens', action='store_true', default=True,
                       help='Remove hydrogen atoms (default: True)')
    parser.add_argument('--tag_hydrogen_bonding', action='store_true',
                       help='Tag H-bond donors/acceptors (requires --remove_hydrogens)')
    parser.add_argument('--find_aromatic_rings', action='store_true',
                       help='Identify aromatic rings and return ring data')
    parser.add_argument('--max_files', type=int, 
                       help='Maximum files per split to process')
    parser.add_argument('--timeout_per_file', type=int, default=300,
                       help='Timeout per file in seconds (default: 300)')
    parser.add_argument('--skip_rdkit', action='store_true', default=True,
                       help='Skip RDKit processing for faster execution (default: True)')
    
    args = parser.parse_args()
    
    print("Processing pre-made splits with the following settings:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Parallel jobs: {args.n_jobs}")
    print(f"  Remove hydrogens: {args.remove_hydrogens}")
    print(f"  Tag hydrogen bonding: {args.tag_hydrogen_bonding}")
    print(f"  Find aromatic rings: {args.find_aromatic_rings}")
    print(f"  Max files per split: {args.max_files}")
    print(f"  Timeout per file: {args.timeout_per_file}s")
    print(f"  Skip RDKit: {args.skip_rdkit}")
    
    # Process the splits
    results = process_premade_splits_skip_rdkit(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        remove_hydrogens=args.remove_hydrogens,
        max_files=args.max_files,
        timeout_per_file=args.timeout_per_file,
        skip_rdkit=args.skip_rdkit,
        tag_hydrogen_bonding=args.tag_hydrogen_bonding,
        find_aromatic_rings=args.find_aromatic_rings
    )
