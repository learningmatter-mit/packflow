#!/usr/bin/env python3
"""
Script to create train/val/test splits from the homomolecular crystal dataset.

This script:
1. Filters the dataset to structures with ≤MAX_ATOMS atoms
2. Uses family-based exclusion to create splits ensuring no refcode families
   are split across train/val/test sets
3. Places all Table S1 refcodes and their families in test set
4. Creates train/val/test directories and copies CIF files
5. Generates annotated JSON files with metadata
6. Uses 80:10:10 split ratio
"""

import os
import json
import random
import math
import shutil

from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DATASET_JSON = os.environ.get("DATASET_PATH", "data/homomolecular_large_dataset.json")
CIF_SOURCE_DIR = "data/homomolecular_large_cifs"

# Output directories
OUTPUT_BASE_DIR = "data"
TRAIN_DIR = f"{OUTPUT_BASE_DIR}/train_large"
VAL_DIR = f"{OUTPUT_BASE_DIR}/val_large" 
TEST_DIR = f"{OUTPUT_BASE_DIR}/test_large"

# Split ratios (80:10:10)
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1

# Maximum atoms per structure
MAX_ATOMS = 250

# For reproducibility
RANDOM_SEED = 42
# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    """Custom encoder to handle Decimal objects from ijson."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def create_directories():
    """Create output directories for train/val/test splits."""
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def load_dataset(json_path):
    """Load dataset.json and filter structures with ≤MAX_ATOMS and non-zero atoms."""
    print(f"Loading dataset from {json_path}...")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Filter out entries with zero atoms
    filtered_data = [entry for entry in data if len(entry.get("atoms", [])) > 0]
    print(f"Filtered out {len(data) - len(filtered_data)} entries with zero atoms.")
    
    # Further filter to entries with ≤MAX_ATOMS
    small_data = [entry for entry in filtered_data if len(entry.get("atoms", [])) <= MAX_ATOMS]
    print(f"Filtered to {len(small_data)} entries with ≤{MAX_ATOMS} atoms.")
    
    return small_data

def write_json_data(filename, data):
    """Write data to JSON with indentation."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def get_refcode_from_entry(entry):
    """Extract refcode from dataset entry."""
    return entry.get("refcode", "").upper()

def get_refcode_family(refcode):
    """Extract refcode family (base part before any numbers)."""
    import re
    # Remove trailing numbers to get family
    family = re.sub(r'\d+$', '', refcode)
    return family.upper()

def random_item_split(items, rng, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    """Split items randomly into train/val/test sets."""
    rng.shuffle(items)
    n = len(items)
    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))
    
    train_items = items[:n_train]
    val_items = items[n_train:n_train+n_val]
    test_items = items[n_train+n_val:]
    
    return train_items, val_items, test_items

def copy_cif_file(refcode, target_dir):
    """Copy CIF file to target directory."""
    # Try both .cif and .mmcif extensions
    for ext in ['.cif', '.mmcif']:
        source_file = Path(CIF_SOURCE_DIR) / f"{refcode}{ext}"
        if source_file.exists():
            target_file = Path(target_dir) / f"{refcode}{ext}"
            shutil.copy2(source_file, target_file)
            return True
    
    print(f"Warning: CIF file not found for {refcode}")
    return False

def create_family_based_splits(dataset, rng):
    """
    Create splits using family-based approach to ensure no refcode families
    are split across train/val/test sets. 
    """
    
    # Group entries by refcode family
    family_to_entries = defaultdict(list)
    
    for entry in dataset:
        refcode = get_refcode_from_entry(entry)
        family = get_refcode_family(refcode)
        family_to_entries[family].append(entry)
    
    print(f"Found {len(family_to_entries)} unique refcode families")
    
    # Create list of families to split
    all_families = list(family_to_entries.items())
    rng.shuffle(all_families)
    
    # Calculate target sizes based on total dataset
    total_entries = len(dataset)
    target_train_size = int(total_entries * TRAIN_FRAC)
    target_val_size = int(total_entries * VAL_FRAC)
    target_test_size = int(total_entries * TEST_FRAC)
    
    print(f"Target sizes: Train={target_train_size}, Val={target_val_size}, Test={target_test_size}")
    
    train_entries = []
    val_entries = []
    test_entries = []
    
    # Greedy allocation to splits
    # We try to fill Test, then Val, then Train
    
    for family, entries in all_families:
        # Check where to put this family
        if len(test_entries) < target_test_size:
            test_entries.extend(entries)
        elif len(val_entries) < target_val_size:
            val_entries.extend(entries)
        else:
            train_entries.extend(entries)
            
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_entries)} entries ({len(train_entries)/total_entries*100:.1f}%)")
    print(f"  Val: {len(val_entries)} entries ({len(val_entries)/total_entries*100:.1f}%)")
    print(f"  Test: {len(test_entries)} entries ({len(test_entries)/total_entries*100:.1f}%)")
    
    # Verify no family conflicts
    train_families = set()
    val_families = set()
    test_families = set()
    
    for entry in train_entries:
        train_families.add(get_refcode_family(get_refcode_from_entry(entry)))
    
    for entry in val_entries:
        val_families.add(get_refcode_family(get_refcode_from_entry(entry)))
    
    for entry in test_entries:
        test_families.add(get_refcode_family(get_refcode_from_entry(entry)))
    
    # Check for conflicts
    conflicts = (train_families & val_families) | (train_families & test_families) | (val_families & test_families)
    if conflicts:
        print(f"WARNING: Family conflicts detected: {conflicts}")
    else:
        print("✓ No family conflicts detected")
    
    return train_entries, val_entries, test_entries

def process_split(entries, split_name, target_dir):
    """Process a dataset split: copy files and create metadata."""
    print(f"\nProcessing {split_name} split ({len(entries)} entries)...")
    
    success_count = 0
    annotated_entries = []
    
    for entry in entries:
        if success_count % 100 == 0:
            print(f"  Processed {success_count}/{len(entries)}...", end='\r')
        refcode = get_refcode_from_entry(entry)
        
        # Copy CIF file
        if copy_cif_file(refcode, target_dir):
            success_count += 1
        
        # Create annotated entry
        annotated_entry = {
            "refcode": refcode,
            "entry": entry,
            "split": split_name,
            "n_atoms": len(entry.get("atoms", [])),
            "cell_volume": calculate_cell_volume(entry.get("cell", {})),
            "formula": entry.get("smiles", ""),
        }
        annotated_entries.append(annotated_entry)
    
    # Save annotated metadata
    metadata_file = Path(target_dir) / f"{split_name}_metadata.json"
    write_json_data(metadata_file, annotated_entries)
    
    print(f"Successfully copied {success_count}/{len(entries)} CIF files to {target_dir}")
    print(f"Saved metadata to {metadata_file}")
    
    return annotated_entries

def calculate_cell_volume(cell_params):
    """Calculate unit cell volume from cell parameters."""
    try:
        a = cell_params.get("a", 0)
        b = cell_params.get("b", 0) 
        c = cell_params.get("c", 0)
        alpha = cell_params.get("alpha", 90) * math.pi / 180
        beta = cell_params.get("beta", 90) * math.pi / 180
        gamma = cell_params.get("gamma", 90) * math.pi / 180
        
        volume = a * b * c * math.sqrt(1 + 2*math.cos(alpha)*math.cos(beta)*math.cos(gamma) 
                                       - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2)
        return volume
    except:
        return 0.0



# ----------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Creating Crystal Structure Dataset Splits (Streaming Optimized)")
    print("=" * 60)
    
    # Create output directories
    create_directories()
    
    # Check if ijson is available
    import ijson
    
    print(f"Scanning dataset at {DATASET_JSON} for refcodes and filtering...")
    
    # Pass 1: Collect valid refcodes and families (Memory Efficient)
    # We store: valid_refcodes = set(refcodes)
    #           family_to_refcodes = {family: [refcode, ...]}
    valid_refcodes = set()
    family_to_refcodes = defaultdict(list)
    
    total_scanned = 0
    passed_filter = 0
    
    with open(DATASET_JSON, 'rb') as f:
        # Use ijson to iterate over items in the list one by one
        # "item" matches each object in the root array
        parser = ijson.items(f, 'item')
        
        for entry in parser:
            total_scanned += 1
            if total_scanned % 10000 == 0:
                print(f"  Scanned {total_scanned} entries...", end='\r')
            
            # Check constraints (avoid loading huge data structures if possible, but here we load the item)
            # Efficiently checking length without deep inspection if possible, but ijson loads the dict.
            # This is still better than loading ALL dicts at once.
            
            atoms = entry.get("atoms", [])
            n_atoms = len(atoms)
            
            if n_atoms > 0 and n_atoms <= MAX_ATOMS:
                refcode = get_refcode_from_entry(entry)
                family = get_refcode_family(refcode)
                
                valid_refcodes.add(refcode)
                family_to_refcodes[family].append(refcode)
                passed_filter += 1
                
    print(f"\nScan complete.")
    print(f"  Total scanned: {total_scanned}")
    print(f"  Passed filter (1 <= atoms <= {MAX_ATOMS}): {passed_filter}")
    
    if passed_filter == 0:
        print("No valid entries found!")
        return

    # Create splits based on families
    print(f"\nCreating splits with ratios {TRAIN_FRAC}:{VAL_FRAC}:{TEST_FRAC}")
    rng = random.Random(RANDOM_SEED)
    
    # Logic similar to original, but working with refcodes instead of full entries
    all_families = list(family_to_refcodes.items())
    rng.shuffle(all_families)
    
    target_train_size = int(passed_filter * TRAIN_FRAC)
    target_val_size = int(passed_filter * VAL_FRAC)
    target_test_size = int(passed_filter * TEST_FRAC)
    
    # We track refcodes assigned to each split so we can route them in Pass 2
    # refcode -> split_name
    refcode_to_split = {}
    
    # Metrics
    train_count = 0
    val_count = 0
    test_count = 0
    
    current_test_size = 0
    current_val_size = 0
    
    for family, refcodes in all_families:
        n_refs = len(refcodes)
        
        # Greedy allocation
        if current_test_size < target_test_size:
            split = "test"
            current_test_size += n_refs
            test_count += n_refs
        elif current_val_size < target_val_size:
            split = "val"
            current_val_size += n_refs
            val_count += n_refs
        else:
            split = "train"
            train_count += n_refs
            
        for ref in refcodes:
            refcode_to_split[ref] = split
            
    print(f"\nSplit assignment complete:")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")
    
    # Open output files for streaming write
    # We will write a JSON list manually to allow streaming: '[' then objects then ']'
    files = {
        "train": open(Path(TRAIN_DIR) / "train_metadata.json", "w"),
        "val":   open(Path(VAL_DIR) / "val_metadata.json", "w"),
        "test":  open(Path(TEST_DIR) / "test_metadata.json", "w")
    }
    
    # Initialize files with opening bracket
    for f_handle in files.values():
        f_handle.write("[\n")
        
    # Keep track of whether we need a comma (first item vs others)
    is_first = { k: True for k in files }
    
    # Track statistics for summary
    split_stats = {
        "train": {"count": 0, "atom_sum": 0},
        "val":   {"count": 0, "atom_sum": 0},
        "test":  {"count": 0, "atom_sum": 0}
    }

    # Pass 2: Stream and Distribute
    print(f"\nPass 2: processing and writing files...")
    processed_count = 0
    
    with open(DATASET_JSON, 'rb') as f:
        parser = ijson.items(f, 'item')
        
        for entry in parser:
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"  Processed {processed_count} entries...", end='\r')
                
            refcode = get_refcode_from_entry(entry)
            
            # Skip if filtered out in pass 1
            if refcode not in refcode_to_split:
                continue
                
            split_name = refcode_to_split[refcode]
            target_dir = TRAIN_DIR if split_name == "train" else (VAL_DIR if split_name == "val" else TEST_DIR)
            
            # Copy CIF
            copy_cif_file(refcode, target_dir)
            
            # Create annotated entry
            n_atoms = len(entry.get("atoms", []))
            annotated_entry = {
                "refcode": refcode,
                "entry": entry,
                "split": split_name,
                "n_atoms": n_atoms,
                "cell_volume": calculate_cell_volume(entry.get("cell", {})),
                "formula": entry.get("smiles", ""),
            }
            
            # Write key statistics
            split_stats[split_name]["count"] += 1
            split_stats[split_name]["atom_sum"] += n_atoms
            
            # Write to file
            f_out = files[split_name]
            if not is_first[split_name]:
                f_out.write(",\n")
            else:
                is_first[split_name] = False
                
            json.dump(annotated_entry, f_out, indent=2, cls=DecimalEncoder)

    # Close JSON arrays and files
    for f_handle in files.values():
        f_handle.write("\n]")
        f_handle.close()
        
    print(f"\nProcessing complete.")
    
    # Create final summary
    summary = {
        "total_structures": passed_filter,
        "max_atoms_filter": MAX_ATOMS,
        "splits": {},
        "random_seed": RANDOM_SEED
    }
    
    for split in ["train", "val", "test"]:
        count = split_stats[split]["count"]
        atom_sum = split_stats[split]["atom_sum"]
        summary["splits"][split] = {
            "count": count,
            "percentage": count / passed_filter * 100 if passed_filter else 0,
            "avg_atoms": atom_sum / count if count else 0
        }
        
    summary_file = f"{OUTPUT_BASE_DIR}/split_summary.json"
    write_json_data(summary_file, summary)
    
    print("\n" + "=" * 60)
    print("Dataset splitting completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

