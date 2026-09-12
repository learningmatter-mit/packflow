
import os
import sys
import json
import tempfile
import subprocess
import torch
import shutil
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from pymatgen.core import Molecule, Structure
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms import isomorphism
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from packflow.utils.crystal_utils import lattice_matrix_to_params, lattice_params_to_matrix
from collections import Counter
import math

def decode_numpy_json(obj):
    """Recursively decode JSON containing ASE/NumPy serialized arrays."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            # Format: [shape, dtype, data]
            shape, dtype, data = obj["__ndarray__"]
            return np.array(data, dtype=dtype).reshape(shape)
        return {k: decode_numpy_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_numpy_json(v) for v in obj]
    return obj


def _extract_rdkit_bonds(mol) -> List[Tuple[int, int]]:
    """Extract bonds from RDKit molecule as (atom1_idx, atom2_idx) pairs.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        List of (atom1_idx, atom2_idx) tuples representing bonds
    """
    bonds = []
    for bond in mol.GetBonds():
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return bonds


def _expand_bonds_to_z_copies(mol_bonds: List[Tuple[int, int]], 
                               num_atoms_per_mol: int, 
                               z_value: int,
                               device: str = 'cpu') -> torch.Tensor:
    """Expand bonds from single molecule to Z copies in crystal.
    
    For each copy i (0 to Z-1), atom indices are offset by i * num_atoms_per_mol.
    Creates bidirectional edges (both directions for each bond).
    
    Args:
        mol_bonds: List of (atom1_idx, atom2_idx) tuples for one molecule
        num_atoms_per_mol: Number of atoms in one molecule copy
        z_value: Number of molecules in the unit cell
        device: Device to put the tensor on
    
    Returns:
        edge_index tensor of shape [2, E] with bidirectional edges
    """
    all_bonds = []
    for copy_idx in range(z_value):
        offset = copy_idx * num_atoms_per_mol
        for a1, a2 in mol_bonds:
            # Add bidirectional edges
            all_bonds.append((a1 + offset, a2 + offset))
            all_bonds.append((a2 + offset, a1 + offset))
    
    if not all_bonds:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    edge_index = torch.tensor(all_bonds, dtype=torch.long, device=device).T  # [2, E]
    return edge_index


class GenarrisWrapper:
    """
    Wrapper for Genarris to function as a crystal generator in the evaluation pipeline.
    
    This wrapper extracts molecule information from the input data (via CIF file lookups to get SMILES),
    creates a temporary configuration and directory structure for Genarris, executes Genarris via MPI,
    and then parses the generated structures back into PyTorch tensors.
    """
    
    def __init__(self, mode: str = "plain", genarris_path: str = None, python_path: str = None, n_procs: int = 4):
        """
        Initialize the Genarris wrapper.
        
        Args:
            mode: Generation mode. "plain" for random generation only, 
                  "rigid_press" for generation followed by symmetric rigid press optimization.
            genarris_path: Path to the Genarris repository root. If None, tries to find it 
                           relative to the project or assumes it's in PYTHONPATH.
            python_path: Path to the python executable to use for running Genarris (e.g., from a specific conda env).
                         If None, uses sys.executable (current env).
            n_procs: Number of MPI processes to use for Genarris execution.
        """
        self.mode = mode
        self.python_path = python_path if python_path else sys.executable
        self.n_procs = n_procs
        
        if self.mode not in ["plain", "rigid_press"]:
            raise ValueError(f"Unknown Genarris mode: {mode}. Must be 'plain' or 'rigid_press'.")
            
        # Locate Genarris
        if genarris_path:
            self.genarris_root = Path(genarris_path)
        elif os.environ.get("GENARRIS_ROOT"):
            self.genarris_root = Path(os.environ["GENARRIS_ROOT"])
        else:
            # Genarris ships as the pinned submodule at <repo>/external/Genarris.
            # This file is packflow/baselines/genarris_wrapper.py -> repo root is 3 levels up.
            project_root = Path(__file__).parent.parent.parent
            self.genarris_root = project_root / "external" / "Genarris"
            
        if not self.genarris_root.exists():
            print(f"WARNING: Genarris root not found at {self.genarris_root}. Assuming 'gnrs' is importable.")
            self.genarris_root = None
            
        # Add to path if found (for local imports if any, though we use subprocess main usually)
        if self.genarris_root:
            sys.path.append(str(self.genarris_root))

    def process_structures_json(self, temp_path: Path):
        """Parse structures.json and return Structure objects directly.
        
        Returns list of (Structure, index) tuples to preserve atom ordering
        (avoids CIF round-trip which can reorder atoms).
        """
        # Determine priority path based on likely last step
        # But safest is to check if 'symm_rigid_press' exists, use that.
        # Else use 'generation'.
        
        rigid_press_json = temp_path / "structures" / "symm_rigid_press" / "structures.json"
        generation_json = temp_path / "structures" / "generation" / "structures.json"
        
        if rigid_press_json.exists():
            json_path = rigid_press_json
            print(f"Found Rigid Press output at {json_path}")
        elif generation_json.exists():
            json_path = generation_json
            print(f"Found Generation output at {json_path}")
        else:
            # Fallback search
            search_res = list(temp_path.rglob("structures.json"))
            if search_res:
                # Use the one with the latest modification time or deepest path?
                # Usually we want the last stage.
                # Let's pick the one that contains 'rigid_press' if available
                rigid = [p for p in search_res if "rigid_press" in str(p)]
                if rigid:
                     json_path = rigid[0]
                else:
                     json_path = search_res[0]
            else:
                return []


        print(f"Parsing structures from {json_path}")
        try:
            from ase.io.jsonio import decode as ase_decode
            from pymatgen.io.ase import AseAtomsAdaptor
            
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
            
            structures = []
            
            # Genarris JSON format: dict of {structure_id: ASE-encoded Atoms object}
            # json.load() parses values as Python dicts, but ASE decode expects JSON strings
            for i, (key, value) in enumerate(raw_data.items()):
                try:
                    # Re-serialize to JSON string for ASE decode
                    json_str = json.dumps(value)
                    
                    # Decode ASE Atoms object from JSON string
                    atoms = ase_decode(json_str)
                    
                    # Convert ASE Atoms to pymatgen Structure
                    s = AseAtomsAdaptor.get_structure(atoms)
                    
                    # Remove Hydrogens to match ground truth
                    s.remove_species(["H", "D", "T"])
                    
                    structures.append((s, i))
                except Exception as e:
                    print(f"Failed to decode structure {key}: {e}")
            
            print(f"Successfully parsed {len(structures)} structures from JSON")
            return structures
            
        except Exception as e:
            print(f"Error processing structures.json: {e}")
            import traceback
            traceback.print_exc()
            return []

    def sample(self, crystal_data: Dict, num_seeds: int = 1, z_value: int = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate crystal structures using Genarris.
        
        Args:
            crystal_data: Dictionary containing crystal information. Must contain 'refcode'.
            num_seeds: Number of structures to generate.
            z_value: Optional override for Z. If None, extracted from CIF.
            
        Returns:
            List of (cart_coords, lattice_params) tuples as tensors.
        """
        device = crystal_data['atom_types'].device
        refcode = crystal_data.get('refcode')
        
        if not refcode:
            print("ERROR: No refcode found in crystal_data. Cannot look up SMILES for Genarris.")
            return [(None, None, None)] * num_seeds
            
        # 1. Get SMILES
        smiles = crystal_data.get('smiles')
        
        if not smiles:
             print(f"ERROR: No SMILES found for {refcode}.")
             return [(None, None, None)] * num_seeds

        # Start timing (Pre-processing)
        t_start = time.time()

        # Infer Z from SMILES
        fragments = smiles.split('.')
        counts = Counter(fragments)
        # Assuming Z is the GCD of counts, i.e., the SMILES string represents the full unit cell content or a multiple of Z
        z_value = math.gcd(*counts.values())
        print(f"Inferred Z={z_value} from SMILES counts: {counts} for {refcode}")

        try:
            # 2. Identify Repeating Unit
            mol_smiles = self._get_repeating_unit(smiles, z_value)
            
            # 3. Build RDKit Molecule & Generate Conformer
            # User wants: "directly create conformer from the smiles string... add hydrogens... remove all hydrogens before genarris"
            
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                raise ValueError(f"Failed to parse SMILES: {mol_smiles}")
                
            mol = Chem.AddHs(mol)
            
            # Generate Conformer
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res == -1:
                 res = AllChem.EmbedMolecule(mol, AllChem.ETKDG(), useRandomCoords=True)
                 if res == -1:
                      # Fallback to 2D -> 3D estimation is better than nothing, or failure
                      print(f"Warning: RDKit conformer generation failed for {refcode}. Using Compute2DCoords.")
                      AllChem.Compute2DCoords(mol)
            
            # Remove Hydrogens for Genarris input (as requested: "remove all the hydrogens before you run genarris command")
            mol = Chem.RemoveHs(mol)
            
            # Optimize (optional, might help strictly geometric issues, but removed Hs might make it weird? 
            # Usually optimization on heavy atoms only is fine if topology is good, but RDKit MMFF needs Hs often.
            # Let's skip optimization after RemoveHs to avoid issues, or optimize BEFORE removing Hs.
            # "add hydrogens for conformer creation and then you can remove all the hydrogens" -> implied opt happens with Hs usually.
            # Let's assume Embed is enough, or optimize with Hs first.
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass
                
            mol = Chem.RemoveHs(mol) # Redundant call if done above? No, I'll move RemoveHs after opt.

            # 4a. Extract bonds from RDKit molecule for visualization
            mol_bonds = _extract_rdkit_bonds(mol)
            num_atoms_per_mol = mol.GetNumAtoms()
            print(f"Extracted {len(mol_bonds)} bonds from molecule with {num_atoms_per_mol} atoms for {refcode}")

            # 4b. Generate XYZ string for Genarris
            conf = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            xyz_lines = [f"{num_atoms}", f"Generated from SMILES for {refcode}"]
            
            for i in range(num_atoms):
                pos = conf.GetAtomPosition(i)
                symbol = mol.GetAtomWithIdx(i).GetSymbol()
                xyz_lines.append(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
                
            mol_xyz_str = "\n".join(xyz_lines)

        except Exception as e:
            print(f"Failed to prepare molecule from SMILES for {refcode}: {e}")
            import traceback
            traceback.print_exc()
            return [(None, None, None, None)] * num_seeds



        results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            
            # 2. Write Molecule File
            mol_path = temp_path / "molecule.xyz"
            with open(mol_path, "w") as f:
                f.write(mol_xyz_str)
            
            # 3. Create Config
            conf_path = temp_path / "ui.conf"
            # If Z parsing failed, default to 4 (common for P21/c) or error out?
            # We'll assume z_value is valid if _prepare_molecule_from_cif didn't raise.
            if z_value is None: 
                print(f"WARNING: Z value could not be determined for {refcode}. Defaulting to Z=4.")
                z_value = 4
                
            self._create_config(conf_path, mol_path, z_value, num_seeds)
            
            # End Pre-processing / Start Generation
            t_prep_end = time.time()
            
            # 4. Run Genarris
            # We must use mpirun. Let's assume `mpirun` is in path.
            # Number of processors: Genarris is parallel.
            n_procs = self.n_procs
            
            cmd = [
                "mpirun", "-np", str(n_procs),
                self.python_path, "-m", "gnrs.cli",
                "--config", str(conf_path)
            ]
            
            # Set PYTHONPATH to include Genarris root if needed
            env = os.environ.copy()
            if self.genarris_root:
                env["PYTHONPATH"] = f"{str(self.genarris_root)}:{env.get('PYTHONPATH', '')}"
            
            try:
                # Remove capture_output=True to see output in real-time
                subprocess.run(cmd, env=env, check=True, cwd=str(temp_path)) # Inherit stdout/stderr
            except subprocess.CalledProcessError as e:
                print(f"Genarris execution failed for {refcode}: {e}")
                # output is already printed if we don't capture
                return [(None, None, None, None)] * num_seeds
            
            # End Generation
            t_gen_end = time.time()


            # 5. Parse Results
            struct_dir = temp_path / "structures"
            cif_files = list(struct_dir.glob("*.cif"))
            structures_from_json = []

            if not cif_files:
                # Handle Genarris 3.0 JSON output if no CIFs found
                # Returns list of (Structure, index) tuples to preserve atom ordering
                structures_from_json = self.process_structures_json(temp_path)

            if not cif_files and not structures_from_json:
                 print(f"WARNING: No structures generated for {refcode}")
                 return [(None, None, None, None)] * num_seeds
            
            # --- Parse ALL structures first ---
            valid_structures = []
            
            # Process structures from JSON (preserves atom ordering from Genarris)
            for s, idx in structures_from_json:
                try:
                    # Remove Hydrogens (already done in process_structures_json, but be defensive)
                    s.remove_species(["H", "D", "T"])
                    
                    # Extract Data
                    coords_tensor = torch.tensor(s.cart_coords, dtype=torch.float32, device=device)
                    # Extract lattice parameters directly from pymatgen lattice
                    lat = s.lattice
                    params = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
                         
                    site_zs = [site.specie.Z for site in s]
                    zs_tensor = torch.tensor(site_zs, dtype=torch.long, device=device)
                    lat_params_tensor = torch.tensor(params, dtype=torch.float32, device=device)
                    
                    # Identify Space Group
                    try:
                        spg_num = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
                    except:
                        spg_num = 0 # Fallback
                        

                    # Compute Z for this structure
                    total_atoms = len(site_zs)
                    if num_atoms_per_mol > 0 and total_atoms % num_atoms_per_mol == 0:
                        z_actual = total_atoms // num_atoms_per_mol
                    else:
                        z_actual = z_value  # Fallback to inferred Z
                    
                    # Expand bonds to Z copies
                    struct_edge_index = _expand_bonds_to_z_copies(mol_bonds, num_atoms_per_mol, z_actual, device)

                    valid_structures.append({
                        'coords': coords_tensor,
                        'lattice': lat_params_tensor,
                        'atoms': zs_tensor,
                        'spg': spg_num,
                        'filename': f"json_struct_{idx}",
                        'edge_index': struct_edge_index
                    })

                except Exception as e:
                    print(f"Failed to process structure {idx}: {e}")
            
            # Debug: show how many structures were processed from JSON
            print(f"DEBUG: Processed {len(structures_from_json)} from JSON, got {len(valid_structures)} valid before CIF processing")
            
            # Process CIF files (legacy path - may reorder atoms)
            for cif in cif_files:
                try:
                    s = Structure.from_file(str(cif))
                    
                    # Remove Hydrogens to match ground truth
                    s.remove_species(["H", "D", "T"])
                    
                    # Extract Data
                    coords_tensor = torch.tensor(s.cart_coords, dtype=torch.float32, device=device)
                    # Extract lattice parameters directly from pymatgen lattice
                    lat = s.lattice
                    params = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
                         
                    site_zs = [site.specie.Z for site in s]
                    zs_tensor = torch.tensor(site_zs, dtype=torch.long, device=device)
                    lat_params_tensor = torch.tensor(params, dtype=torch.float32, device=device)
                    
                    try:
                        spg_num = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
                    except:
                        spg_num = 0

                    total_atoms = len(site_zs)
                    if num_atoms_per_mol > 0 and total_atoms % num_atoms_per_mol == 0:
                        z_actual = total_atoms // num_atoms_per_mol
                    else:
                        z_actual = z_value
                    
                    struct_edge_index = _expand_bonds_to_z_copies(mol_bonds, num_atoms_per_mol, z_actual, device)

                    valid_structures.append({
                        'coords': coords_tensor,
                        'lattice': lat_params_tensor,
                        'atoms': zs_tensor,
                        'spg': spg_num,
                        'filename': cif.name,
                        'edge_index': struct_edge_index
                    })

                except Exception as e:
                    print(f"Failed to parse CIF {cif}: {e}")
            
            # --- FILTER BY ATOM COUNT ---
            # Now checking against GT atom count
            gt_labels = crystal_data['atom_types']
            final_structures = []
            
            # Debug: print expected vs actual atom counts
            if valid_structures:
                first_struct = valid_structures[0]
                print(f"DEBUG: GT has {gt_labels.size(0)} atoms, first generated structure has {first_struct['atoms'].size(0)} atoms")
            
            for struct in valid_structures:
                # Check 1: Total atom count must match GT
                if struct['atoms'].size(0) != gt_labels.size(0):
                    print(f"Warning: Discarding structure {struct['filename']} - atom count mismatch (Generated: {struct['atoms'].size(0)} vs GT: {gt_labels.size(0)})")
                    continue
                    
                # Check 2 (Optional but good): Atomic numbers should match ignoring order? 
                # Or exact element counts must match.
                # A simple sort check is robust.
                gen_elements_sorted, _ = torch.sort(struct['atoms'])
                gt_elements_sorted, _ = torch.sort(gt_labels)
                
                if not torch.equal(gen_elements_sorted.cpu(), gt_elements_sorted.cpu()):
                     print(f"Warning: Discarding structure {struct['filename']} - composition mismatch")
                     continue
                     
                final_structures.append(struct)
                
            valid_structures = final_structures
            # -----------------------------
            
            # --- CALCULATE WALL CLOCK TIME ---
            # Prep time: RDKit + Config writing
            # Gen time: Subprocess run
            # We generated `len(final_structures)` valid structures.
            # But we only requested `num_seeds`.
            # To be comparable to Flow Matching (which generates `num_seeds`), we normalize the Genarris time.
            # Time = Prep_Time + (Gen_Time_Total / N_Valid_Generated * N_Requested)
            
            prep_time = t_prep_end - t_start
            gen_time_total = t_gen_end - t_prep_end
            n_valid_generated = len(valid_structures)
            
            if n_valid_generated > 0:
                # Time per structure
                time_per_struct = gen_time_total / n_valid_generated
                # Normalized total time for the requested batch
                comparable_time = prep_time + (time_per_struct * num_seeds)
            else:
                comparable_time = prep_time + gen_time_total
                
            self.last_inference_time = comparable_time
            print(f"Timing for {refcode}: Prep={prep_time:.3f}s, GenTotal={gen_time_total:.3f}s, N_Gen={n_valid_generated}, Comparable={comparable_time:.3f}s")
            
            # -----------------------------

            # --- CHECK COUNT ---
            if len(valid_structures) == 0:
                print(f"WARNING: No valid structures generated for {refcode}. All seeds will fail.")
                return [(None, None, None, None)] * num_seeds
            
            if len(valid_structures) < num_seeds:
                print(f"WARNING: Generated ({len(valid_structures)}) < Requested ({num_seeds}) for {refcode}. Using available structures.")
                
            # --- DIVERSE SELECTION ---
            # Group by Space Group
            spg_groups = {}
            for struct in valid_structures:
                spg = struct['spg']
                if spg not in spg_groups:
                    spg_groups[spg] = []
                spg_groups[spg].append(struct)
            
            unique_spgs = list(spg_groups.keys())
            print(f"Generated {len(valid_structures)} structures across Space Groups: {sorted(unique_spgs)} for {refcode}")
            
            selected_structures = []
            
            # Calculate Quotas
            num_spgs = len(unique_spgs)
            base_quota = num_seeds // num_spgs
            remainder = num_seeds % num_spgs
            
            quotas = {spg: base_quota for spg in unique_spgs}
            
            # Distribute remainder to random (or first) SPGs
            # To be deterministic with fixed seed, sort keys first
            sorted_spgs = sorted(unique_spgs)
            
            # Using numpy random for selection (controlled by external seed)
            remainder_spgs = np.random.choice(sorted_spgs, size=remainder, replace=False)
            for spg in remainder_spgs:
                quotas[spg] += 1
                
            # Selection Loop (handling potential deficits)
            # If a group has fewer than quota, take all and redistribute deficit
            # Simple approach: Linear pass, if deficit, add to 'needed' pile, redistribute at end?
            # Or just iterative.
            
            # Simplification: Since we generate 50 PER SPG (requested), deficit is unlikely unless generation completely failed for that SPG.
            # But if we have valid_structures, we have at least some.
            
            for spg in sorted_spgs:
                candidates = spg_groups[spg]
                n_needed = quotas[spg]
                
                if len(candidates) >= n_needed:
                    # Sample without replacement
                    # Allow replace if somehow needed? No, guaranteed unique candidates.
                    idxs = np.random.choice(len(candidates), size=n_needed, replace=False)
                    for idx in idxs:
                        selected_structures.append(candidates[idx])
                else:
                    # Take all, deficit handled later? 
                    # If we take all, we miss (quota - len) structures.
                    # We can pick random from OTHER groups to fill up.
                    selected_structures.extend(candidates)
            
            # Fill deficit if any (from rounding or small groups)
            while len(selected_structures) < num_seeds:
                # Pool of unselected
                # Optimization: just pick random from Valid that are not in Selected (by index/filename?)
                # Or just pick random from Valid (allow duplicate? No avoiding duplicates is better).
                
                # Let's map filename to struct for easy tracking
                selected_filenames = set(s['filename'] for s in selected_structures)
                remaining_pool = [s for s in valid_structures if s['filename'] not in selected_filenames]
                
                if not remaining_pool:
                    break # All available structures selected; padding with None will happen below
                    
                idx = np.random.choice(len(remaining_pool))
                new_struct = remaining_pool[idx]
                selected_structures.append(new_struct)
                selected_filenames.add(new_struct['filename'])

            # Final check (trim if over-selected due to logic bugs, though logic seems sound)
            selected_structures = selected_structures[:num_seeds]
            
            # Format Results
            results = []
            spg_counts = {}
            for s in selected_structures:
                results.append((s['coords'], s['lattice'], s['atoms'], s['edge_index']))
                spg = s['spg']
                spg_counts[spg] = spg_counts.get(spg, 0) + 1
            
            print(f"Selected {len(results)} structures. Distribution: {spg_counts}")
        
        # Pad with None if Genarris generated fewer structures than requested
        # These will be counted as failed seeds in the evaluation
        while len(results) < num_seeds:
            results.append((None, None, None, None))
            
        return results



    def _get_repeating_unit(self, smiles: str, z: int) -> str:
        """
        Extracts the smallest repeating unit from a crystal SMILES string given Z.
        """
        if not smiles:
            return ""
            
        fragments = smiles.split('.')
        counts = Counter(fragments)
        
        unit_parts = []
        # Maintain order of first appearance
        seen = set()
        order = []
        for f in fragments:
            if f not in seen:
                order.append(f)
                seen.add(f)
                
        for frag in order:
            count = counts[frag]
            
            # Calculate how many of this fragment should be in the repeating unit
            # Ideally count % z == 0
            if z > 0:
                unit_count = count // z
            else:
                unit_count = count # Should not happen if Z safe
            
            # If Z is larger than count, we might have a problem (Z > num_mols), unless Z refers to asymmetric unit?
            # But assume unit_count >= 1 for normal crystals. 
            # If unit_count == 0, it means this fragment appears fewer times than Z.
            # This implies the provided SMILES might not be the full unit cell content or Z is defined differently.
            # Taking max(1, unit_count) is a safe fallback to ensure we have at least one.
            unit_count = max(1, unit_count)

            unit_parts.extend([frag] * unit_count)
            
        return ".".join(unit_parts)

    def _create_config(self, path: Path, mol_path: Path, z: int, num_structs: int):
        # Determine tasks
        tasks = ['generation']
        if "rigid_press" in self.mode:
            tasks.append("symm_rigid_press")
            
        # Write config
        # We need to make sure we generate enough structures.
        # Genarris generates per space group. 
        # We usually want `num_structs` total.
        # We'll set a high enough number per spg or pick specific Space Groups?
        # Default Genarris behavior picks random SPGs.
        
        config_content = f"""
[master]
name = genarris_run
molecule_path = ["{str(mol_path)}"]
Z = {z}
log_level = info

[workflow]
tasks = {str(tasks)}

[generation]
num_structures_per_spg = {max(10, num_structs)} # Ensure at least 10, but typically match num_seeds (e.g. 50)
sr = 0.85
max_attempts_per_spg = 100000
tol = 0.1
unit_cell_volume_mean = predict
volume_mult = 1.0
max_attempts_per_volume = 1000
spg_distribution_type = standard
generation_type = crystal
natural_cutoff_mult = 1.0

[symm_rigid_press]
sr = 0.85
method = BFGS
tol = 0.01
natural_cutoff_mult = 1.0
debug_flag = False
maxiter = 100
"""
        with open(path, "w") as f:
            f.write(config_content)
