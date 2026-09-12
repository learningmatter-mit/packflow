#!/usr/bin/env python3

"""
Crystal Processing Utilities

This module contains crystallographic processing functions extracted from data_utils_reference.py
to serve as a lightweight dependency for final_crystal_processor.py.

Refactored to:
- Enforce consistent Pymatgen lattice conventions (c || z).
- Remove redundant Niggli reductions (pure structure building).
- Remove unused rigid-body, MOF, optimization, visualization, and conformer generation code.
- Focus strictly on graph construction and molecule unwrapping.
"""

import torch
import numpy as np
import math
from collections import deque

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Geometry

from pymatgen.core import Structure, Lattice


# Feature lists for molecular property encoding
bond_features_list = {
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "bond_stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "is_conjugated": [False, True],
}

atom_features_list = {
    "atomic_num": list(range(1, 75)) + ["misc"],
    "chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "numring": [0, 1, 2, "misc"],
    "implicit_valence": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "numH": [0, 1, 2, 3, 4, "misc"],
    "number_radical_e": [0, 1, 2, 3, 4, "misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "is_aromatic": [False, True],
    "is_in_ring3": [False, True],
    "is_in_ring4": [False, True],
    "is_in_ring5": [False, True],
    "is_in_ring6": [False, True],
    "is_in_ring7": [False, True],
    "is_in_ring8": [False, True],
}


def safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index"""
    try:
        return l.index(e)
    except:
        return len(l) - 1


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """
    Converts lattice from abc, angles to matrix.
    
    CRITICAL: Uses Pymatgen's default convention (c aligned with z)
    to ensure consistency with the rest of the pipeline.
    """
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma).matrix


def frac_to_cart(frac_coords, lattice_matrix):
    """
    Convert fractional coordinates to cartesian coordinates.
    
    Args:
        frac_coords: Fractional coordinates as numpy array or torch tensor, shape (N, 3)
        lattice_matrix: Lattice matrix (3x3) as numpy array or torch tensor
                      (rows are basis vectors, matching Pymatgen convention)
        
    Returns:
        cart_coords: Cartesian coordinates in same format as input
    """
    # Handle torch tensors
    if torch.is_tensor(frac_coords):
        if torch.is_tensor(lattice_matrix):
            return frac_coords @ lattice_matrix
        else:
            return frac_coords @ torch.from_numpy(lattice_matrix).to(frac_coords.device)
    else:
        # numpy arrays
        if torch.is_tensor(lattice_matrix):
            return np.array(frac_coords) @ lattice_matrix.cpu().numpy()
        else:
            return np.array(frac_coords) @ np.array(lattice_matrix)


def cart_to_frac(cart_coords, lattice_matrix):
    """
    Convert cartesian coordinates to fractional coordinates.
    
    Args:
        cart_coords: Cartesian coordinates as numpy array or torch tensor, shape (N, 3)
        lattice_matrix: Lattice matrix (3x3) as numpy array or torch tensor
                      (rows are basis vectors, matching Pymatgen convention)
        
    Returns:
        frac_coords: Fractional coordinates in same format as input
    """
    # Handle torch tensors
    if torch.is_tensor(cart_coords):
        if torch.is_tensor(lattice_matrix):
            inv_lattice = torch.linalg.inv(lattice_matrix)
            return cart_coords @ inv_lattice
        else:
            inv_lattice = torch.linalg.inv(torch.from_numpy(lattice_matrix).to(cart_coords.device))
            return cart_coords @ inv_lattice
    else:
        # numpy arrays
        if torch.is_tensor(lattice_matrix):
            inv_lattice = np.linalg.inv(lattice_matrix.cpu().numpy())
        else:
            inv_lattice = np.linalg.inv(np.array(lattice_matrix))
        return np.array(cart_coords) @ inv_lattice


def frac_to_cart_batched(frac_coords, lattice_matrices):
    """
    Convert fractional coordinates to cartesian coordinates (batched version).
    
    Args:
        frac_coords: Fractional coordinates as torch tensor, shape (N, 3)
        lattice_matrices: Lattice matrices as torch tensor, shape (N, 3, 3)
                         (rows are basis vectors, matching Pymatgen convention)
                         One lattice matrix per coordinate.
        
    Returns:
        cart_coords: Cartesian coordinates, shape (N, 3)
    """
    if not torch.is_tensor(frac_coords) or not torch.is_tensor(lattice_matrices):
        raise ValueError("Batched version only supports torch tensors")
    
    # Use einsum for batched matrix-vector multiplication: [N, 3] @ [N, 3, 3] -> [N, 3]
    return torch.einsum('ni,nij->nj', frac_coords, lattice_matrices)


def cart_to_frac_batched(cart_coords, lattice_matrices):
    """
    Convert cartesian coordinates to fractional coordinates (batched version).
    
    Args:
        cart_coords: Cartesian coordinates as torch tensor, shape (N, 3)
        lattice_matrices: Lattice matrices as torch tensor, shape (N, 3, 3)
                        (rows are basis vectors, matching Pymatgen convention)
                        One lattice matrix per coordinate.
        
    Returns:
        frac_coords: Fractional coordinates, shape (N, 3)
    """
    if not torch.is_tensor(cart_coords) or not torch.is_tensor(lattice_matrices):
        raise ValueError("Batched version only supports torch tensors")
    
    # Compute inverse lattice matrices: [N, 3, 3]
    inv_lattice_matrices = torch.linalg.inv(lattice_matrices)
    
    # Use einsum for batched matrix-vector multiplication: [N, 3] @ [N, 3, 3] -> [N, 3]
    return torch.einsum('ni,nij->nj', cart_coords, inv_lattice_matrices)


def lattice_matrix_to_params(lattice_matrix):
    """
    Convert lattice matrix to lattice parameters [a, b, c, alpha, beta, gamma].
    Inverse of lattice_params_to_matrix.
    
    Args:
        lattice_matrix: Lattice matrix (3x3) as numpy array, where rows are basis vectors
                       (Pymatgen convention)
    Returns:
        params: numpy array [a, b, c, alpha, beta, gamma] where angles are in degrees
    """
    lattice_matrix = np.array(lattice_matrix)
    if lattice_matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {lattice_matrix.shape}")
    
    # Extract basis vectors (rows of matrix in Pymatgen convention)
    a_vec = lattice_matrix[0, :]
    b_vec = lattice_matrix[1, :]
    c_vec = lattice_matrix[2, :]
    
    # Compute lengths
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    
    # Compute angles
    def _cos(u, v):
        return np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
    
    cos_alpha = _cos(b_vec, c_vec)
    cos_beta = _cos(a_vec, c_vec)
    cos_gamma = _cos(a_vec, b_vec)
    
    alpha = np.degrees(np.arccos(cos_alpha))
    beta = np.degrees(np.arccos(cos_beta))
    gamma = np.degrees(np.arccos(cos_gamma))
    
    return np.array([a, b, c, alpha, beta, gamma], dtype=np.float32)


def lattice_params_to_matrix_torch(lattice_params_deg: torch.Tensor) -> torch.Tensor:
    """
    Convert lattice parameters to lattice matrix (torch version).
    Uses Pymatgen convention where rows are basis vectors.
    
    This matches Pymatgen's Lattice.from_parameters() implementation exactly.
    
    Args:
        lattice_params_deg: [B, 6] or [6] tensor with [a, b, c, alpha°, beta°, gamma°]
    Returns:
        H: [B, 3, 3] or [3, 3] tensor where rows are basis vectors (Pymatgen convention)
    """
    # Handle both batched and unbatched
    was_unbatched = lattice_params_deg.dim() == 1
    if was_unbatched:
        lattice_params_deg = lattice_params_deg.unsqueeze(0)
    
    a, b, c, alpha, beta, gamma = [lattice_params_deg[:, i] for i in range(6)]
    d2r = math.pi / 180.0
    alpha_rad = alpha * d2r
    beta_rad = beta * d2r
    gamma_rad = gamma * d2r
    
    cos_alpha = torch.cos(alpha_rad)
    cos_beta = torch.cos(beta_rad)
    cos_gamma = torch.cos(gamma_rad)
    sin_alpha = torch.sin(alpha_rad)
    sin_beta = torch.sin(beta_rad)
    sin_gamma = torch.sin(gamma_rad)
    
    # Pymatgen's convention (non-vesta mode)
    # Compute gamma_star
    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = torch.clamp(val, -1.0, 1.0)  # Handle rounding errors
    gamma_star = torch.arccos(val)
    
    # Build matrix with rows as basis vectors (Pymatgen convention)
    # vector_a = [a * sin_beta, 0.0, a * cos_beta]
    # vector_b = [-b * sin_alpha * cos(gamma_star), b * sin_alpha * sin(gamma_star), b * cos_alpha]
    # vector_c = [0.0, 0.0, c]
    B = lattice_params_deg.size(0)
    H = torch.zeros((B, 3, 3), device=lattice_params_deg.device, dtype=lattice_params_deg.dtype)
    
    # Row 0: vector_a
    H[:, 0, 0] = a * sin_beta
    H[:, 0, 1] = 0.0
    H[:, 0, 2] = a * cos_beta
    
    # Row 1: vector_b
    H[:, 1, 0] = -b * sin_alpha * torch.cos(gamma_star)
    H[:, 1, 1] = b * sin_alpha * torch.sin(gamma_star)
    H[:, 1, 2] = b * cos_alpha
    
    # Row 2: vector_c
    H[:, 2, 0] = 0.0
    H[:, 2, 1] = 0.0
    H[:, 2, 2] = c
    
    if was_unbatched:
        H = H.squeeze(0)
    
    return H


def lattice_matrix_to_params_torch(H: torch.Tensor) -> torch.Tensor:
    """
    Convert lattice matrix to lattice parameters (torch version).
    Inverse of lattice_params_to_matrix_torch.
    
    Args:
        H: [B, 3, 3] or [3, 3] tensor where rows are basis vectors (Pymatgen convention)
    Returns:
        params: [B, 6] or [6] tensor with [a, b, c, alpha°, beta°, gamma°]
    """
    # Handle both batched and unbatched
    was_unbatched = H.dim() == 2
    if was_unbatched:
        H = H.unsqueeze(0)
    
    # Extract basis vectors (rows of matrix in Pymatgen convention)
    a_vec = H[:, 0, :]  # First row
    b_vec = H[:, 1, :]  # Second row
    c_vec = H[:, 2, :]  # Third row
    
    def _norm(v):
        return torch.linalg.norm(v, dim=-1).clamp_min(1e-12)
    
    a = _norm(a_vec)
    b = _norm(b_vec)
    c = _norm(c_vec)
    
    def _cos(u, v):
        return (torch.sum(u * v, dim=-1) / (_norm(u) * _norm(v))).clamp(-1.0, 1.0)
    
    cos_alpha = _cos(b_vec, c_vec)
    cos_beta = _cos(a_vec, c_vec)
    cos_gamma = _cos(a_vec, b_vec)
    
    d2r = math.pi / 180.0
    alpha = torch.arccos(cos_alpha) / d2r
    beta = torch.arccos(cos_beta) / d2r
    gamma = torch.arccos(cos_gamma) / d2r
    
    params = torch.stack([a, b, c, alpha, beta, gamma], dim=-1)
    
    if was_unbatched:
        params = params.squeeze(0)
    
    return params


def build_crystal_from_dict(entry):
    """
    Given a dictionary entry with keys "cell", "atoms", build a pymatgen Structure.
    
    CRITICAL: This function is a Pure Builder. 
    It does NOT perform Niggli reduction or rotation. 
    """
    # 1) Lattice
    cell = entry["cell"]
    lattice = Lattice.from_parameters(
        cell["a"], cell["b"], cell["c"], 
        cell["alpha"], cell["beta"], cell["gamma"]
    )

    # 2) Fractional coords
    atoms = entry["atoms"]
    frac_coords = [[atom["fract_x"], atom["fract_y"], atom["fract_z"]] for atom in atoms]
    species = [atom['element'] for atom in atoms]

    # 3) Build structure
    crystal = Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        to_unit_cell=True,
        coords_are_cartesian=False,
    )

    return crystal


def best_image_shift(base_frac, trial_frac):
    """
    For a pair of fractional coordinates base_frac and trial_frac,
    find the integer shift in { -1, 0, 1 }^3 that brings them closest.
    """
    delta = trial_frac - base_frac
    return -np.round(delta)


def find_connected_components(num_atoms, bonds):
    """
    Return list of connected components (molecule indices) from bonds.
    """
    adjacency = [[] for _ in range(num_atoms)]
    for b in bonds:
        i = b["atom1_idx"]
        j = b["atom2_idx"]
        adjacency[i].append(j)
        adjacency[j].append(i)

    visited = [False] * num_atoms
    components = []

    for start_atom in range(num_atoms):
        if not visited[start_atom]:
            queue = deque([start_atom])
            visited[start_atom] = True
            component = [start_atom]
            while queue:
                current = queue.popleft()
                for neigh in adjacency[current]:
                    if not visited[neigh]:
                        visited[neigh] = True
                        queue.append(neigh)
                        component.append(neigh)
            components.append(component)

    return components


def make_molecules_whole(frac_coords, bonds):
    """
    Make molecules whole by shifting atoms to be near their bonded neighbors
    using fractional coordinates and PBC.
    """
    # Convert to numpy if tensor
    if torch.is_tensor(frac_coords):
        frac_coords = frac_coords.detach().cpu().numpy()
    
    num_atoms = len(frac_coords)
    molecule_lists = find_connected_components(num_atoms, bonds)
    
    # Build adjacency list
    adjacency = [[] for _ in range(num_atoms)]
    for b in bonds:
        i, j = b['atom1_idx'], b['atom2_idx']
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    unwrapped_coords = frac_coords.copy()
    
    for mol_indices in molecule_lists:
        if len(mol_indices) == 1:
            continue
            
        mol_coords = frac_coords[mol_indices].copy()
        
        # BFS from first atom to make whole
        visited = {0}
        queue = deque([0])
        
        while queue:
            current = queue.popleft()
            current_coord = mol_coords[current]
            
            for global_neigh in adjacency[mol_indices[current]]:
                if global_neigh in mol_indices:
                    local_neigh = mol_indices.index(global_neigh)
                    if local_neigh not in visited:
                        shift = best_image_shift(current_coord, mol_coords[local_neigh])
                        mol_coords[local_neigh] += shift
                        visited.add(local_neigh)
                        queue.append(local_neigh)
        
        # Recenter around geometric center to minimize drift
        center = mol_coords.mean(axis=0)
        distances = np.linalg.norm(mol_coords - center, axis=1)
        central_atom_idx = np.argmin(distances)
        
        # Second pass BFS from central atom
        mol_coords = frac_coords[mol_indices].copy() # reset
        visited = {central_atom_idx}
        queue = deque([central_atom_idx])
        
        while queue:
            current = queue.popleft()
            current_coord = mol_coords[current]
            for global_neigh in adjacency[mol_indices[current]]:
                if global_neigh in mol_indices:
                    local_neigh = mol_indices.index(global_neigh)
                    if local_neigh not in visited:
                        shift = best_image_shift(current_coord, mol_coords[local_neigh])
                        mol_coords[local_neigh] += shift
                        visited.add(local_neigh)
                        queue.append(local_neigh)
                        
        unwrapped_coords[mol_indices] = mol_coords
        
    return unwrapped_coords


def process_crystal(entry, cartesian=False):
    """
    Unwrap molecules for a single crystal entry.
    """
    cell = entry["cell"]
    latt = lattice_params_to_matrix(cell["a"], cell["b"], cell["c"], 
                                   cell["alpha"], cell["beta"], cell["gamma"])

    atoms = entry["atoms"]
    frac_coords = np.array([[a["fract_x"], a["fract_y"], a["fract_z"]] for a in atoms], dtype=float)
    bonds = entry["bonds"]

    unwrapped_frac = make_molecules_whole(frac_coords, bonds)

    if cartesian:
        return unwrapped_frac @ latt
    else:
        return unwrapped_frac


def update_datapoint_info(datapoint, crystal):
    """Update datapoint with unwrapped cartesian coordinates."""
    atoms = datapoint["atoms"]
    unwrapped_frac = process_crystal(datapoint, cartesian=False)
    # crystal.lattice.matrix is consistent (c || z)
    unwrapped_cart = unwrapped_frac @ crystal.lattice.matrix
    
    for i, atom in enumerate(atoms):
        atom["cart_x_unwrapped"] = unwrapped_cart[i, 0]
        atom["cart_y_unwrapped"] = unwrapped_cart[i, 1]
        atom["cart_z_unwrapped"] = unwrapped_cart[i, 2]
    return datapoint


def error_dict(datapoint, error_msg):
    return {
        'refcode': datapoint['refcode'],
        'graph_arrays': None,
        'molecule_ids': None,
        'smiles': datapoint['smiles'],
        'error': error_msg
    }


def featurize_atoms(mol):
    """Extract atom features from RDKit molecule."""
    atom_features = []
    ringinfo = mol.GetRingInfo()

    def safe_index_(key, val):
        return safe_index(atom_features_list[key], val)

    for atom in mol.GetAtoms():
        atom.UpdatePropertyCache()

    for idx, atom in enumerate(mol.GetAtoms()):
        features = [
            safe_index_("atomic_num", atom.GetAtomicNum()),
            safe_index_("degree", atom.GetTotalDegree()),
            safe_index_("numring", ringinfo.NumAtomRings(idx)),
            safe_index_("implicit_valence", atom.GetImplicitValence()),
            safe_index_("formal_charge", atom.GetFormalCharge()),
            safe_index_("numH", atom.GetTotalNumHs()),
            safe_index_("hybridization", str(atom.GetHybridization())),
            safe_index_("is_aromatic", atom.GetIsAromatic()),
            safe_index_("is_in_ring5", ringinfo.IsAtomInRingOfSize(idx, 5)),
            safe_index_("is_in_ring6", ringinfo.IsAtomInRingOfSize(idx, 6)),
        ]
        atom_features.append(features)

    return torch.tensor(atom_features)


def featurize_bond(bond):
    """Extract bond features from RDKit bond."""
    bond_feature = [
        safe_index(bond_features_list["bond_type"], str(bond.GetBondType())),
        safe_index(bond_features_list["is_conjugated"], bond.GetIsConjugated()),
    ]
    return bond_feature


def get_bond_edges(mol):
    """Extract edge indices and features from RDKit molecule bonds."""
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_attr += [featurize_bond(bond)]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr)
    edge_attr = torch.concat([edge_attr, edge_attr], 0)
    return edge_index, edge_attr.type(torch.uint8)


_BOND_TYPE_MAP = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "AROMATIC": Chem.BondType.AROMATIC,
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def _to_bond_type(value):
    """Coerce a user-supplied bond type into an RDKit ``BondType``."""
    if isinstance(value, Chem.BondType):
        return value
    if isinstance(value, str):
        return _BOND_TYPE_MAP[value.strip().upper()]
    return _BOND_TYPE_MAP[int(value)]


def build_mol_from_graph(elements, bond_index, bond_types=None, smiles=None):
    """Build a (coordinate-free) RDKit molecule from a molecular graph.

    This is the entry point for running inference on an arbitrary molecule for
    which you only know the atoms and their bonds (no 3D coordinates needed).
    The resulting molecule is sanitised so it carries exactly the same RDKit
    features the model was trained on (hybridisation, ring membership, implicit
    hydrogen counts, etc.).

    Args:
        elements: per-atom elements, as symbols (``["C", "C", "O", ...]``) or
            atomic numbers (``[6, 6, 8, ...]``).
        bond_index: bonds as a ``[2, E]`` array/tensor or a list of ``(i, j)``
            pairs. Direction is ignored and duplicate (i, j)/(j, i) edges are
            collapsed to a single bond.
        bond_types: optional per-bond order, one entry per *unique* bond in the
            order bonds first appear in ``bond_index``. Accepts ``"SINGLE"``/
            ``"DOUBLE"``/``"TRIPLE"``/``"AROMATIC"``, integers (1/2/3), or RDKit
            ``BondType`` values. Defaults to single bonds.
        smiles: optional SMILES for the same molecule; when given, bond orders
            are assigned from it (handy for aromatic systems where single-bond
            skeletons fail to sanitise).

    Returns:
        A sanitised RDKit ``Mol`` whose atom order matches ``elements``.
    """
    import numpy as _np

    pt = Chem.GetPeriodicTable()
    mol = Chem.RWMol()
    for el in elements:
        z = el if isinstance(el, (int, _np.integer)) else pt.GetAtomicNumber(str(el))
        mol.AddAtom(Chem.Atom(int(z)))

    if torch.is_tensor(bond_index):
        bond_index = bond_index.cpu().numpy()
    bond_index = _np.asarray(bond_index)
    if bond_index.ndim == 2 and bond_index.shape[0] == 2:
        pairs = list(zip(bond_index[0].tolist(), bond_index[1].tolist()))
    else:  # list of (i, j) pairs
        pairs = [tuple(p) for p in bond_index]

    seen = set()
    bt_idx = 0
    for i, j in pairs:
        key = tuple(sorted((int(i), int(j))))
        if key in seen:
            continue
        seen.add(key)
        bt = _to_bond_type(bond_types[bt_idx]) if bond_types is not None else Chem.BondType.SINGLE
        mol.AddBond(int(i), int(j), bt)
        bt_idx += 1

    mol = mol.GetMol()
    if smiles is not None:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol)

    Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms():
        atom.UpdatePropertyCache()
    return mol


def make_rdkit_mol_from_dict(datapoint, smiles, RemoveHs=True):
    """Build an RDKit molecule from a datapoint dictionary."""
    mol = Chem.RWMol()
    old_to_new = {}
    new_idx = 0

    # Add atoms
    for i, atom in enumerate(datapoint["atoms"]):
        element = atom["element"]
        if RemoveHs and element == "H":
            continue

        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
        rdatom = Chem.Atom(atomic_num)
        mol.AddAtom(rdatom)
        old_to_new[i] = new_idx
        new_idx += 1

    unwrapped_cart_coords = []
    conformer = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conformer, assignId=True)

    # Set coords
    for i, atom in enumerate(datapoint["atoms"]):
        element = atom["element"]
        if RemoveHs and element == "H":
            continue
        x, y, z = atom["cart_x_unwrapped"], atom["cart_y_unwrapped"], atom["cart_z_unwrapped"]
        unwrapped_cart_coords.append([x, y, z])
        mol.GetConformer(0).SetAtomPosition(old_to_new[i], (x, y, z))

    # Add bonds
    for bond in datapoint["bonds"]:
        a1, a2 = bond["atom1_idx"], bond["atom2_idx"]
        if RemoveHs:
            if (datapoint["atoms"][a1]["element"] == "H" or datapoint["atoms"][a2]["element"] == "H"):
                continue
        mol.AddBond(old_to_new[a1], old_to_new[a2], Chem.rdchem.BondType.SINGLE)

    # Assign Bond Orders
    template = Chem.MolFromSmiles(smiles)
    mol = AllChem.AssignBondOrdersFromTemplate(template, mol)

    if RemoveHs:
        mol = Chem.RemoveHs(mol)
    
    Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms():
        atom.UpdatePropertyCache()

    return mol, torch.tensor(unwrapped_cart_coords)


def _identify_hydrogen_bonding_atoms(datapoint):
    """Identify hydrogen bond donors and acceptors before hydrogen removal."""
    atoms = datapoint["atoms"]
    bonds = datapoint["bonds"]
    num_atoms = len(atoms)
    
    atomic_numbers = []
    for atom in atoms:
        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(atom["element"])
        atomic_numbers.append(atomic_num)
    atomic_numbers = np.array(atomic_numbers)
    
    adjacency = [[] for _ in range(num_atoms)]
    for bond in bonds:
        adjacency[bond["atom1_idx"]].append(bond["atom2_idx"])
        adjacency[bond["atom2_idx"]].append(bond["atom1_idx"])
    
    # Acceptors: N(7), O(8), F(9)
    acceptor_mask = np.isin(atomic_numbers, [7, 8, 9])
    
    # Donors: N, O, F connected to H(1)
    donor_mask = np.zeros(num_atoms, dtype=bool)
    for i in range(num_atoms):
        if acceptor_mask[i]:
            for neighbor_idx in adjacency[i]:
                if atomic_numbers[neighbor_idx] == 1:
                    donor_mask[i] = True
                    break
    return donor_mask, acceptor_mask


def _identify_aromatic_rings(mol, frac_coords, molecule_ids, lengths, angles, edge_index):
    """Identify aromatic rings and compute ring data."""
    try:
        Chem.SetAromaticity(mol)
    except:
        pass
    
    ring_info = mol.GetRingInfo()
    if ring_info is None:
        return torch.empty(0,2,dtype=torch.long), torch.empty(0,3), torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.long), 0

    aromatic_rings = []
    for ring in ring_info.AtomRings():
        if len(ring) < 3: continue
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append(ring)
    
    if len(aromatic_rings) == 0:
        return torch.empty(0,2,dtype=torch.long), torch.empty(0,3), torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.long), 0

    if torch.is_tensor(frac_coords): frac_coords_np = frac_coords.cpu().numpy()
    else: frac_coords_np = np.array(frac_coords)

    # Unwrap logic for rings
    bonds = []
    if torch.is_tensor(edge_index): edge_index_np = edge_index.cpu().numpy()
    else: edge_index_np = np.array(edge_index)

    for i in range(edge_index_np.shape[1]):
        bonds.append({'atom1_idx': int(edge_index_np[0,i]), 'atom2_idx': int(edge_index_np[1,i])})
        
    whole_frac_coords = make_molecules_whole(frac_coords_np, bonds)
    
    # Convert to Cartesian
    latt = lattice_params_to_matrix(lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2])
    whole_cart_coords = whole_frac_coords @ latt
    
    cart_coords_mean = whole_cart_coords.mean(axis=0)
    centered_cart_coords = torch.from_numpy(whole_cart_coords - cart_coords_mean).float()

    # Collect ring data
    ring_atom_pairs_list = []
    ring_centroids_list = []
    ring_molecule_ids_list = []
    ring_sizes_list = []

    for ring_idx, ring in enumerate(aromatic_rings):
        ring_atom_indices = list(ring)
        ring_coords = centered_cart_coords[ring_atom_indices]
        ring_centroid = ring_coords.mean(dim=0)
        
        # Mol ID
        ring_mol_ids = molecule_ids[ring_atom_indices] if torch.is_tensor(molecule_ids) else torch.tensor(molecule_ids)[ring_atom_indices]
        ring_mol_id = torch.mode(ring_mol_ids).values.item()

        for atom_idx in ring_atom_indices:
            ring_atom_pairs_list.append([ring_idx, atom_idx])
            
        ring_centroids_list.append(ring_centroid)
        ring_molecule_ids_list.append(ring_mol_id)
        ring_sizes_list.append(len(ring_atom_indices))

    return (torch.tensor(ring_atom_pairs_list, dtype=torch.long), 
            torch.stack(ring_centroids_list), 
            torch.tensor(ring_molecule_ids_list, dtype=torch.long),
            torch.tensor(ring_sizes_list, dtype=torch.long),
            len(ring_centroids_list))


def build_bonded_crystal_graph_from_dict(crystal, datapoint, smiles, RemoveHs=False, tag_hydrogen_bonding=False, find_aromatic_rings=False):
    """Build crystal graph arrays."""
    frac_coords = crystal.frac_coords
    atom_types = torch.tensor(crystal.atomic_numbers)
    
    # H-Bonding tagging
    donor_mask_full, acceptor_mask_full = None, None
    if tag_hydrogen_bonding and RemoveHs:
        donor_mask_full, acceptor_mask_full = _identify_hydrogen_bonding_atoms(datapoint)

    if RemoveHs:
        heavy_atoms_mask = (atom_types != 1)
        frac_coords = frac_coords[heavy_atoms_mask]
        atom_types = atom_types[heavy_atoms_mask]
        
        if tag_hydrogen_bonding and donor_mask_full is not None:
            donor_mask = donor_mask_full[heavy_atoms_mask.numpy()]
            acceptor_mask = acceptor_mask_full[heavy_atoms_mask.numpy()]
        else:
            donor_mask, acceptor_mask = None, None
    else:
        donor_mask, acceptor_mask = None, None

    lattice_parameters = crystal.lattice.parameters
    lengths, angles = lattice_parameters[:3], lattice_parameters[3:]
    
    try:
        mol, unwrapped_cart_coords = make_rdkit_mol_from_dict(datapoint, smiles, RemoveHs=RemoveHs)
    except:
        return None

    atom_features = featurize_atoms(mol)
    edge_index, edge_attr = get_bond_edges(mol)
    num_atoms = len(atom_types)

    # Aromatic Rings
    ring_data = (None, None, None, None, 0)
    if find_aromatic_rings:
        molecule_ids, _ = get_molecule_ids(edge_index, num_atoms)
        ring_data = _identify_aromatic_rings(mol, frac_coords, molecule_ids, lengths, angles, edge_index)

    return (frac_coords, unwrapped_cart_coords, atom_types.numpy(), np.array(lengths), np.array(angles), 
            atom_features, edge_index, edge_attr, num_atoms, donor_mask, acceptor_mask, *ring_data)


def validate_crystal(frac_coords, lengths, angles):
    lattice = Lattice.from_parameters(*lengths, *angles)
    if lattice.volume < 5 or lattice.volume > 1e6: return False
    return True


def get_molecule_ids(edge_index, num_atoms):
    """Get molecule IDs from edge connectivity."""
    bonds = []
    if torch.is_tensor(edge_index): edge_index = edge_index.cpu().numpy()
    
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] < edge_index[1, i]:
            bonds.append({'atom1_idx': int(edge_index[0, i]), 'atom2_idx': int(edge_index[1, i])})
            
    molecule_lists = find_connected_components(num_atoms, bonds)
    molecule_lists = [sorted(l) for l in molecule_lists]
    
    molecule_ids = torch.zeros(num_atoms, dtype=torch.long)
    for mol_idx, atom_indices in enumerate(molecule_lists):
        molecule_ids[atom_indices] = mol_idx
        
    return molecule_ids, molecule_lists


def get_crystal_info(datapoint, RemoveHs=False, tag_hydrogen_bonding=False, find_aromatic_rings=False):
    """Pipeline: Build Crystal -> Update Info -> Build Graph"""
    crystal = build_crystal_from_dict(datapoint)
    datapoint = update_datapoint_info(datapoint, crystal)
    
    graph_arrays = build_bonded_crystal_graph_from_dict(
        crystal, datapoint, datapoint['smiles'], RemoveHs=RemoveHs,
        tag_hydrogen_bonding=tag_hydrogen_bonding, find_aromatic_rings=find_aromatic_rings
    )
    
    if graph_arrays is None:
        return error_dict(datapoint, "Couldn't build graph arrays")
        
    if not validate_crystal(graph_arrays[0], graph_arrays[3], graph_arrays[4]):
        return error_dict(datapoint, "Invalid crystal structure")
        
    return crystal, datapoint, graph_arrays


def _assign_bond_orders_try_templates(mol, smiles_string):
    templates = [s for s in smiles_string.split(".") if s.strip()]
    templates.append(smiles_string)
    for smi in templates:
        tpl = Chem.MolFromSmiles(smi)
        if tpl:
            try: return AllChem.AssignBondOrdersFromTemplate(tpl, mol)
            except ValueError: continue
    Chem.SanitizeMol(mol)
    return mol


def _build_rdkit_submol(gt_coords, atom_types, edge_indices, atom_indices, smiles, RemoveHs=False):
    rwmol = Chem.RWMol()
    old_to_new = {}
    new_i = 0
    
    for i, old_i in enumerate(atom_indices):
        try: element = Chem.GetPeriodicTable().GetElementSymbol(int(atom_types[i].item()))
        except: element = 'C'
        
        a = Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(element))
        rwmol.AddAtom(a)
        old_to_new[old_i] = new_i
        new_i += 1
        
    for i in range(edge_indices.shape[1]):
        a1, a2 = edge_indices[0][i].item(), edge_indices[1][i].item()
        if a1 in old_to_new and a2 in old_to_new:
            if rwmol.GetBondBetweenAtoms(old_to_new[a1], old_to_new[a2]) is None:
                rwmol.AddBond(old_to_new[a1], old_to_new[a2], Chem.rdchem.BondType.SINGLE)
                
    mol = rwmol.GetMol()
    mol.RemoveAllConformers()
    
    conf = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conf, assignId=True)
    for i, pos in enumerate(gt_coords):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(*pos))
        
    mol = _assign_bond_orders_try_templates(mol, smiles)
    Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms(): atom.UpdatePropertyCache(strict=False)
    
    return mol, old_to_new


def lattice_matrix_to_k_basis(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 lattice matrix to the O(3)-invariant k-basis representation.
    
    Mathematical formulation:
    1. Compute Metric Tensor: J = M @ M.T
    2. Compute Symmetric Logarithm: S = 0.5 * log(J)
    3. Project S onto basis matrices to get k coefficients.
    """
    # 1. Metric Tensor (Gram Matrix)
    J = matrix @ matrix.T
    
    # 2. Symmetric Logarithm S = 0.5 * log(J)
    # Use eigen-decomposition for stable matrix log of a symmetric positive-definite matrix
    eigvals, eigvecs = np.linalg.eigh(J)
    # Clamp small eigenvalues for numerical stability
    eigvals = np.maximum(eigvals, 1e-9) 
    log_J = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
    S = 0.5 * log_J
    
    # 3. Project onto k basis
    k = np.zeros(6, dtype=np.float32)
    
    # Off-diagonal elements
    k[0] = S[0, 1]  # k1
    k[1] = S[0, 2]  # k2
    k[2] = S[1, 2]  # k3
    
    # Diagonal elements (linear combinations)
    S00, S11, S22 = S[0, 0], S[1, 1], S[2, 2]
    
    k[3] = (S00 - S11) / 2.0                # k4
    k[4] = (S00 + S11 - 2 * S22) / 6.0      # k5
    k[5] = (S00 + S11 + S22) / 3.0          # k6
    
    return k


def k_basis_to_lattice_matrix(k: np.ndarray) -> np.ndarray:
    """
    Convert the O(3)-invariant k-basis representation back to a 3x3 lattice matrix.
    
    This performs the inverse operation:
    1. Reconstruct Symmetric Matrix S from k.
    2. Recover Metric Tensor J = exp(2*S).
    3. Extract lattice parameters from J.
    4. Construct canonical lattice matrix (c || z).
    """
    # 1. Reconstruct Symmetric Matrix S
    # Formulas derived from reversing the projection (See DiffCSP++ Appendix A.2)
    k1, k2, k3, k4, k5, k6 = k
    
    S = np.zeros((3, 3), dtype=np.float32)
    
    # Off-diagonals
    S[0, 1] = S[1, 0] = k1
    S[0, 2] = S[2, 0] = k2
    S[1, 2] = S[2, 1] = k3
    
    # Diagonals
    # Derived from: 
    # k4 = (S00 - S11)/2
    # k5 = (S00 + S11 - 2S22)/6
    # k6 = (S00 + S11 + S22)/3
    S[0, 0] = k6 + k5 + k4
    S[1, 1] = k6 + k5 - k4
    S[2, 2] = k6 - 2 * k5
    
    # 2. Recover Metric Tensor J = exp(2*S)
    # Use eigen-decomposition for numerical stability
    eigvals, eigvecs = np.linalg.eigh(S)
    J = eigvecs @ np.diag(np.exp(2 * eigvals)) @ eigvecs.T
    
    # 3. Extract Lattice Parameters from Metric Tensor J (Gram Matrix)
    # J_ij = dot(vec_i, vec_j)
    a = np.sqrt(max(J[0, 0], 1e-6))
    b = np.sqrt(max(J[1, 1], 1e-6))
    c = np.sqrt(max(J[2, 2], 1e-6))
    
    # Clip cosines to [-1, 1] to handle numerical noise
    cos_alpha = np.clip(J[1, 2] / (b * c), -1, 1)
    cos_beta  = np.clip(J[0, 2] / (a * c), -1, 1)
    cos_gamma = np.clip(J[0, 1] / (a * b), -1, 1)
    
    alpha = np.degrees(np.arccos(cos_alpha))
    beta  = np.degrees(np.arccos(cos_beta))
    gamma = np.degrees(np.arccos(cos_gamma))
    
    # 4. Construct Canonical Lattice Matrix
    return lattice_params_to_matrix(a, b, c, alpha, beta, gamma)


def lattice_matrix_to_k_basis_torch(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3x3 lattice matrix to the O(3)-invariant k-basis representation (torch version).
    
    Mathematical formulation:
    1. Compute Metric Tensor: J = M @ M.T
    2. Compute Symmetric Logarithm: S = 0.5 * log(J)
    3. Project S onto basis matrices to get k coefficients.
    
    Args:
        matrix: [B, 3, 3] or [3, 3] tensor with rows as basis vectors (Pymatgen convention)
    Returns:
        k: [B, 6] or [6] tensor with k_basis representation
    """
    # Handle both batched and unbatched
    was_unbatched = matrix.dim() == 2
    if was_unbatched:
        matrix = matrix.unsqueeze(0)
    
    B = matrix.size(0)
    device = matrix.device
    dtype = matrix.dtype
    
    # 1. Metric Tensor (Gram Matrix)
    J = torch.bmm(matrix, matrix.transpose(1, 2))  # [B, 3, 3]
    
    # 2. Symmetric Logarithm S = 0.5 * log(J)
    # Use eigen-decomposition for stable matrix log of a symmetric positive-definite matrix
    eigvals, eigvecs = torch.linalg.eigh(J)  # [B, 3], [B, 3, 3]
    # Clamp small eigenvalues for numerical stability
    eigvals = torch.clamp(eigvals, min=1e-9)
    log_eigvals = torch.log(eigvals)  # [B, 3]
    
    # Compute log_J = eigvecs @ diag(log_eigvals) @ eigvecs.T
    log_J = torch.bmm(
        torch.bmm(eigvecs, torch.diag_embed(log_eigvals)),
        eigvecs.transpose(1, 2)
    )
    S = 0.5 * log_J  # [B, 3, 3]
    
    # 3. Project onto k basis
    k = torch.zeros((B, 6), device=device, dtype=dtype)
    
    # Off-diagonal elements
    k[:, 0] = S[:, 0, 1]  # k1
    k[:, 1] = S[:, 0, 2]  # k2
    k[:, 2] = S[:, 1, 2]  # k3
    
    # Diagonal elements (linear combinations)
    S00, S11, S22 = S[:, 0, 0], S[:, 1, 1], S[:, 2, 2]
    
    k[:, 3] = (S00 - S11) / 2.0                # k4
    k[:, 4] = (S00 + S11 - 2 * S22) / 6.0      # k5
    k[:, 5] = (S00 + S11 + S22) / 3.0          # k6
    
    if was_unbatched:
        k = k.squeeze(0)
    
    return k


def k_basis_to_lattice_matrix_torch(k: torch.Tensor) -> torch.Tensor:
    """
    Convert the O(3)-invariant k-basis representation back to a 3x3 lattice matrix (torch version).
    
    This performs the inverse operation:
    1. Reconstruct Symmetric Matrix S from k.
    2. Recover Metric Tensor J = exp(2*S).
    3. Extract lattice parameters from J.
    4. Construct canonical lattice matrix (c || z).
    
    Args:
        k: [B, 6] or [6] tensor with k_basis representation
    Returns:
        matrix: [B, 3, 3] or [3, 3] tensor with rows as basis vectors (Pymatgen convention)
    """
    # Handle both batched and unbatched
    was_unbatched = k.dim() == 1
    if was_unbatched:
        k = k.unsqueeze(0)
    
    B = k.size(0)
    device = k.device
    dtype = k.dtype
    
    # 1. Reconstruct Symmetric Matrix S
    k1, k2, k3, k4, k5, k6 = k[:, 0], k[:, 1], k[:, 2], k[:, 3], k[:, 4], k[:, 5]
    
    S = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    
    # Off-diagonals
    S[:, 0, 1] = S[:, 1, 0] = k1
    S[:, 0, 2] = S[:, 2, 0] = k2
    S[:, 1, 2] = S[:, 2, 1] = k3
    
    # Diagonals
    # Derived from: 
    # k4 = (S00 - S11)/2
    # k5 = (S00 + S11 - 2S22)/6
    # k6 = (S00 + S11 + S22)/3
    S[:, 0, 0] = k6 + k5 + k4
    S[:, 1, 1] = k6 + k5 - k4
    S[:, 2, 2] = k6 - 2 * k5
    
    # 2. Recover Metric Tensor J = exp(2*S)
    # Use eigen-decomposition for numerical stability
    eigvals, eigvecs = torch.linalg.eigh(S)  # [B, 3], [B, 3, 3]
    exp_2eigvals = torch.exp(2 * eigvals)  # [B, 3]
    J = torch.bmm(
        torch.bmm(eigvecs, torch.diag_embed(exp_2eigvals)),
        eigvecs.transpose(1, 2)
    )  # [B, 3, 3]
    
    # 3. Extract Lattice Parameters from Metric Tensor J (Gram Matrix)
    # J_ij = dot(vec_i, vec_j)
    a = torch.sqrt(torch.clamp(J[:, 0, 0], min=1e-6))
    b = torch.sqrt(torch.clamp(J[:, 1, 1], min=1e-6))
    c = torch.sqrt(torch.clamp(J[:, 2, 2], min=1e-6))
    
    # Clip cosines to [-1, 1] to handle numerical noise
    cos_alpha = torch.clamp(J[:, 1, 2] / (b * c), -1, 1)
    cos_beta  = torch.clamp(J[:, 0, 2] / (a * c), -1, 1)
    cos_gamma = torch.clamp(J[:, 0, 1] / (a * b), -1, 1)
    
    # Convert radians to degrees (consistent with existing code)
    d2r = math.pi / 180.0
    alpha = torch.arccos(cos_alpha) / d2r
    beta  = torch.arccos(cos_beta) / d2r
    gamma = torch.arccos(cos_gamma) / d2r
    
    # Stack parameters [B, 6]
    lattice_params = torch.stack([a, b, c, alpha, beta, gamma], dim=-1)
    
    # 4. Construct Canonical Lattice Matrix using existing function
    matrix = lattice_params_to_matrix_torch(lattice_params)  # [B, 3, 3]
    
    if was_unbatched:
        matrix = matrix.squeeze(0)
    
    return matrix
