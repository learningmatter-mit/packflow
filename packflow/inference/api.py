"""High-level inference API for PackFlow.

Typical use::

    from packflow import load_checkpoint, run_inference

    model = load_checkpoint("packflow-pa", device="cuda")          # or a .pt path
    results = run_inference(model, mmcif="my_crystal.mmcif",
                            n_samples=8, n_steps=100)
    for r in results:
        print(r["refcode"], r["sample_index"], r["lattice_params"])

``run_inference`` takes a *new* molecular crystal (an mmCIF file, or a pre-processed
``.pt`` dataset + optional refcode filter), conditions on its molecular graph /
composition, and samples crystal packings (Cartesian coordinates + lattice).
Optionally it scores each sample with a single-point UMA energy.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

from packflow.checkpoints import list_models, resolve as _resolve_checkpoint  # noqa: F401


def load_checkpoint(name_or_path: str = "packflow-pa", device: str = "cpu"):
    """Load a PackFlow checkpoint into a ``CrystalFlowMatching`` model.

    Args:
        name_or_path: A model-zoo name (``packflow-2M``/``packflow-20M``/
            ``packflow-ddp``/``packflow-60M``/``packflow-pa``) or a path to a ``.pt``.
        device: ``"cpu"`` or ``"cuda"``.
    """
    from packflow.grpo.model_loader import load_pretrained_model

    ckpt = _resolve_checkpoint(name_or_path)
    return load_pretrained_model(ckpt, device=device)


def _dataset_from_mmcif(mmcif_path: str, tmpdir: str):
    """Process a single mmCIF into a CrystalDataset (in-memory)."""
    from packflow.utils.batch_processor import batch_process_crystals
    from packflow.data.crystal_datamodule import CrystalDataset

    out = os.path.join(tmpdir, "inference_inputs.pt")
    batch_process_crystals(
        [mmcif_path],
        out,
        n_jobs=1,
        remove_hydrogens=True,
        skip_rdkit=False,
        tag_hydrogen_bonding=True,
        find_aromatic_rings=True,
    )
    # CrystalDataset loads everything into memory in __init__, so it stays valid
    # after the temporary directory is removed.
    return CrystalDataset(out)


def run_inference(
    model,
    mmcif: Optional[str] = None,
    processed_path: Optional[str] = None,
    refcode: Optional[Union[str, List[str]]] = None,
    n_samples: int = 1,
    n_steps: int = 100,
    temperature: float = 1.0,
    device: Optional[str] = None,
    score_energy: bool = False,
    uma_device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sample crystal packings for one or more molecular crystals.

    Provide exactly one input source:
      - ``mmcif``: path to a single mmCIF file (will be processed on the fly), or
      - ``processed_path``: a pre-processed ``.pt`` dataset (optionally filtered by ``refcode``).

    Args:
        model: a model returned by :func:`load_checkpoint`.
        n_samples: number of independent packings to draw per crystal.
        n_steps: ODE integration steps.
        temperature: low-temperature lambda (1.0 = default; >1.0 = more deterministic).
        score_energy: if True, also compute a single-point UMA energy per sample
            (requires fairchem; see ``packflow.relaxation``).
        uma_device: device for the UMA calculator (defaults to ``device``).

    Returns:
        A list of dicts with keys ``refcode``, ``sample_index``, ``cart_coords``
        ([N,3]), ``lattice_params`` ([6]), ``atom_types`` ([N]), and (if requested)
        ``uma_energy``.
    """
    from packflow.grpo.sampler import prepare_crystal_template, sample_batch

    if (mmcif is None) == (processed_path is None):
        raise ValueError("Provide exactly one of `mmcif` or `processed_path`.")

    device = device or getattr(model, "device", "cpu")

    tmp = None
    try:
        if mmcif is not None:
            tmp = tempfile.TemporaryDirectory()
            dataset = _dataset_from_mmcif(mmcif, tmp.name)
        else:
            from packflow.data.crystal_datamodule import CrystalDataset

            rc = [refcode] if isinstance(refcode, str) else refcode
            dataset = CrystalDataset(processed_path, refcode_filter=rc)

        if len(dataset) == 0:
            raise ValueError("No crystals to sample (empty dataset / refcode filter).")

        records: List[Dict[str, Any]] = []
        for idx in range(len(dataset)):
            data = dataset[idx]
            template = prepare_crystal_template(data, device=device)
            samples = sample_batch(
                model, template, n_steps=n_steps, num_seeds=n_samples, lambda_val=temperature
            )
            rc = getattr(data, "refcode", None)
            atom_types = data.atom_types.detach().cpu()
            sample_records = []
            for s, (coords, lattice) in enumerate(samples):
                sample_records.append(
                    {
                        "refcode": rc,
                        "sample_index": s,
                        "cart_coords": coords.detach().cpu(),
                        "lattice_params": lattice.detach().cpu(),
                        "atom_types": atom_types,
                    }
                )
            if score_energy:
                _attach_uma_energies(sample_records, samples, data.atom_types,
                                     device=uma_device or device)
            records.extend(sample_records)
        return records
    finally:
        if tmp is not None:
            tmp.cleanup()


def build_sampling_template(
    elements,
    bond_index,
    *,
    bond_types=None,
    smiles: Optional[str] = None,
    z: int = 1,
    remove_hydrogens: bool = True,
):
    """Build a sampling template from an arbitrary molecular graph (no coordinates).

    Featurises the molecule exactly like the training pipeline (same RDKit atom/bond
    features) and tiles it ``z`` times so the model packs ``z`` molecules per unit cell.

    Args:
        elements: per-atom element symbols or atomic numbers.
        bond_index: bonds as ``[2, E]`` or a list of ``(i, j)`` pairs.
        bond_types: optional per-bond orders (see ``build_mol_from_graph``).
        smiles: optional SMILES used to assign bond orders.
        z: number of molecules to place in the cell (graph is tiled ``z`` times).
        remove_hydrogens: drop explicit hydrogens to match the training distribution.

    Returns:
        A dict with ``atom_types``, ``node_features``, ``edge_index`` and
        ``bond_features`` tensors, ready for :func:`packflow.grpo.sampler.sample_batch`.
    """
    import torch
    from rdkit import Chem

    from packflow.utils.crystal_utils import (
        build_mol_from_graph,
        featurize_atoms,
        get_bond_edges,
    )

    mol = build_mol_from_graph(elements, bond_index, bond_types=bond_types, smiles=smiles)
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)

    atom_types = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    node_features = featurize_atoms(mol)
    edge_index, bond_features = get_bond_edges(mol)

    if edge_index.numel() == 0:
        raise ValueError("Molecule has no bonds; PackFlow requires a bonded graph.")

    if z > 1:
        n = atom_types.shape[0]
        atom_types = atom_types.repeat(z)
        node_features = node_features.repeat(z, 1)
        bond_features = bond_features.repeat(z, 1)
        edge_index = torch.cat([edge_index + i * n for i in range(z)], dim=1)

    return {
        "atom_types": atom_types,
        "node_features": node_features,
        "edge_index": edge_index,
        "bond_features": bond_features,
    }


def predict(
    elements,
    bond_index,
    *,
    model=None,
    n_samples: int = 1,
    n_steps: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
    bond_types=None,
    smiles: Optional[str] = None,
    z: int = 1,
    remove_hydrogens: bool = True,
    score_energy: bool = False,
    uma_device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sample crystal packings for an arbitrary molecule given only its graph.

    Unlike :func:`run_inference` (which needs a structure file), this conditions
    purely on a molecular graph - the atoms and which atoms are bonded - so you can
    run molecular crystal-structure prediction on any molecule you can describe.

    Example::

        from packflow import load_checkpoint, predict, write_cif

        model = load_checkpoint("packflow-60M", device="cpu")
        # Benzene: 6 carbons in a ring (hydrogens are implicit).
        results = predict(
            elements=["C"] * 6,
            bond_index=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
            smiles="c1ccccc1",
            model=model, n_samples=4,
        )
        write_cif(results[0], "benzene_packing.cif")

    Args:
        elements: per-atom element symbols or atomic numbers.
        bond_index: bonds as ``[2, E]`` or a list of ``(i, j)`` pairs.
        model: a model from :func:`load_checkpoint` (loads ``packflow-pa`` if None).
        n_samples: number of independent packings to draw.
        n_steps: ODE integration steps.
        temperature: low-temperature lambda (1.0 = default; >1.0 = more deterministic).
        bond_types: optional per-bond orders (see ``build_mol_from_graph``).
        smiles: optional SMILES used to assign bond orders.
        z: molecules per unit cell (the graph is tiled ``z`` times).
        remove_hydrogens: drop explicit hydrogens to match training.
        score_energy: also attach a single-point UMA energy per sample.
        uma_device: device for the UMA calculator (defaults to ``device``).

    Returns:
        The same record schema as :func:`run_inference`, so ``to_structure`` and
        ``write_cif`` work directly on each result.
    """
    from packflow.grpo.sampler import sample_batch

    if model is None:
        model = load_checkpoint(device=device)

    template = build_sampling_template(
        elements,
        bond_index,
        bond_types=bond_types,
        smiles=smiles,
        z=z,
        remove_hydrogens=remove_hydrogens,
    )
    template = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in template.items()
    }

    samples = sample_batch(
        model, template, n_steps=n_steps, num_seeds=n_samples, lambda_val=temperature
    )

    atom_types = template["atom_types"].detach().cpu()
    refcode = smiles or "molecule"
    records: List[Dict[str, Any]] = []
    for s, (coords, lattice) in enumerate(samples):
        records.append(
            {
                "refcode": refcode,
                "sample_index": s,
                "cart_coords": coords.detach().cpu(),
                "lattice_params": lattice.detach().cpu(),
                "atom_types": atom_types,
            }
        )
    if score_energy:
        _attach_uma_energies(records, samples, template["atom_types"],
                             device=uma_device or device)
    return records


def _attach_uma_energies(sample_records, samples, atom_types, device="cpu"):
    """Compute single-point UMA energies and attach them to records (best-effort)."""
    from packflow.relaxation.uma import (
        UMAEnergyCalculator,
        samples_to_crystal_list,
    )

    calc = UMAEnergyCalculator(device=device)
    crystals = samples_to_crystal_list(samples, atom_types)
    energies = calc.compute_energies_batch(crystals)
    for rec, energy in zip(sample_records, energies):
        rec["uma_energy"] = float(energy)


def to_structure(record: Dict[str, Any]):
    """Convert an inference record into a pymatgen ``Structure``.

    Uses the sampled lattice parameters [a, b, c, alpha, beta, gamma] and Cartesian
    coordinates. Atomic numbers come from ``atom_types``.
    """
    import numpy as np
    from pymatgen.core import Lattice, Structure

    lp = record["lattice_params"].numpy().tolist()
    if len(lp) == 6:
        lattice = Lattice.from_parameters(*lp)
    else:  # 3x3 matrix flattened
        lattice = Lattice(np.array(lp).reshape(3, 3))
    coords = record["cart_coords"].numpy()
    species = [int(z) for z in record["atom_types"].numpy()]
    return Structure(lattice, species, coords, coords_are_cartesian=True)


def write_cif(record: Dict[str, Any], path: str) -> str:
    """Write an inference record to a CIF file and return the path."""
    structure = to_structure(record)
    structure.to(filename=path, fmt="cif")
    return path


# Friendly public alias.
generate = run_inference
