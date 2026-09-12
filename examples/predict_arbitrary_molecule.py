#!/usr/bin/env python3
"""Predict crystal packings for an arbitrary molecule.

PackFlow can pack a molecule it has never seen, given only its molecular graph
(which atoms, and which atoms are bonded) - no input coordinates required.

This example packs benzene with the GRPO-finetuned model and writes the sampled
packings to CIF files. Run it from a source checkout after ``uv sync``::

    python examples/predict_arbitrary_molecule.py

Tip: the equivalent one-liner from the command line is::

    packflow predict --smiles "c1ccccc1" --model packflow-60M --n_samples 4 --out_dir benzene_cifs
"""

from packflow import load_checkpoint, predict, to_structure, write_cif


def main() -> None:
    # A model from the zoo (downloaded from Hugging Face on first use) or a .pt path.
    model = load_checkpoint("packflow-60M", device="cpu")

    # Benzene as a graph: 6 carbons in a ring. Hydrogens are implicit, so we only
    # describe the heavy-atom skeleton. The SMILES lets RDKit assign aromatic bond
    # orders; alternatively pass bond_types=["AROMATIC", ...].
    elements = ["C"] * 6
    bond_index = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    records = predict(
        elements,
        bond_index,
        model=model,
        smiles="c1ccccc1",
        n_samples=4,
        n_steps=100,
    )

    for rec in records:
        structure = to_structure(rec)
        path = write_cif(rec, f"benzene_sample_{rec['sample_index']}.cif")
        a, b, c, alpha, beta, gamma = rec["lattice_params"].tolist()
        print(
            f"sample {rec['sample_index']}: "
            f"cell=({a:.2f}, {b:.2f}, {c:.2f}) A, "
            f"angles=({alpha:.1f}, {beta:.1f}, {gamma:.1f}) deg, "
            f"{len(structure)} atoms -> {path}"
        )


if __name__ == "__main__":
    main()
