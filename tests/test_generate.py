"""CPU generate smoke test: sampling runs and produces a valid crystal.

Uses the smallest model (``packflow-2M``) and a tiny synthetic molecular graph so
no shipped dataset is required. Verifies the ODE sampler runs on CPU and returns
finite Cartesian coordinates plus a physically valid 6-parameter lattice.
"""

import pytest

torch = pytest.importorskip("torch")


def _synthetic_template(device="cpu"):
    # 4-atom molecule (C, C, O, N) with a small connected bond graph.
    atom_types = torch.tensor([6, 6, 8, 7], dtype=torch.long, device=device)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long, device=device
    )
    n_atoms = atom_types.shape[0]
    n_edges = edge_index.shape[1]
    # rdkit feature dims for the released models: node=10, bond=2.
    return {
        "atom_types": atom_types,
        "edge_index": edge_index,
        "node_features": torch.zeros(n_atoms, 10, device=device),
        "bond_features": torch.zeros(n_edges, 2, device=device),
    }


def test_cpu_generate_valid_crystal(local_models):
    if "packflow-2M" not in local_models:
        pytest.skip("packflow-2M checkpoint not present locally")

    from packflow import load_checkpoint

    model = load_checkpoint("packflow-2M", device="cpu")
    template = _synthetic_template("cpu")

    with torch.no_grad():
        cart_coords, lattice_params = model.sample(
            template, n_steps=3, low_temperature_lambda=1.0
        )

    assert cart_coords.shape[0] == template["atom_types"].shape[0]
    assert cart_coords.shape[1] == 3
    assert torch.isfinite(cart_coords).all(), "non-finite coordinates"

    lattice = lattice_params.reshape(-1)
    assert lattice.numel() == 6
    assert torch.isfinite(lattice).all(), "non-finite lattice params"
    a, b, c = lattice[:3].tolist()
    alpha, beta, gamma = lattice[3:].tolist()
    assert a > 0 and b > 0 and c > 0, f"non-positive cell lengths: {(a, b, c)}"
    for ang in (alpha, beta, gamma):
        assert 0.0 < ang < 180.0, f"cell angle out of range: {ang}"
