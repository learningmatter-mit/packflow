"""Checkpoint loading tests (heavy: needs torch + local checkpoints).

These verify the hard reproducibility constraint: every shipped checkpoint loads
its ``state_dict`` cleanly into the canonical model, with no missing/unexpected
keys (i.e. the architecture still matches what was trained).
"""

import pytest

torch = pytest.importorskip("torch")

from packflow import checkpoints  # noqa: E402


@pytest.mark.parametrize("name", ["packflow-2M", "packflow-20M", "packflow-ddp", "packflow-pa"])
def test_state_dict_matches_model(name, local_models):
    if name not in local_models:
        pytest.skip(f"{name} checkpoint not present locally")

    from packflow.grpo.model_loader import get_model_config

    path = checkpoints.resolve(name, download=False)
    ckpt = torch.load(path, map_location="cpu")
    assert "model_state_dict" in ckpt

    # Build the model exactly as the loader does (config from checkpoint).
    from packflow.models import CrystalTransformerEncoder

    valid = {
        "d_model", "nhead", "dim_feedforward", "activation", "dropout", "norm_first", "bias",
        "num_layers", "time_embed_dim", "cart_coords_dim", "lattice_dim",
        "use_attention_bias_from_graph", "attn_bias_heads", "attn_bias_combine",
        "attn_bias_baseline", "bias_scale", "learnable_bias_scale",
        "use_positional_embeddings", "use_rdkit_features", "lattice_token",
        "rdkit_bond_feat_dim", "rdkit_node_feat_dim",
        "periodic_coord_emb", "periodic_nmax", "periodic_topk", "use_fractional_coords",
    }
    cfg = {k: v for k, v in ckpt.get("model_config", {}).items() if k in valid}
    for k, v in get_model_config("packflow-ddp").items():
        cfg.setdefault(k, v)

    model = CrystalTransformerEncoder(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing, f"missing keys for {name}: {missing[:10]}"
    assert not unexpected, f"unexpected keys for {name}: {unexpected[:10]}"


def test_load_checkpoint_api(local_models):
    if "packflow-2M" not in local_models:
        pytest.skip("packflow-2M not present locally")
    from packflow import load_checkpoint

    model = load_checkpoint("packflow-2M", device="cpu")
    assert model is not None
