"""Training package tests.

The strongest guarantee we can make without a GPU + full dataset is that the
compact trainer constructs *exactly* the architecture stored in the released
checkpoints (so a reproduction run would load/compare cleanly). We verify the
freshly-built model's ``state_dict`` keys match the packflow-ddp checkpoint.
"""

import pytest

torch = pytest.importorskip("torch")

from packflow.training import TrainConfig, build_flow_matching  # noqa: E402


def test_config_roundtrip():
    cfg = TrainConfig(d_model=640, nhead=10, use_rdkit_features=True)
    d = cfg.to_dict()
    cfg2 = TrainConfig.from_dict(d)
    assert cfg2.d_model == 640 and cfg2.nhead == 10 and cfg2.use_rdkit_features
    # Unknown keys are ignored.
    assert TrainConfig.from_dict({"d_model": 8, "bogus": 1}).d_model == 8


def test_build_matches_ddp_checkpoint(local_models):
    if "packflow-ddp" not in local_models:
        pytest.skip("packflow-ddp checkpoint not present locally")

    from packflow import checkpoints

    # Config mirrors train_60M_512.sh (the run that produced packflow-ddp).
    cfg = TrainConfig(
        coordinate_system="cartesian", d_model=640, nhead=10, dim_feedforward=2560,
        num_transformer_layers=12, use_rdkit_features=True,
        use_attention_bias_from_graph=True, use_positional_embeddings=False,
        lattice_loss_weight=1.0,
    )
    fm = build_flow_matching(cfg, device="cpu")
    built_keys = set(fm.model.state_dict().keys())

    ckpt = torch.load(checkpoints.resolve("packflow-ddp", download=False), map_location="cpu")
    saved_keys = set(ckpt["model_state_dict"].keys())

    assert built_keys == saved_keys, (
        f"architecture mismatch: missing={sorted(saved_keys - built_keys)[:5]} "
        f"unexpected={sorted(built_keys - saved_keys)[:5]}"
    )
