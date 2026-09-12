"""Top-level import + graceful-degradation tests."""

import importlib


def test_import_packflow_never_raises():
    pkg = importlib.import_module("packflow")
    # Subsystem import failures are recorded, not raised.
    assert isinstance(pkg._import_errors, dict)


def test_config_and_checkpoints_always_available():
    from packflow import checkpoints, config

    assert config.REPO_ROOT.exists()
    zoo = checkpoints.load_zoo()
    assert "packflow-pa" in zoo


def test_models_available_with_torch():
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        import pytest

        pytest.skip("torch not installed")

    import packflow

    # With the full stack installed, the models/inference subsystems must import.
    assert packflow._import_errors.get("models") is None, packflow._import_errors.get("models")
    assert packflow.CrystalFlowMatching is not None
    assert packflow.load_checkpoint is not None
    assert packflow.list_models is not None
