"""Model-zoo resolution tests (light: no torch required)."""

import os

import pytest

from packflow import checkpoints


def test_zoo_has_expected_models(zoo):
    assert set(zoo) == {"packflow-2M", "packflow-20M", "packflow-ddp", "packflow-pa"}
    for name, meta in zoo.items():
        assert "path" in meta
        assert meta["path"].endswith("best_model.pt")
        # HF download metadata is wired (upload deferred).
        assert meta.get("hf_filename")


def test_alias_resolution():
    assert checkpoints.canonical_name("packflow-60M") == "packflow-ddp"
    assert checkpoints.canonical_name("packflow-ddp") == "packflow-ddp"
    assert checkpoints.canonical_name("does-not-exist") is None


def test_resolve_local_first(local_models):
    if not local_models:
        pytest.skip("no local checkpoints present")
    for name in local_models:
        path = checkpoints.resolve(name, download=False)
        assert os.path.exists(path)
        assert path.endswith("best_model.pt")


def test_resolve_path_passthrough(tmp_path):
    f = tmp_path / "custom.pt"
    f.write_bytes(b"x")
    assert checkpoints.resolve(str(f)) == str(f)


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        checkpoints.resolve("totally-unknown-model", download=False)
