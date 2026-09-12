"""Model-zoo resolution and (optional) Hugging Face download for PackFlow.

Resolution order for a model name:

1. A local file ``<checkpoint_dir>/<name>/best_model.pt`` (the source of truth
   for all local testing).
2. If missing, a one-time download from the Hugging Face Hub -- only attempted
   when ``huggingface_hub`` is installed and the zoo entry declares ``hf_repo`` /
   ``hf_filename``.

No upload ever happens here; see ``scripts/upload_checkpoints.py`` for the
deferred, manual upload step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config


def zoo_path() -> Path:
    return config.checkpoint_dir() / "model_zoo.json"


def load_zoo() -> Dict[str, Any]:
    """Return the model-zoo mapping (name -> metadata)."""
    with open(zoo_path()) as f:
        return json.load(f)


def list_models() -> Dict[str, Any]:
    """Alias for :func:`load_zoo` (public-API friendly name)."""
    return load_zoo()


def canonical_name(name: str, zoo: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Map a model name or alias to its canonical zoo key (or ``None``)."""
    zoo = zoo if zoo is not None else load_zoo()
    if name in zoo:
        return name
    for key, meta in zoo.items():
        if name in meta.get("aliases", []):
            return key
    return None


def local_path(name: str, zoo: Optional[Dict[str, Any]] = None) -> Path:
    """Expected local checkpoint path for a canonical/alias model name."""
    zoo = zoo if zoo is not None else load_zoo()
    key = canonical_name(name, zoo) or name
    rel = zoo.get(key, {}).get("path", f"{key}/best_model.pt")
    return config.checkpoint_dir() / rel


def download_checkpoint(name: str, zoo: Optional[Dict[str, Any]] = None) -> str:
    """Download a single checkpoint from the Hugging Face Hub into the zoo dir.

    Raises a clear error if huggingface_hub is unavailable or the entry has no
    HF metadata. This is only invoked as a fallback when the local file is
    missing; it is never called during the refactor's local testing.
    """
    zoo = zoo if zoo is not None else load_zoo()
    key = canonical_name(name, zoo)
    if key is None:
        raise ValueError(f"Unknown model '{name}'. Available: {list(zoo)}")
    meta = zoo[key]
    repo = meta.get("hf_repo") or config.hf_repo()
    filename = meta.get("hf_filename")
    if not filename:
        raise FileNotFoundError(
            f"Checkpoint for '{key}' is not present locally at {local_path(key, zoo)} "
            "and the model zoo has no 'hf_filename' to download from. "
            "See packflow/checkpoints/README.md."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "huggingface_hub is required to download checkpoints. "
            "Install it (`uv pip install huggingface_hub`) or place the file at "
            f"{local_path(key, zoo)}."
        ) from exc

    dest = local_path(key, zoo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo, filename=filename)
    if Path(cached).resolve() != dest.resolve():
        # Symlink the HF cache file into the zoo layout so loaders find it.
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(cached, dest)
    return str(dest)


def resolve(name_or_path: str, download: bool = True) -> str:
    """Resolve a model name/alias or a filesystem path to a checkpoint file.

    Args:
        name_or_path: ``packflow-2M``/``packflow-20M``/``packflow-ddp``/
            ``packflow-pa`` (or an alias such as ``packflow-60M``), or a direct
            path to a ``.pt`` checkpoint.
        download: if True, fall back to a Hugging Face download when the local
            file is missing.
    """
    if os.path.exists(name_or_path):
        return name_or_path

    zoo = load_zoo()
    key = canonical_name(name_or_path, zoo)
    if key is None:
        raise ValueError(
            f"Unknown model '{name_or_path}'. Available: {list(zoo)} "
            "(or pass a path to a .pt checkpoint)."
        )

    path = local_path(key, zoo)
    if path.exists():
        return str(path)

    if download:
        return download_checkpoint(key, zoo)

    raise FileNotFoundError(
        f"Checkpoint for '{key}' not found at {path}. "
        "See packflow/checkpoints/README.md for how to obtain checkpoints."
    )


def download_checkpoints(names: Optional[List[str]] = None) -> Dict[str, str]:
    """Download one or more checkpoints (defaults to all zoo entries)."""
    zoo = load_zoo()
    names = names or list(zoo)
    return {n: resolve(n, download=True) for n in names}
