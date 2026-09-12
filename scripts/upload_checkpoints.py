#!/usr/bin/env python3
"""Upload the curated PackFlow checkpoints to the Hugging Face Hub.

This is a deferred, manual step -- run it once, after the refactor is confirmed,
with an HF token that can write to the target repo. Nothing in the library calls
this; it only mirrors the local model zoo to the Hub so fresh clones can download
checkpoints via ``packflow.download_checkpoints()``.

Usage::

    huggingface-cli login            # or set HF_TOKEN
    python scripts/upload_checkpoints.py --repo aksub99/packflow-checkpoints

Add ``--create`` to create the repo if it does not exist yet.
"""

from __future__ import annotations

import argparse
import sys

from packflow import checkpoints as ckpt
from packflow import config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=config.hf_repo(),
                        help="Target HF repo id (default: %(default)s).")
    parser.add_argument("--token", default=None,
                        help="HF token (else uses cached login / HF_TOKEN).")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model-zoo names to upload (default: all).")
    parser.add_argument("--create", action="store_true",
                        help="Create the repo if it does not exist.")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is required: uv pip install huggingface_hub", file=sys.stderr)
        return 1

    zoo = ckpt.load_zoo()
    names = args.models or list(zoo)
    api = HfApi(token=args.token)

    if args.create:
        api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)

    for name in names:
        meta = zoo[name]
        local = ckpt.local_path(name, zoo)
        if not local.exists():
            print(f"[skip] {name}: local file missing at {local}")
            continue
        filename = meta.get("hf_filename", meta["path"])
        print(f"[upload] {name}: {local} -> {args.repo}/{filename}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=filename,
            repo_id=args.repo,
            repo_type="model",
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
