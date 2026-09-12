# Model zoo

Curated final checkpoints for PackFlow. Each model lives at
`<name>/best_model.pt` and is described in [`model_zoo.json`](model_zoo.json).

| Name | Params | Role | Notes |
|------|--------|------|-------|
| `packflow-2M`  | ~2M  | base    | size sweep (Tables 1/2) |
| `packflow-20M` | ~20M | base    | size sweep (Tables 1/2) |
| `packflow-ddp` | ~60M | base    | **canonical 60M** (main tables, GRPO base); alias `packflow-60M` |
| `packflow-pa`  | ~60M | GRPO PA | preference-aligned: GRPO finetune of `packflow-ddp` (E025, λ=1.0); Figures 4 & 5 |

Load with the high-level API:

```python
from packflow import load_checkpoint
model = load_checkpoint("packflow-pa")          # or "packflow-ddp", "packflow-20M", ...
```

or via the GRPO loader:

```python
from packflow.grpo import load_model_by_type
model = load_model_by_type("packflow-ddp", device="cuda")
```

## Distribution / version control

These files total ~1.6 GB, so they are **not** committed to git (`*.pt` is in
`.gitignore`) and the repository stays small and clone-able. Instead:

- Checkpoints are resolved **local-first** by [`packflow/checkpoints.py`](../checkpoints.py):
  if `packflow/checkpoints/<name>/best_model.pt` exists it is used directly.
- Otherwise they are downloaded once from the Hugging Face Hub repo named in
  `model_zoo.json` (`hf_repo`/`hf_filename`), controllable via `PACKFLOW_HF_REPO`.

```python
from packflow import download_checkpoints, load_checkpoint
download_checkpoints()                 # fetch all (only if not already local)
model = load_checkpoint("packflow-pa") # auto-downloads if missing, else uses local
```

Uploading the checkpoints to the Hub is a separate, manual step (run once, needs
an HF token): see [`scripts/upload_checkpoints.py`](../../scripts/upload_checkpoints.py).
