# `scripts/`

Everything in the reproduction pipeline is driven through the `packflow`
command-line interface (`packflow --help`) — there are no separate wrapper
scripts to learn.

This directory holds one maintainer utility:

| Script | Purpose |
|--------|---------|
| `upload_checkpoints.py` | Push the curated model-zoo checkpoints to the Hugging Face Hub. |

For training, evaluation, relaxation, inference, and preprocessing, use the CLI:

```bash
packflow predict  --smiles "c1ccccc1" --model packflow-60M
packflow generate --model packflow-pa --mmcif my_crystal.mmcif
packflow train    --processed_data_dir data/processed --batch_size 128
packflow evaluate --model packflow-pa --data_dir data/processed
packflow relax    --eval_dir <evaluation_results_dir>
packflow download --list
```
