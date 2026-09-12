# PackFlow
[![arXiv](https://img.shields.io/badge/arXiv-2410.08833-84cc16)](https://arxiv.org/abs/2602.20140)
[![MIT](https://img.shields.io/badge/License-MIT-3b82f6.svg)](https://opensource.org/license/mit)

This repository is the code accompanying the paper
[PackFlow: Generative Molecular Crystal Structure Prediction via Reinforcement Learning Alignment](https://arxiv.org/abs/2602.20140).

PackFlow predicts how a molecule packs into a crystal. Given a molecule's graph
(its atoms and bonds — no coordinates), it samples full crystal structures:
Cartesian coordinates for every atom plus the unit cell. This is a
self-contained, `uv`-installable reproduction of the paper, covering the
full pipeline: data preprocessing (from the CSD), base-model training
(2M / 20M / 60M), GRPO post-training (the preference-aligned "PA" model),
evaluation (with the Genarris and UMA baselines), and relaxation / lattice-energy
analysis.

```
packflow/
├── packflow/                # importable package
│   ├── models/model.py      # Cartesian crystal transformer + flow matching (canonical architecture)
│   ├── data/                # dataset / datamodule / collate
│   ├── processing/          # mmCIF/SMILES -> training tensors
│   ├── utils/               # crystal / lattice helpers + coordinate-free graph builder
│   ├── training/            # compact trainer + TrainConfig + train()
│   ├── grpo/                # GRPO post-training (trainer, sampler, model loader)
│   ├── evaluation/          # crystal-matching metrics + evaluator + evaluate()
│   ├── relaxation/          # UMA energy/forces, relaxation, lattice energy (Figure 5 backbone)
│   ├── baselines/           # Genarris wrapper
│   ├── inference/           # high-level load_checkpoint() / predict() / run_inference()
│   ├── checkpoints/         # model_zoo.json (the .pt files are fetched from Hugging Face)
│   ├── config.py            # central paths / env vars
│   ├── checkpoints.py       # zoo resolution + HF download
│   └── cli.py               # `packflow` command-line interface
├── examples/                # runnable usage examples
├── tests/                   # pytest suite
├── external/                # pinned git submodules: Crystal_Math, fairchem (UMA), Genarris
├── preprocessing/           # our Crystal_Math edits (overlay) + split scripts (CSD license)
└── data/refcodes/           # CSD refcodes for train/val/test + blind test (NOT the dataset)
```

## Installation

PackFlow uses [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo> packflow && cd packflow

# Pinned external code: Crystal_Math (preprocessing), fairchem (UMA), Genarris (baseline)
git submodule update --init --recursive

uv sync                       # core: training, sampling/inference, evaluation metrics

# Optional extras
uv pip install -e ".[grpo]"   # GRPO post-training (wandb, torch-scatter)

# UMA / fairchem (relaxation, lattice energy, GRPO reward) — installed from the submodule:
uv pip install -e external/fairchem/packages/fairchem-core
```

> **Python**: 3.10 is recommended. The **CSD preprocessing** step needs Python
> **3.9** and a proprietary CCDC license (see
> [`preprocessing/README.md`](preprocessing/README.md)); it is only needed to
> rebuild the training dataset.

## Quickstart: pack an arbitrary molecule

PackFlow can pack a molecule it has never seen, given only its graph. The easiest
way is a SMILES string:

```python
from packflow import load_checkpoint, predict, write_cif

model = load_checkpoint("packflow-60M", device="cpu")   # downloaded from HF on first use

# Benzene: 6 carbons in a ring (hydrogens implicit). SMILES sets the bond orders.
results = predict(
    elements=["C"] * 6,
    bond_index=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    smiles="c1ccccc1",
    model=model,
    n_samples=8,
)
write_cif(results[0], "benzene_sample0.cif")
```

…or from the command line:

```bash
packflow predict --smiles "c1ccccc1" --model packflow-60M --n_samples 8 --out_dir generated
```

You can describe any molecule by its atoms and bonds (see
[`examples/predict_arbitrary_molecule.py`](examples/predict_arbitrary_molecule.py)),
or start from a structure file with `packflow generate --mmcif ...` /
`run_inference(...)`.

Available models: `packflow-2M`, `packflow-20M`, `packflow-ddp` (≡ `packflow-60M`),
`packflow-pa` (GRPO). Run `packflow download --list`; details in
[`packflow/checkpoints/README.md`](packflow/checkpoints/README.md).

### The `packflow` CLI

A single entry point covers the whole pipeline (each subcommand forwards `--help`):

```bash
packflow download [names...]      # fetch checkpoints from the model zoo
packflow predict   ...            # pack an arbitrary molecule (SMILES/graph) -> CIFs
packflow generate  ...            # sample crystals from a structure file -> CIFs
packflow train     ...            # train a base flow-matching model
packflow train-grpo ...           # GRPO post-training (the "PA" model)
packflow evaluate  ...            # full test-set / blind-test evaluation
packflow relax     ...            # UMA relaxation + lattice energy (Figure 5)
packflow preprocess {splits,premade} ...
```

## End-to-end reproduction

All commands run from the repository root. The `packflow` CLI is the supported
entry point on any machine (cluster job scripts are not part of this checkout).

1. **Preprocess** (CSD license) — regenerate the exact splits from the shipped
   refcodes in [`data/refcodes/`](data/refcodes); see
   [`preprocessing/README.md`](preprocessing/README.md):
   ```bash
   packflow preprocess premade ...
   ```

2. **Train base models** (2M / 20M / 60M):
   ```bash
   export PACKFLOW_DATA_DIR=<your processed-data dir>
   packflow train --batch_size 128 --use_rdkit_features --use_attention_bias_from_graph --lr 3e-5
   ```

3. **GRPO post-training** (the PA model — a GRPO finetune of `packflow-ddp`):
   ```bash
   packflow train-grpo --model_type packflow-ddp \
       --checkpoint_path packflow/checkpoints/packflow-ddp/best_model.pt
   ```

4. **Evaluate** (Tables 1/2, Figure 4 λ-sweep, Figure 5 blind test):
   ```bash
   packflow evaluate --model packflow-pa --data_dir "$PACKFLOW_DATA_DIR"
   ```

5. **Relaxation + lattice energy** (Figure 5 backbone) — see
   [`packflow/relaxation/README.md`](packflow/relaxation/README.md):
   ```bash
   packflow relax --eval_dir <evaluation_results_dir> --fairchem_python "$FAIRCHEM_PYTHON"
   ```

The figure- and table-rendering code is maintained separately from this package
(it depends on a large internal cache of evaluation artifacts) and is not part of
the open-source surface.

## External requirements

- **Checkpoints (required for inference/eval).** The `.pt` files (~1.6 GB) are kept
  off git and fetched from the Hugging Face Hub on first use; resolution is
  local-first (`packflow/checkpoints/<name>/best_model.pt` if present). Run
  `packflow download` to pre-fetch. See
  [`packflow/checkpoints/README.md`](packflow/checkpoints/README.md).
- **UMA weights (for relaxation / lattice energy / GRPO reward).** Installed via the
  `external/fairchem` submodule; model weights download automatically on first use
  (cache via `FAIRCHEM_CACHE_DIR`).
- **CSD (only to rebuild the dataset).** A CCDC license + the CSD Python API
  (Python 3.9) are needed for preprocessing. Inference, training-from-processed-data,
  and evaluation do **not** need the CSD. No raw/processed dataset is shipped; the
  refcodes in [`data/refcodes/`](data/refcodes) let you regenerate the exact splits.

### Useful environment variables

| Variable | Purpose |
|----------|---------|
| `PACKFLOW_DATA_DIR` | processed-data directory used by training/eval |
| `PACKFLOW_CHECKPOINT_DIR` | where model-zoo checkpoints live |
| `FAIRCHEM_PYTHON` / `FAIRCHEM_CACHE_DIR` | UMA interpreter / weight cache |
| `GENARRIS_PYTHON` | python interpreter for the Genarris baseline |
| `CSD_DATABASE_PATH` | path to your local CSD database (preprocessing) |
| `PACKFLOW_EVAL_STAGING_DIR` | optional fast scratch dir for UMA staging |

All paths funnel through [`packflow/config.py`](packflow/config.py).

## External code (pinned submodules)

| Submodule | Commit | Use |
|-----------|--------|-----|
| [`external/Crystal_Math`](external/Crystal_Math) | `8808429` | CSD data extraction (our edits overlaid via `preprocessing/`) |
| [`external/fairchem`](external/fairchem) | `a0a984b7` | UMA energy/relaxation + GRPO reward |
| [`external/Genarris`](external/Genarris) | `301a8ea` | CSP baseline |

## Citation

If you use PackFlow, please cite the paper
([arXiv:2602.20140](https://arxiv.org/abs/2602.20140)):

```bibtex
@article{subramanian2026packflow,
  title={PackFlow: Generative Molecular Crystal Structure Prediction via Reinforcement Learning Alignment},
  author={Subramanian, Akshay and Pan, Elton and Nam, Juno and Weiler, Maurice and Qu, Shuhui and Park, Cheol Woo and Jaakkola, Tommi S and Olivetti, Elsa and Gomez-Bombarelli, Rafael},
  journal={arXiv preprint arXiv:2602.20140},
  year={2026}
}
```

This project is released under the MIT License ([`LICENSE`](LICENSE)).
