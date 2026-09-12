# Data preprocessing (CSD → training tensors)

PackFlow's training data is extracted from the **Cambridge Structural Database (CSD)**
using a lightly-modified fork of [Crystal_Math](https://github.com/nigalanakis/Crystal_Math).

> **CSD license required.** The CSD and its Python API (`ccdc`) are proprietary and
> are **not** redistributable. You need a valid CCDC license and the CSD Python API
> installed (Python **3.9**). We therefore ship only our *edits* to Crystal_Math
> (as an overlay) plus the refcode lists in [`../data/refcodes/`](../data/refcodes),
> not the raw structures or processed tensors.

## What's here

```
preprocessing/
├── crystal_math_overlay/        # ONLY our edits to Crystal_Math @ 8808429
│   ├── source_code/             #   csd_operations.py, csd_data_extraction.py, create_reference_fragments.py
│   ├── input_files/             #   input_data_extraction.txt (+ all-crystals variant)
│   └── source_data/             #   reference_fragment_list.json
├── apply_overlay.sh             # copies the overlay onto external/Crystal_Math
├── create_dataset_splits.py     # build train/val/test splits (seed 42)
└── process_premade_splits.py    # process refcode splits → model-ready tensors
```

Run these via the CLI: `packflow preprocess {splits,premade} ...`.

The upstream Crystal_Math code lives in the pinned submodule
[`../external/Crystal_Math`](../external/Crystal_Math) (commit `8808429`).

## Steps

1. **Init the submodule and apply our overlay:**
   ```bash
   git submodule update --init external/Crystal_Math
   bash preprocessing/apply_overlay.sh
   ```

2. **Point at your CSD database and activate your license:**
   ```bash
   export CSD_DATABASE_PATH=/path/to/your/csd/<db>.sqlite
   # one-time license activation with YOUR key (see extract_data.sh)
   ```

3. **Extract homomolecular crystals** (run inside the CSD/ccdc env, Python 3.9):
   ```bash
   cd external/Crystal_Math
   bash extract_data.sh        # writes CIFs under csd_db_analysis_large/db_data/
   ```

4. **Build splits + process into tensors** (back in the packflow env):
   ```bash
   # Reproduce our exact splits: first arrange the extracted CIFs into
   # train/ val/ test/ subdirectories according to the shipped refcode lists in
   # ../data/refcodes/, then process each split into tensors:
   packflow preprocess premade \
       --data_dir <dir-with-train_val_test-subdirs> \
       --output_dir data/processed_data_lt200_symmetrized_niggli_tag_hbond_hydrogen_aromatic_rings \
       --tag_hydrogen_bonding --find_aromatic_rings

   # …or build a brand-new split from scratch (does NOT reproduce our lists):
   packflow preprocess splits
   ```

The resulting `processed_data_*` directory is what `packflow train` consumes via
`--processed_data_dir` (or `PACKFLOW_DATA_DIR`).
