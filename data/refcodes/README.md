# CSD refcode lists

These are the **CSD refcodes** that define PackFlow's dataset splits and blind
test. We ship the refcodes (not the structures or processed tensors) so the exact
splits can be reproduced from the CSD by anyone with a license.

| File | Count | Description |
|------|-------|-------------|
| `train.txt`       | 77 663 | family-disjoint training split |
| `val.txt`         | 28 310 | family-disjoint validation split |
| `test.txt`        | 312    | family-disjoint evaluation set used for Tables 1 & 2 |
| `blind_test.txt`  | 2      | CSP blind-test crystals used in Figure 5 (`XAFPAY01`, `OBEQOD`) |

Families are the standard CSD 6-character refcode root (`XAFPAY01` → `XAFPAY`).
A family is assigned to exactly one of train / val / test. `blind_test.txt` is a
separate pair of held-out CSP Blind Test case studies (not part of the 312); those
families are also excluded from train and val.

`test.txt` is the **N = 312** evaluation set on which all reported test-set
metrics are computed. Train and val keep approximately the same size ratio as
the original lists (~73 : 27).

The released checkpoints were trained on homomolecular crystals with
**≤ 200 atoms**, with unit cells symmetrized and Niggli-reduced. Those
checkpoints used the earlier train/val lists (which shared some refcodes);
the lists here are the family-disjoint replacement for future training.

## Reproducing the dataset from these refcodes

1. Extract the structures from the CSD (license required) — see
   [`../../preprocessing/README.md`](../../preprocessing/README.md).
2. Keep only the refcodes listed here for each split, then process them into
   tensors with `preprocessing/process_premade_splits.py`.

These lists are the authoritative definition of the evaluation splits used in
the paper. `preprocessing/create_dataset_splits.py` is a general-purpose
splitting utility for building a *new* split from scratch — it does not
regenerate these exact lists.
