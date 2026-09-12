"""Unified ``packflow`` command-line interface.

A single entry point over the library:

    packflow download [names...]        # fetch checkpoints from the model zoo
    packflow predict   ...              # pack an arbitrary molecule (SMILES/graph) -> CIFs
    packflow generate  ...              # sample crystals from a structure file -> CIFs
    packflow train     ...              # train a base flow-matching model
    packflow train-grpo ...             # GRPO post-training (the "PA" model)
    packflow evaluate  ...              # full test-set / blind-test evaluation
    packflow relax     ...              # UMA relaxation + lattice energy (Figure 5)
    packflow preprocess {splits,premade} ...   # build dataset splits

Each subcommand forwards its remaining arguments to the matching library entry
point, so ``packflow train --help`` shows the full training options.
"""

from __future__ import annotations

import sys
from typing import List, Optional

_SUBCOMMANDS = [
    "download", "predict", "generate", "train", "train-grpo",
    "evaluate", "relax", "preprocess",
]


def _call_with_argv(func, name: str, rest: List[str]):
    """Run ``func`` whose ``main`` reads ``sys.argv`` (argparse-based)."""
    old = sys.argv
    try:
        sys.argv = [f"packflow {name}", *rest]
        return func()
    finally:
        sys.argv = old


def _cmd_download(rest: List[str]) -> None:
    import argparse

    from packflow import checkpoints

    p = argparse.ArgumentParser(prog="packflow download")
    p.add_argument("names", nargs="*", help="model names (default: all)")
    p.add_argument("--list", action="store_true", help="list available models")
    args = p.parse_args(rest)
    if args.list:
        for name in sorted(checkpoints.load_zoo()):
            print(name)
        return
    for name, path in checkpoints.download_checkpoints(args.names or None).items():
        print(f"{name}: {path}")


def _cmd_predict(rest: List[str]) -> None:
    import argparse
    import os

    from packflow.inference import load_checkpoint, predict, write_cif

    p = argparse.ArgumentParser(
        prog="packflow predict",
        description="Predict crystal packings for an arbitrary molecule (no coordinates needed).",
    )
    src = p.add_argument_group("molecule (use --smiles, or --elements + --bonds)")
    src.add_argument("--smiles", help="SMILES of the molecule to pack")
    src.add_argument("--elements", nargs="*", help="element symbols, e.g. C C O")
    src.add_argument("--bonds", nargs="*", help="bonds as i-j pairs, e.g. 0-1 1-2")
    p.add_argument("--model", default="packflow-pa", help="zoo name or checkpoint path")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--z", type=int, default=1, help="molecules per unit cell")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="generated", help="directory for output CIFs")
    args = p.parse_args(rest)

    smiles = args.smiles
    if args.elements:
        elements = args.elements
        bond_index = [tuple(int(x) for x in b.split("-")) for b in (args.bonds or [])]
    elif smiles:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            sys.stderr.write(f"packflow predict: could not parse SMILES {smiles!r}\n")
            raise SystemExit(2)
        elements = [a.GetSymbol() for a in mol.GetAtoms()]
        bond_index = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    else:
        sys.stderr.write("packflow predict: provide --smiles or --elements (+ --bonds)\n")
        raise SystemExit(2)

    model = load_checkpoint(args.model, device=args.device)
    records = predict(
        elements,
        bond_index,
        model=model,
        smiles=smiles,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        temperature=args.temperature,
        z=args.z,
        device=args.device,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    for rec in records:
        stem = "".join(c if c.isalnum() else "_" for c in str(rec.get("refcode", "molecule")))
        name = f"{stem}_{rec.get('sample_index', 0)}.cif"
        path = write_cif(rec, os.path.join(args.out_dir, name))
        print(path)


def _cmd_generate(rest: List[str]) -> None:
    import argparse

    from packflow.inference import load_checkpoint, run_inference, write_cif

    p = argparse.ArgumentParser(prog="packflow generate")
    p.add_argument("--model", default="packflow-pa", help="zoo name or checkpoint path")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mmcif", help="input mmCIF file (processed on the fly)")
    src.add_argument("--processed_path", help="pre-processed .pt dataset")
    p.add_argument("--refcode", nargs="*", default=None, help="filter processed_path by refcode(s)")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="generated", help="directory for output CIFs")
    args = p.parse_args(rest)

    import os

    model = load_checkpoint(args.model, device=args.device)
    records = run_inference(
        model,
        mmcif=args.mmcif,
        processed_path=args.processed_path,
        refcode=args.refcode,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        temperature=args.temperature,
        device=args.device,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    for rec in records:
        name = f"{rec.get('refcode', 'crystal')}_{rec.get('sample_index', 0)}.cif"
        path = write_cif(rec, os.path.join(args.out_dir, name))
        print(path)


def _cmd_train(rest: List[str]) -> None:
    from packflow.training.cli import main
    main(rest)


def _cmd_train_grpo(rest: List[str]) -> None:
    from packflow.grpo.trainer import main
    _call_with_argv(main, "train-grpo", rest)


def _cmd_evaluate(rest: List[str]) -> None:
    from packflow.evaluation.evaluator import main
    _call_with_argv(main, "evaluate", rest)


def _cmd_relax(rest: List[str]) -> None:
    from packflow.relaxation.pipeline import main
    _call_with_argv(main, "relax", rest)


def _cmd_preprocess(rest: List[str]) -> None:
    import os
    import runpy

    from packflow import config

    cmds = {
        "splits": "create_dataset_splits.py",
        "premade": "process_premade_splits.py",
    }
    if not rest or rest[0] not in cmds:
        sys.stderr.write(f"usage: packflow preprocess {{{'|'.join(cmds)}}} [args...]\n")
        raise SystemExit(2)
    script = os.path.join(config.REPO_ROOT, "preprocessing", cmds[rest[0]])
    if not os.path.exists(script):
        sys.stderr.write(
            f"preprocessing script not found: {script}\n"
            "Run from a source checkout (the preprocessing/ dir is not shipped in wheels).\n"
        )
        raise SystemExit(1)
    old = sys.argv
    try:
        sys.argv = [script, *rest[1:]]
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = old


_DISPATCH = {
    "download": _cmd_download,
    "predict": _cmd_predict,
    "generate": _cmd_generate,
    "train": _cmd_train,
    "train-grpo": _cmd_train_grpo,
    "evaluate": _cmd_evaluate,
    "relax": _cmd_relax,
    "preprocess": _cmd_preprocess,
}


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        prog = "packflow"
        print(f"usage: {prog} {{{','.join(_SUBCOMMANDS)}}} ...\n")
        print("Subcommands:")
        for name in _SUBCOMMANDS:
            print(f"  {name}")
        print(f"\nRun '{prog} <subcommand> --help' for details.")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in _DISPATCH:
        sys.stderr.write(f"packflow: unknown subcommand {cmd!r}. "
                         f"Choose from: {', '.join(_SUBCOMMANDS)}\n")
        raise SystemExit(2)
    _DISPATCH[cmd](rest)


if __name__ == "__main__":
    main()
