"""Command-line interface for base-model training.

Exposes ``build_parser`` / ``main`` behind the ``packflow train`` subcommand so
the argument surface lives in exactly one place.
"""

from __future__ import annotations

import argparse


def optional_float(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    return None if s in ("none", "null", "nan") else float(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a PackFlow base flow-matching model.")
    # data / run
    p.add_argument("--processed_data_dir", default="processed_data")
    p.add_argument("--experiments_root", default="experiments")
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    # optimisation
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--validate_every", type=int, default=1)
    p.add_argument("--log_every_n_batches", type=int, default=5)
    p.add_argument("--distributed", action="store_true")
    # architecture
    p.add_argument("--coordinate_system", default="cartesian", choices=["cartesian"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--num_transformer_layers", type=int, default=4)
    p.add_argument("--disable_positional_embeddings", action="store_true")
    p.add_argument("--use_rdkit_features", action="store_true")
    p.add_argument("--lattice_loss_weight", type=float, default=0.1)
    p.add_argument("--shared_time", action="store_true")
    # attention bias
    p.add_argument("--use_attention_bias_from_graph", action="store_true")
    p.add_argument("--attn_bias_heads", type=int, default=None)
    p.add_argument("--attn_bias_combine", default="sum", choices=["sum", "mean"])
    p.add_argument("--attn_bias_baseline", type=float, default=0.0)
    p.add_argument("--bias_scale", type=float, default=1.0)
    p.add_argument("--learnable_bias_scale", action="store_true")
    # gnn for bonds
    p.add_argument("--use_gnn_for_bonds", action="store_true")
    p.add_argument("--num_gnn_layers", type=int, default=3)
    p.add_argument("--gat_heads", type=int, default=4)
    # timestep resampling
    p.add_argument("--use_logit_normal_resampling", action="store_true")
    p.add_argument("--logit_normal_m", type=float, default=-0.8)
    p.add_argument("--logit_normal_s", type=float, default=1.7)
    p.add_argument("--logit_normal_mix", type=float, default=0.02)
    p.add_argument("--t_eps", type=float, default=0.0)
    p.add_argument("--fixed_time", type=float, default=None)
    # auxiliary losses
    p.add_argument("--use_smooth_lddt_loss", action="store_true")
    p.add_argument("--smooth_lddt_loss_weight", type=float, default=1.0)
    p.add_argument("--lddt_cutoff", type=float, default=15.0)
    p.add_argument("--lddt_weight_schedule", action="store_true")
    p.add_argument("--use_bond_length_loss", action="store_true")
    p.add_argument("--bond_length_loss_weight", type=float, default=1.0)
    p.add_argument("--use_periodic_lddt_loss", action="store_true")
    p.add_argument("--periodic_lddt_loss_weight", type=float, default=1.0)
    p.add_argument("--periodic_lddt_cutoff", type=optional_float, default=None)
    p.add_argument("--periodic_lddt_warmup_epochs", type=int, default=0)
    # coordinate / periodic embedding
    p.add_argument("--lattice_token", action="store_true")
    p.add_argument("--periodic_coord_emb", action="store_true")
    p.add_argument("--periodic_nmax", type=int, default=5)
    p.add_argument("--periodic_topk", type=int, default=512)
    p.add_argument("--add_periodic_edges", action="store_true")
    p.add_argument("--periodic_edge_cutoff", type=float, default=5.0)
    p.add_argument("--periodic_edge_time_cutoff", type=optional_float, default=0.5)
    p.add_argument("--periodic_edge_periodic", dest="periodic_edge_periodic", action="store_true")
    p.add_argument("--no_periodic_edge_periodic", dest="periodic_edge_periodic", action="store_false")
    p.set_defaults(periodic_edge_periodic=True)
    p.add_argument("--use_fractional_coords", action="store_true")
    p.add_argument("--use_k_basis_representation", action="store_true")
    # logging
    p.add_argument("--grad_log_every", type=int, default=0)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="molecular-crystals")
    p.add_argument("--wandb_name", default=None)
    return p


def args_to_config(args):
    from packflow.training import TrainConfig

    data = vars(args).copy()
    data["use_positional_embeddings"] = not data.pop("disable_positional_embeddings")
    return TrainConfig.from_dict(data)


def main(argv=None) -> None:
    import os

    import torch
    import torch.distributed as dist

    from packflow.training import train

    args = build_parser().parse_args(argv)

    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    cfg = args_to_config(args)
    train(cfg)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
