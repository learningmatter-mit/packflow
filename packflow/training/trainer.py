"""Compact training loop for the PackFlow Cartesian flow-matching model.

A faithful, visualization-free training loop: the optimizer, gradient clipping,
validation/best-model selection and checkpoint format are preserved exactly so
training reproduces the paper checkpoints.

Public entry point: :func:`train` (takes a :class:`TrainConfig`).
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from packflow.data.crystal_datamodule import CrystalDataModule
from packflow.models.model import CrystalFlowMatching, CrystalTransformerEncoder
from packflow.utils.batch_processor import load_processed_data

from .config import TrainConfig


# ---------------------------------------------------------------------------
# Distributed / reproducibility helpers
# ---------------------------------------------------------------------------
def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all DDP ranks."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def check_and_fix_system_limits() -> Tuple[int, int]:
    """Raise the open-file soft limit to avoid multiprocessing DataLoader crashes."""
    import resource

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    recommended_limit = 4096
    if soft_limit < recommended_limit:
        try:
            new_soft_limit = min(recommended_limit, hard_limit)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"Warning: could not raise file-descriptor limit: {exc}")
    return soft_limit, hard_limit


def set_all_seeds(seed: int = 42) -> None:
    """Seed python / numpy / torch (and make cuDNN deterministic) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_run_dirs(experiments_root: str = "experiments") -> Dict[str, Path]:
    """Create ``experiments/<run_id>/{checkpoints,plots,wandb}/`` and return the paths."""
    root = Path(experiments_root)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = os.environ.get("SLURM_JOB_NAME", "").strip()
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    parts: List[str] = [ts]
    if job_name:
        parts.append("".join(c if (c.isalnum() or c in "-_.") else "_" for c in job_name))
    if job_id:
        parts.append(f"jid{job_id}")
    run_id = "_".join(parts)
    run_dir = root / run_id
    dirs = {
        "run_id": run_id,
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "plots_dir": run_dir / "plots",
        "wandb_dir": run_dir / "wandb",
    }
    for key in ("checkpoints_dir", "plots_dir", "wandb_dir"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def load_data_splits(processed_data_dir: str) -> Dict[str, str]:
    """Resolve and validate train/val/test ``.pt`` split paths."""
    paths = {s: os.path.join(processed_data_dir, f"{s}.pt") for s in ("train", "val", "test")}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} split file: {path}")
        data = load_processed_data(path)
        print(f"Found {name} split: {len(data)} crystals at {path}")
    return paths


def safe_wandb_log(data: dict, step: int, use_wandb: bool) -> None:
    if not use_wandb:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(data, step=step)
    except Exception as exc:  # pragma: no cover - logging best-effort
        print(f"Warning: wandb logging failed: {exc}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        return "cpu"
    return device


# ---------------------------------------------------------------------------
# Model construction (Cartesian only -- the architecture of every checkpoint)
# ---------------------------------------------------------------------------
def build_flow_matching(cfg: TrainConfig, device: str) -> CrystalFlowMatching:
    if cfg.coordinate_system != "cartesian":
        raise ValueError(
            "Only the Cartesian coordinate system is supported; it is the architecture "
            "used for every released PackFlow checkpoint."
        )
    rdkit_bond_feat_dim = 3 if cfg.add_periodic_edges else 2
    model = CrystalTransformerEncoder(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        num_layers=cfg.num_transformer_layers,
        time_embed_dim=64,
        cart_coords_dim=3,
        lattice_dim=6,
        rdkit_bond_feat_dim=rdkit_bond_feat_dim,
        use_attention_bias_from_graph=cfg.use_attention_bias_from_graph,
        attn_bias_heads=cfg.attn_bias_heads,
        attn_bias_combine=cfg.attn_bias_combine,
        attn_bias_baseline=cfg.attn_bias_baseline,
        bias_scale=cfg.bias_scale,
        learnable_bias_scale=cfg.learnable_bias_scale,
        use_positional_embeddings=cfg.use_positional_embeddings,
        use_rdkit_features=cfg.use_rdkit_features,
        lattice_token=cfg.lattice_token,
        periodic_coord_emb=cfg.periodic_coord_emb,
        periodic_nmax=cfg.periodic_nmax,
        periodic_topk=cfg.periodic_topk,
        use_fractional_coords=cfg.use_fractional_coords,
        use_gnn_for_bonds=cfg.use_gnn_for_bonds,
        num_gnn_layers=cfg.num_gnn_layers,
        gat_heads=cfg.gat_heads,
    )
    flow_matching = CrystalFlowMatching(
        model,
        device=device,
        lattice_loss_weight=cfg.lattice_loss_weight,
        shared_time=cfg.shared_time,
        use_wandb=cfg.use_wandb,
        use_logit_normal_resampling=cfg.use_logit_normal_resampling,
        logit_normal_m=cfg.logit_normal_m,
        logit_normal_s=cfg.logit_normal_s,
        logit_normal_mix=cfg.logit_normal_mix,
        t_eps=cfg.t_eps,
        fixed_time=cfg.fixed_time,
        use_smooth_lddt_loss=cfg.use_smooth_lddt_loss,
        smooth_lddt_loss_weight=cfg.smooth_lddt_loss_weight,
        lddt_cutoff=cfg.lddt_cutoff,
        lddt_weight_schedule=cfg.lddt_weight_schedule,
        use_bond_length_loss=cfg.use_bond_length_loss,
        bond_length_loss_weight=cfg.bond_length_loss_weight,
        grad_log_every=cfg.grad_log_every,
        add_periodic_edges=cfg.add_periodic_edges,
        periodic_edge_cutoff=cfg.periodic_edge_cutoff,
        periodic_edge_periodic=cfg.periodic_edge_periodic,
        periodic_edge_time_cutoff=cfg.periodic_edge_time_cutoff,
        use_periodic_lddt_loss=cfg.use_periodic_lddt_loss,
        periodic_lddt_loss_weight=cfg.periodic_lddt_loss_weight,
        periodic_lddt_cutoff=cfg.periodic_lddt_cutoff,
        periodic_lddt_warmup_epochs=cfg.periodic_lddt_warmup_epochs,
        use_k_basis_representation=cfg.use_k_basis_representation,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Active trainable parameters: {n_params / 1e6:.2f}M")
    return flow_matching


# The checkpoint config dicts are kept identical to the original training script so
# checkpoints written here are byte-for-byte compatible with the model loader.
def _checkpoint_model_config(cfg: TrainConfig) -> dict:
    return {
        "max_num_elements": 100,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "dim_feedforward": cfg.dim_feedforward,
        "num_layers": cfg.num_transformer_layers,
        "time_embed_dim": 64,
        "frac_coords_dim": 3,
        "lattice_dim": 6,
        "num_gnn_layers": 3,
    }


def _checkpoint_training_config(cfg: TrainConfig) -> dict:
    return {
        "lattice_loss_weight": cfg.lattice_loss_weight,
        "shared_time": cfg.shared_time,
        "lr": cfg.lr,
        "n_epochs": cfg.n_epochs,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "log_every_n_batches": cfg.log_every_n_batches,
        "validate_every": cfg.validate_every,
    }


# ---------------------------------------------------------------------------
# Core training loop (Cartesian; logic preserved from the original script)
# ---------------------------------------------------------------------------
def _train_loop(
    cfg: TrainConfig,
    flow_matching: CrystalFlowMatching,
    data_paths: Dict[str, str],
    device: str,
    checkpoint_path: str,
    checkpoints_root: str,
) -> Tuple[CrystalFlowMatching, Dict[str, List[float]]]:
    distributed = cfg.distributed
    local_rank = int(os.environ["LOCAL_RANK"]) if distributed else 0

    data_module = CrystalDataModule(
        train_path=data_paths["train"],
        val_path=data_paths["val"],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=bool(distributed),
    )
    data_module.setup("fit")

    if distributed:
        from torch_geometric.loader import DataLoader

        train_sampler = DistributedSampler(data_module.train_dataset, shuffle=True)
        val_sampler = DistributedSampler(data_module.val_dataset, shuffle=False)
        train_loader = DataLoader(data_module.train_dataset, batch_size=cfg.batch_size,
                                  sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(data_module.val_dataset, batch_size=cfg.batch_size,
                                sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        train_sampler = None

    if cfg.checkpoint_path is None and checkpoints_root:
        flow_matching.checkpoint_dir = checkpoints_root
    if distributed:
        flow_matching.model = DDP(flow_matching.model, device_ids=[local_rank],
                                  output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(flow_matching.model.parameters(), lr=cfg.lr)

    losses: Dict[str, List[float]] = {"total": [], "cart": [], "lattice": []}
    best_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    global_step = 0
    model_config = _checkpoint_model_config(cfg)
    training_config = _checkpoint_training_config(cfg)

    if is_main_process():
        print(f"\nTraining (distributed={distributed}); "
              f"{len(train_loader)} train / {len(val_loader)} val batches per epoch.")

    for epoch in range(cfg.n_epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(flow_matching, "set_current_epoch"):
            flow_matching.set_current_epoch(epoch)

        epoch_losses = {k: [] for k in ("total", "cart", "lattice",
                                        "smooth_lddt_loss", "bond_length_loss", "periodic_lddt_loss")}

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            loss_dict = flow_matching.train_step_batch(batch, optimizer, step=global_step)
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(flow_matching.model.parameters(), max_norm=cfg.grad_clip_norm)

            if distributed:
                for key in ("total_loss", "lattice_loss", "cart_loss"):
                    loss_dict[key] = reduce_tensor(torch.tensor(loss_dict[key], device=device)).item()

            epoch_losses["total"].append(loss_dict["total_loss"])
            epoch_losses["cart"].append(loss_dict["cart_loss"])
            epoch_losses["lattice"].append(loss_dict["lattice_loss"])
            for key in ("smooth_lddt_loss", "bond_length_loss", "periodic_lddt_loss"):
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key])

            global_step += 1

            if (batch_idx + 1) % cfg.log_every_n_batches == 0 and is_main_process():
                n = cfg.log_every_n_batches
                avg_total = sum(epoch_losses["total"][-n:]) / n
                avg_cart = sum(epoch_losses["cart"][-n:]) / n
                avg_lattice = sum(epoch_losses["lattice"][-n:]) / n
                print(f"Epoch {epoch+1}/{cfg.n_epochs}, Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Total: {avg_total:.6f}, Cart: {avg_cart:.6f}, Lattice: {avg_lattice:.6f}", flush=True)

        avg_epoch_loss = {
            "total_loss": sum(epoch_losses["total"]) / len(epoch_losses["total"]),
            "cart_loss": sum(epoch_losses["cart"]) / len(epoch_losses["cart"]),
            "lattice_loss": sum(epoch_losses["lattice"]) / len(epoch_losses["lattice"]),
        }
        losses["total"].append(avg_epoch_loss["total_loss"])
        losses["cart"].append(avg_epoch_loss["cart_loss"])
        losses["lattice"].append(avg_epoch_loss["lattice_loss"])

        if is_main_process():
            print(f"\nEpoch {epoch+1}/{cfg.n_epochs} Summary: "
                  f"Total: {avg_epoch_loss['total_loss']:.6f}, "
                  f"Cart: {avg_epoch_loss['cart_loss']:.6f}, "
                  f"Lattice: {avg_epoch_loss['lattice_loss']:.6f}")
            safe_wandb_log({
                "epoch": epoch + 1,
                "train/epoch_total_loss": avg_epoch_loss["total_loss"],
                "train/epoch_cart_loss": avg_epoch_loss["cart_loss"],
                "train/epoch_lattice_loss": avg_epoch_loss["lattice_loss"],
            }, global_step, cfg.use_wandb)

        # Regular (non-best) checkpoint -- unwrap DDP for a clean state_dict.
        model_to_save = flow_matching.model.module if distributed else flow_matching.model
        original_model = flow_matching.model
        flow_matching.model = model_to_save
        flow_matching.save_checkpoint(
            epoch=epoch + 1, optimizer=optimizer, loss=avg_epoch_loss["total_loss"],
            is_best=False, model_config=model_config, training_config=training_config,
        )
        flow_matching.model = original_model

        if cfg.validate_every > 0 and (epoch + 1) % cfg.validate_every == 0:
            val_losses = flow_matching.validate_on_loader(val_loader, max_batches=None)
            if distributed:
                for key in ("total_loss", "lattice_loss", "cart_loss"):
                    val_losses[key] = reduce_tensor(torch.tensor(val_losses[key], device=device)).item()

            if is_main_process():
                print(f"   Validation - Total: {val_losses['total_loss']:.6f}, "
                      f"Cart: {val_losses['cart_loss']:.6f}, Lattice: {val_losses['lattice_loss']:.6f}", flush=True)
                safe_wandb_log({
                    "val/total_loss": val_losses["total_loss"],
                    "val/cart_loss": val_losses["cart_loss"],
                    "val/lattice_loss": val_losses["lattice_loss"],
                }, global_step, cfg.use_wandb)

                if val_losses["total_loss"] < best_loss:
                    best_loss = val_losses["total_loss"]
                    model_state = (flow_matching.model.module if distributed else flow_matching.model).state_dict()
                    flow_matching.best_val_loss = best_loss
                    best_model_state = model_state.copy()
                    best_epoch = epoch + 1
                    print(f"   New best validation loss: {best_loss:.6f}")
                    original_model_best = flow_matching.model
                    flow_matching.model = model_to_save
                    flow_matching.save_checkpoint(
                        epoch=epoch + 1, optimizer=optimizer, loss=val_losses["total_loss"],
                        is_best=True, model_config=model_config, training_config=training_config,
                    )
                    flow_matching.model = original_model_best
                safe_wandb_log({"val/best_loss": best_loss}, global_step, cfg.use_wandb)

    if is_main_process() and best_model_state is not None:
        (flow_matching.model.module if distributed else flow_matching.model).load_state_dict(best_model_state)
        print(f"\nLoaded best model from epoch {best_epoch} (val loss {best_loss:.6f})")

    if is_main_process() and checkpoint_path:
        model_state_for_checkpoint = best_model_state if best_model_state is not None else (
            flow_matching.model.module if distributed else flow_matching.model).state_dict()
        torch.save({
            "model_state_dict": model_state_for_checkpoint,
            "model_config": model_config,
            "training_config": {k: training_config[k] for k in
                                ("lattice_loss_weight", "shared_time", "lr", "n_epochs", "batch_size", "num_workers")},
            "training_results": {
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "final_losses": {k: v[-1] for k, v in losses.items()},
                "loss_history": losses,
            },
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    return flow_matching, losses


def train(cfg: TrainConfig) -> CrystalFlowMatching:
    """Train a PackFlow base model from a :class:`TrainConfig`.

    Reproduces the paper's training run: builds the Cartesian transformer, trains
    with AdamW + gradient clipping, validates and checkpoints the best model.
    Returns the trained :class:`~packflow.models.model.CrystalFlowMatching`.
    """
    set_all_seeds(cfg.seed)
    check_and_fix_system_limits()

    device = f"cuda:{int(os.environ['LOCAL_RANK'])}" if cfg.distributed else resolve_device(cfg.device)
    print(f"Using device: {device}")

    run_dirs = create_run_dirs(cfg.experiments_root)
    checkpoints_root = str(run_dirs["checkpoints_dir"])
    if is_main_process():
        print(f"Experiment run directory: {run_dirs['run_dir']}")

    data_paths = load_data_splits(cfg.processed_data_dir)

    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = str(run_dirs["checkpoints_dir"] / "best_crystal_flow_model_cartesian_splits.pt")

    if cfg.use_wandb and is_main_process():
        try:
            import wandb

            os.environ.setdefault("WANDB_DIR", str(run_dirs["wandb_dir"]))
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_name or run_dirs["run_id"],
                       config=cfg.to_dict(), reinit=True, dir=str(run_dirs["wandb_dir"]))
        except ImportError:
            print("Warning: wandb requested but not installed.")
            cfg.use_wandb = False

    flow_matching = build_flow_matching(cfg, device)
    flow_matching, losses = _train_loop(cfg, flow_matching, data_paths, device,
                                        checkpoint_path, checkpoints_root)

    # Final test-set evaluation (same metric the original script reported).
    try:
        test_module = CrystalDataModule(train_path=data_paths["test"], batch_size=cfg.batch_size,
                                        num_workers=0, pin_memory=False)
        test_module.setup("fit")
        test_losses = flow_matching.validate_on_loader(test_module.train_dataloader(), max_batches=None)
        if is_main_process():
            print(f"Final test - Total: {test_losses['total_loss']:.6f}, "
                  f"Cart: {test_losses['cart_loss']:.6f}, Lattice: {test_losses['lattice_loss']:.6f}")
    except Exception as exc:  # pragma: no cover - test split optional
        print(f"Warning: final testing failed: {exc}")

    if cfg.use_wandb and is_main_process():
        import wandb

        wandb.finish()
    return flow_matching
