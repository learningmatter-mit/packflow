"""Configuration for PackFlow base-model (flow-matching) training.

``TrainConfig`` is a flat dataclass mirroring the hyper-parameters that produced
the paper checkpoints. Build it directly in Python, from a dict, or from parsed
CLI args (the ``packflow train`` command).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass
class TrainConfig:
    # --- data / run ---
    processed_data_dir: str = "processed_data"
    experiments_root: str = "experiments"
    checkpoint_path: Optional[str] = None
    device: str = "auto"
    seed: int = 42

    # --- optimisation ---
    batch_size: int = 32
    n_epochs: int = 1000
    lr: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    validate_every: int = 1
    log_every_n_batches: int = 5
    distributed: bool = False

    # --- model architecture (Cartesian transformer) ---
    coordinate_system: str = "cartesian"
    d_model: int = 256
    nhead: int = 8
    dim_feedforward: int = 512
    num_transformer_layers: int = 4
    use_positional_embeddings: bool = True
    use_rdkit_features: bool = False
    lattice_loss_weight: float = 0.1
    shared_time: bool = False

    # --- attention-bias options ---
    use_attention_bias_from_graph: bool = False
    attn_bias_heads: Optional[int] = None
    attn_bias_combine: str = "sum"
    attn_bias_baseline: float = 0.0
    bias_scale: float = 1.0
    learnable_bias_scale: bool = False

    # --- GNN-for-bonds (alternative to attention bias) ---
    use_gnn_for_bonds: bool = False
    num_gnn_layers: int = 3
    gat_heads: int = 4

    # --- timestep resampling ---
    use_logit_normal_resampling: bool = False
    logit_normal_m: float = -0.8
    logit_normal_s: float = 1.7
    logit_normal_mix: float = 0.02
    t_eps: float = 0.0
    fixed_time: Optional[float] = None

    # --- auxiliary losses ---
    use_smooth_lddt_loss: bool = False
    smooth_lddt_loss_weight: float = 1.0
    lddt_cutoff: float = 15.0
    lddt_weight_schedule: bool = False
    use_bond_length_loss: bool = False
    bond_length_loss_weight: float = 1.0
    use_periodic_lddt_loss: bool = False
    periodic_lddt_loss_weight: float = 1.0
    periodic_lddt_cutoff: Optional[float] = None
    periodic_lddt_warmup_epochs: int = 0

    # --- coordinate / periodic embedding ---
    lattice_token: bool = False
    periodic_coord_emb: bool = False
    periodic_nmax: int = 5
    periodic_topk: int = 512
    add_periodic_edges: bool = False
    periodic_edge_cutoff: float = 5.0
    periodic_edge_periodic: bool = True
    periodic_edge_time_cutoff: Optional[float] = 0.5
    use_fractional_coords: bool = False
    use_k_basis_representation: bool = False

    # --- logging ---
    grad_log_every: int = 0
    use_wandb: bool = False
    wandb_project: str = "molecular-crystals"
    wandb_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
