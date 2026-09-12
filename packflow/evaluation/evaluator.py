#!/usr/bin/env python3
"""
Test set evaluation script for crystal structure prediction models.
Loads a checkpoint, runs sampling at lambda=1, and evaluates crystal metrics on the test set.

Usage:
    python evaluate_test_set_metrics.py --checkpoint_path <path> --data_dir <path> [options]
"""

import torch
import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import subprocess
import tempfile
from pymatgen.core import Structure, Lattice, Molecule
from packflow.utils.crystal_utils import lattice_params_to_matrix, make_molecules_whole, lattice_params_to_matrix_torch, lattice_matrix_to_params_torch
from rdkit import Chem


# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def _uma_scripts_dir() -> str:
    """Directory holding the UMA / relaxation driver scripts.

    All UMA, relaxation, lattice-energy and hydrogen-addition code lives in the
    ``packflow.relaxation`` package (the single source of truth). These scripts
    are invoked via subprocess in the fairchem environment.
    """
    try:
        import packflow.relaxation as _rlx
        return os.path.dirname(os.path.abspath(_rlx.__file__))
    except Exception:
        # Fallback: relaxation package sitting next to the repo's evaluation/ dir.
        return os.path.join(str(project_root.parent), "packflow", "relaxation")


UMA_SCRIPTS_DIR = _uma_scripts_dir()

# Python interpreter for the fairchem/UMA environment (energy + relaxation).
# Set FAIRCHEM_PYTHON to the python inside your fairchem env; defaults to the
# current interpreter (works if fairchem is installed in the same environment).
FAIRCHEM_PYTHON = os.environ.get("FAIRCHEM_PYTHON", sys.executable)

# CSP Blind Test refcodes (Figure 5 case studies). Loaded from
# ``data/refcodes/blind_test.txt`` so the list stays in one place.
from packflow.config import load_refcodes as _load_refcodes
CSP_BLIND_TEST_REFCODES = _load_refcodes("blind_test.txt") or [
    "XAFPAY01",
    "OBEQOD",
]

from packflow.baselines.genarris_wrapper import GenarrisWrapper


from packflow.models.model import (
    CrystalFlowMatching,
    CrystalTransformerEncoder as CartesianCrystalTransformerEncoder,
)
from packflow.data.crystal_datamodule import CrystalDataModule
from packflow.evaluation.metrics import (
    compute_all_metrics_for_crystal,
    compute_metrics_batched,
    get_default_constants,
    mean_metrics,
)

class _JsonArrayAppender:
    """
    Incrementally maintain a *valid* top-level JSON array on disk.

    We keep the file ending with '\\n]\\n' at all times and insert new elements
    right before the closing bracket. This avoids a single huge `json.dump(...)`
    at the end for very large evaluations.
    """

    def __init__(self, path: str):
        self.path = path
        self._count = 0
        # Always keep the file a valid JSON list, even mid-run.
        with open(self.path, "wb") as f:
            f.write(b"[\n]\n")

    def append(self, obj) -> None:
        payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        with open(self.path, "r+b") as f:
            # Seek to before the closing `]\n` (we always maintain `\n]\n`)
            f.seek(-2, os.SEEK_END)
            if self._count > 0:
                f.write(b",\n")
            f.write(payload)
            f.write(b"\n]\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Best-effort only; some filesystems may not support fsync.
                pass
        self._count += 1


def set_deterministic_seed(seed: int = 42) -> None:
    """Set all random seeds for deterministic behavior."""
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms in PyTorch (if available)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass  # Some versions might not support this
    
    print(f"Deterministic seed set to: {seed}")


def get_model_config(model_type: str) -> Dict:
    """Get model configuration based on model type."""
    
    # Common default configuration (must mirror defaults in
    # train_crystal_flow_matching.py / CrystalTransformerEncoder)
    base_config = {
        'cart_coords_dim': 3,
        'lattice_dim': 6,
        'time_embed_dim': 64,
        'rdkit_bond_feat_dim': 2,
        'rdkit_node_feat_dim': 10,
        'activation': 'gelu',
        'norm_first': True,
        'bias': True,
        'dropout': 0.0,
        # Attention bias settings
        'attn_bias_combine': 'sum',
        'attn_bias_baseline': 0.0,
        'bias_scale': 1.0,
        'learnable_bias_scale': False,
        # Data / input feature flags
        'lattice_token': False,
        'periodic_coord_emb': False,
        'periodic_nmax': 5,
        'periodic_topk': 512,
        'use_fractional_coords': False,
        'use_rdkit_features': True,
        'use_positional_embeddings': False,
        'use_attention_bias_from_graph': True,
    }
    
    if model_type == "packflow-20M":
        config = base_config.copy()
        config.update({
            # From train_20M.sh
            'd_model': 320,
            'nhead': 8,
            'dim_feedforward': 1280,
            'num_layers': 12,
        })
        return config
        
    elif model_type == "packflow-2M":
        config = base_config.copy()
        config.update({
            # From train_2M.sh (uses defaults: d_model=256, nhead=8, dim_feedforward=512, num_transformer_layers=4)
            'd_model': 256,
            'nhead': 8,
            'dim_feedforward': 512,
            'num_layers': 4,
        })
        return config
        
    elif model_type == "packflow-80M":
        config = base_config.copy()
        config.update({
            # From train_80M.sh
            'd_model': 800,
            'nhead': 10,
            'dim_feedforward': 3072,
            'num_layers': 10,
        })
        return config
        
    elif model_type == "packflow-60M-no-bonds":
        config = base_config.copy()
        config.update({
            # From train_60M_no_bonds.sh
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
            # No bonds → no attention bias from graph
            'use_attention_bias_from_graph': False,
        })
        return config
        
    elif model_type == "packflow-60M-periodic-aux-loss-frac-input":
        config = base_config.copy()
        config.update({
            # From train_60M_periodic_aux_loss_frac_input.sh
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
            # Uses additional fractional coordinates as input
            'use_fractional_coords': True,
        })
        return config

    elif model_type == "packflow-60M":
        # From train_60M.sh
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-60M-2-nodes":
        # From train_60M_2_nodes.sh (same model, different distributed setup)
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-60M-bond-aux-loss":
        # From train_60M_bond_aux_loss.sh (same architecture, extra bond-length aux loss)
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type in ["packflow-60M-k-basis", "packflow-60M-k-basis-weight-10"]:
        # From train_60M_k_basis*.sh – architecture identical to base 60M;
        # use_k_basis_representation is handled by CrystalFlowMatching (from checkpoint config)
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-60M-periodic-aux-loss":
        # From train_60M_periodic_aux_loss.sh – periodic lDDT loss only affects training objective
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-60M-periodic-edges":
        # From train_60M_periodic_edges.sh – periodic edges affect data/graph construction during training,
        # not the Transformer architecture; encoder config is identical.
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-60M-shared-time":
        # From train_60M_shared_time.sh – shared_time handled by CrystalFlowMatching, not encoder
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-ddp":
        # DDP (Distributed Data Parallel) training setup - same architecture as base 60M
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-ddp-4n":
        # DDP (Distributed Data Parallel) training setup (4 nodes) - same architecture as base 60M
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
        })
        return config

    elif model_type == "packflow-60M-gat":
        # From train_60M_gat.sh - uses GAT-based bond encoding instead of attention bias
        config = base_config.copy()
        config.update({
            'd_model': 640,
            'nhead': 10,
            'dim_feedforward': 2560,
            'num_layers': 12,
            # GAT-based bond encoding (mutually exclusive with attention bias)
            'use_attention_bias_from_graph': False,
            'use_gnn_for_bonds': True,
            'num_gnn_layers': 3,
            'gat_heads': 4,
        })
        return config
        
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported types: "
            "packflow-20M, packflow-2M, packflow-60M, packflow-60M-2-nodes, "
            "packflow-60M-bond-aux-loss, packflow-60M-gat, packflow-60M-k-basis, "
            "packflow-60M-k-basis-weight-10, packflow-60M-no-bonds, "
            "packflow-60M-periodic-aux-loss, packflow-60M-periodic-aux-loss-frac-input, "
            "packflow-60M-periodic-edges, packflow-60M-shared-time, packflow-80M, "
            "packflow-ddp, packflow-ddp-4n"
        )


def get_default_checkpoint_path(model_type: str) -> str:
    """Get default checkpoint path based on model type.

    Resolution order:
      1. The curated package model-zoo: ``packflow/checkpoints/<model_type>/best_model.pt``.
      2. A raw training-run directory under ``$PACKFLOW_EXPERIMENTS_DIR`` (defaults
         to ``<repo>/experiments``) for ablation variants not shipped in the zoo.
    """
    # 1. Curated model-zoo shipped inside the package.
    try:
        import packflow
        pkg_dir = os.path.dirname(os.path.abspath(packflow.__file__))
        zoo_path = os.path.join(pkg_dir, "checkpoints", model_type, "best_model.pt")
        if os.path.exists(zoo_path):
            return zoo_path
    except Exception:
        pass

    # 2. Raw training-run directories (configurable; for ablation variants).
    experiments_dir = os.environ.get(
        "PACKFLOW_EXPERIMENTS_DIR", os.path.join(str(project_root.parent), "experiments")
    )

    # Map model types to their specific run folders
    run_folders = {
        "packflow-2M": "20251130-044730_packflow-2M_jid3797662",
        "packflow-20M": "20251130-044907_packflow-20M_jid3797673",
        "packflow-60M": "20251130-052224_packflow-60M_jid3797682",
        "packflow-80M": "20251130-052224_packflow-80M_jid3797683",
        "packflow-60M-no-bonds": "20251130-045022_packflow-60M-no-bonds_jid3797674",
        "packflow-60M-bond-aux-loss": "20251130-045333_packflow-60M-bond-aux-loss_jid3797723",
        "packflow-60M-periodic-aux-loss": "20251130-052224_packflow-60M-periodic-aux-loss-warmup_jid3797680",
        "packflow-60M-shared-time": "20251130-052224_packflow-60M-shared-time_jid3797681",
        "packflow-60M-k-basis": "20251130-054456_packflow-60M-k-basis_jid3799057",
        "packflow-60M-k-basis-weight-10": "20251130-220611_packflow-60M-k-basis-weight-10_jid3859294",
        "packflow-60M-periodic-edges": "20251130-231733_packflow-60M-periodic-edges_jid3863934",
        "packflow-60M-periodic-aux-loss-frac-input": "20251130-235335_packflow-60M-periodic-aux-loss-frac-input_jid3866115",
        "packflow-ddp": "20251201-011524_packflow-ddp_jid3871200",
        "packflow-ddp-4n": "20251209-212607_packflow-ddp-4n_jid3930959",
        "packflow-60M-gat": "20260103-175802_packflow-60M-gat_jid4006476",
    }
    
    if model_type in run_folders:
        return os.path.join(experiments_dir, run_folders[model_type], "checkpoints", "best_model.pt")
        
    # Fallback for unknown models - try to import from evaluate_with_structural_data
    try:
        from evaluate_with_structural_data import get_default_checkpoint_path as get_path
        return get_path(model_type)
    except:
        raise ValueError(f"Unknown model type: {model_type}. Please provide --checkpoint_path or use a supported model type.")


def load_pretrained_model(checkpoint_path: str, device: str = 'cpu',
                          schedule_type: Optional[str] = None, model_type: str = "different-cosmos") -> CrystalFlowMatching:
    """Load pretrained crystal flow matching model."""
    print(f"Loading pretrained model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration and training configuration from checkpoint
    checkpoint_model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    
    # Valid arguments for CartesianCrystalTransformerEncoder.__init__()
    valid_encoder_args = {
        'd_model', 'nhead', 'dim_feedforward', 'activation', 'dropout', 'norm_first', 'bias',
        'num_layers', 'time_embed_dim', 'cart_coords_dim', 'lattice_dim',
        'use_attention_bias_from_graph', 'attn_bias_heads', 'attn_bias_combine',
        'attn_bias_baseline', 'bias_scale', 'learnable_bias_scale',
        'use_positional_embeddings', 'use_rdkit_features', 'lattice_token',
        'rdkit_bond_feat_dim', 'rdkit_node_feat_dim',
        'periodic_coord_emb', 'periodic_nmax', 'periodic_topk',
        'use_fractional_coords',
        # GNN-based bond encoding parameters
        'use_gnn_for_bonds', 'num_gnn_layers', 'gat_heads', 'gat_concat', 'gat_dropout',
    }
    
    # Use checkpoint's model_config if available (source of truth), otherwise fall back to get_model_config
    if checkpoint_model_config:
        print(f"Using model configuration from checkpoint (source of truth)")
        # Filter checkpoint config to only include valid encoder arguments
        final_config = {k: v for k, v in checkpoint_model_config.items() if k in valid_encoder_args}
        # Log any filtered-out fields
        filtered_out = [k for k in checkpoint_model_config.keys() if k not in valid_encoder_args]
        if filtered_out:
            print(f"   Filtered out invalid encoder args from checkpoint: {filtered_out}")
        # Ensure all required fields are present (fill missing ones from get_model_config defaults)
        fallback_config = get_model_config(model_type)
        for k, v in fallback_config.items():
            if k not in final_config:
                final_config[k] = v
                print(f"   Added missing field '{k}': {v} (from get_model_config fallback)")
    else:
        print(f"WARNING: Checkpoint missing 'model_config', using get_model_config for model type: {model_type}")
        final_config = get_model_config(model_type)
    
    print(f"Creating CartesianCrystalTransformerEncoder with config:")
    for k, v in final_config.items():
        print(f"   {k}: {v}")
    
    # Create model using the correct class with the actual training arguments
    model = CartesianCrystalTransformerEncoder(**final_config)
    
    # Load model weights with flexible mapping
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"WARNING: Strict loading failed, trying flexible loading: {e}")
        try:
            # Try loading with strict=False to ignore missing/unexpected keys
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"   Missing keys (ignored): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"   Unexpected keys (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        except Exception as e2:
            print(f"ERROR: Flexible loading also failed: {e2}")
            raise e2
    
    # Create flow matching instance
    lattice_loss_weight = checkpoint.get('lattice_loss_weight', 1.0)
    shared_time = checkpoint.get('shared_time', False)
    
    # Flow-matching / training-time flags: default from training_config, with safe fallbacks
    # These match the arguments in CrystalFlowMatching.__init__ and train_crystal_flow_matching.py
    use_logit_normal_resampling = training_config.get('use_logit_normal_resampling', False)
    logit_normal_m = training_config.get('logit_normal_m', -0.8)
    logit_normal_s = training_config.get('logit_normal_s', 1.7)
    logit_normal_mix = training_config.get('logit_normal_mix', 0.02)
    t_eps = training_config.get('t_eps', 0.0)
    fixed_time = training_config.get('fixed_time', None)
    use_smooth_lddt_loss = training_config.get('use_smooth_lddt_loss', False)
    smooth_lddt_loss_weight = training_config.get('smooth_lddt_loss_weight', 1.0)
    lddt_cutoff = training_config.get('lddt_cutoff', 15.0)
    lddt_weight_schedule = training_config.get('lddt_weight_schedule', False)
    grad_log_every = training_config.get('grad_log_every', 0)
    use_bond_length_loss = training_config.get('use_bond_length_loss', False)
    bond_length_loss_weight = training_config.get('bond_length_loss_weight', 1.0)
    add_periodic_edges = training_config.get('add_periodic_edges', False)
    periodic_edge_cutoff = training_config.get('periodic_edge_cutoff', 5.0)
    periodic_edge_periodic = training_config.get('periodic_edge_periodic', True)
    periodic_edge_time_cutoff = training_config.get('periodic_edge_time_cutoff', 0.5)
    use_periodic_lddt_loss = training_config.get('use_periodic_lddt_loss', False)
    periodic_lddt_loss_weight = training_config.get('periodic_lddt_loss_weight', 1.0)
    periodic_lddt_cutoff = training_config.get('periodic_lddt_cutoff', None)
    periodic_lddt_warmup_epochs = training_config.get('periodic_lddt_warmup_epochs', 0)
    use_k_basis_representation = training_config.get('use_k_basis_representation', False)
    
    # From checkpoint if present; override if explicitly provided
    ckpt_schedule = checkpoint.get('schedule_type', 'ot')
    chosen_schedule = schedule_type if schedule_type is not None else ckpt_schedule
    
    flow_matching = CrystalFlowMatching(
        model=model, 
        device=device,
        lattice_loss_weight=lattice_loss_weight,
        shared_time=shared_time,
        use_logit_normal_resampling=use_logit_normal_resampling,
        logit_normal_m=logit_normal_m,
        logit_normal_s=logit_normal_s,
        logit_normal_mix=logit_normal_mix,
        t_eps=t_eps,
        fixed_time=fixed_time,
        use_smooth_lddt_loss=use_smooth_lddt_loss,
        smooth_lddt_loss_weight=smooth_lddt_loss_weight,
        lddt_cutoff=lddt_cutoff,
        lddt_weight_schedule=lddt_weight_schedule,
        grad_log_every=grad_log_every,
        use_bond_length_loss=use_bond_length_loss,
        bond_length_loss_weight=bond_length_loss_weight,
        add_periodic_edges=add_periodic_edges,
        periodic_edge_cutoff=periodic_edge_cutoff,
        periodic_edge_periodic=periodic_edge_periodic,
        periodic_edge_time_cutoff=periodic_edge_time_cutoff,
        use_periodic_lddt_loss=use_periodic_lddt_loss,
        periodic_lddt_loss_weight=periodic_lddt_loss_weight,
        periodic_lddt_cutoff=periodic_lddt_cutoff,
        periodic_lddt_warmup_epochs=periodic_lddt_warmup_epochs,
        use_k_basis_representation=use_k_basis_representation,
    )
    # Store schedule info for reference (not currently used by CrystalFlowMatching)
    flow_matching.schedule_type = chosen_schedule
    
    print(f"Model loaded successfully! Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   - Device: {device}")
    print(f"   - Lattice loss weight: {lattice_loss_weight}")
    print(f"   - Shared time: {shared_time}")
    print(f"   - Schedule: {chosen_schedule}")
    
    return flow_matching


def load_test_data(data_dir: str, batch_size: int = 1, max_samples: Optional[int] = None, refcode_filter: Optional[List[str]] = None) -> CrystalDataModule:
    """Load test dataset.
    
    Args:
        data_dir: Directory containing test.pt
        batch_size: Batch size for data loading
        max_samples: Maximum number of samples to load
        refcode_filter: List of refcodes to include (if None, loads all)
    """
    test_path = os.path.join(data_dir, "test.pt")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")
    
    print(f"Loading test data from: {test_path}")
    if refcode_filter:
        print(f"Filtering to {len(refcode_filter)} specific refcodes: {refcode_filter}")
    
    data_module = CrystalDataModule(
        test_path=test_path,
        batch_size=batch_size,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,
        max_samples=max_samples,
        refcode_filter=refcode_filter
    )
    
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    if max_samples:
        print(f"Limited to {max_samples} samples")
    
    return data_module, test_loader


def sample_crystal_with_lambda(flow_matching: CrystalFlowMatching, crystal_data: Dict[str, torch.Tensor], 
                              n_steps: int = 500, lambda_val: float = 1.0,
                              coords_time_grid_type: str = "linear", lattice_time_grid_type: str = "linear",
                              num_seeds: int = 1, seed_batch_size: Optional[int] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample multiple seeds for a crystal structure in parallel (batched).
    Returns a list of (cart_coords, lattice) tuples.
    
    Args:
        flow_matching: The flow matching model
        crystal_data: Dictionary containing crystal data
        n_steps: Number of sampling steps
        lambda_val: Sampling temperature
        coords_time_grid_type: Time grid type for coordinates
        lattice_time_grid_type: Time grid type for lattice
        num_seeds: Total number of seeds to sample
        seed_batch_size: Number of seeds to process per batch. If None, processes all at once.
    """
    # If seed_batch_size not specified or larger than num_seeds, process all at once
    if seed_batch_size is None or seed_batch_size >= num_seeds:
        return _sample_crystal_batch(flow_matching, crystal_data, n_steps, lambda_val,
                                     coords_time_grid_type, lattice_time_grid_type, num_seeds)
    
    # Process in batches
    all_results = []
    for batch_start in range(0, num_seeds, seed_batch_size):
        batch_end = min(batch_start + seed_batch_size, num_seeds)
        current_batch_size = batch_end - batch_start
        
        batch_results = _sample_crystal_batch(flow_matching, crystal_data, n_steps, lambda_val,
                                              coords_time_grid_type, lattice_time_grid_type, current_batch_size)
        all_results.extend(batch_results)
        
        # Clear GPU cache between batches to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_results


def _sample_crystal_batch(flow_matching: CrystalFlowMatching, crystal_data: Dict[str, torch.Tensor], 
                          n_steps: int, lambda_val: float,
                          coords_time_grid_type: str, lattice_time_grid_type: str,
                          num_seeds: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Internal function to sample a batch of seeds for a crystal structure.
    Returns a list of (cart_coords, lattice) tuples.
    """
    try:
        # Ensure model is in eval mode for deterministic sampling
        flow_matching.model.eval()
        
        # Create a batch of crystal data
        # We need to use PyG Batch to handle edge_index and batch vector correctly
        from torch_geometric.data import Data, Batch
        
        # Convert dictionary back to Data object to use Batch.from_data_list
        data_obj = Data(
            atom_types=crystal_data['atom_types'],
            edge_index=crystal_data['edge_index'],
            num_nodes=len(crystal_data['atom_types'])
        )
        if 'node_features' in crystal_data and crystal_data['node_features'] is not None:
             data_obj.node_features = crystal_data['node_features']
        if 'bond_features' in crystal_data and crystal_data['bond_features'] is not None:
             data_obj.bond_features = crystal_data['bond_features']
             
        # Replicate for num_seeds
        data_list = [data_obj] * num_seeds
        batch_data = Batch.from_data_list(data_list)
        
        # Move to device
        device = crystal_data['atom_types'].device
        batch_data = batch_data.to(device)
        
        # Convert back to dict for sample method
        batch_dict = {
            'atom_types': batch_data.atom_types,
            'edge_index': batch_data.edge_index,
            'batch': batch_data.batch,
            # Add optional features if present
        }
        if hasattr(batch_data, 'node_features'):
             batch_dict['node_features'] = batch_data.node_features
        if hasattr(batch_data, 'bond_features'):
             batch_dict['bond_features'] = batch_data.bond_features
        
        with torch.no_grad():
            batched_cart_coords, batched_lattice = flow_matching.sample(
                batch_dict, 
                n_steps=n_steps, 
                low_temperature_lambda=lambda_val,
                coords_time_grid_type=coords_time_grid_type,
                lattice_time_grid_type=lattice_time_grid_type
            )
            
        # Unbatch results
        results = []
        # batched_lattice is (B, 6) or (B, 3, 3) dep on output - sample returns lattice params for now?
        # sample returns cart_coords (N_total, 3) and lattice_state (B, 6) or (B, 3, 3)
        # Checking sample method return: 
        # final_lattice_constrained (B, 6)
        # sampled_cart_coords (N, 3)
        
        batch_vector = batch_data.batch
        for i in range(num_seeds):
            mask = (batch_vector == i)
            coords = batched_cart_coords[mask]
            lattice = batched_lattice[i]
            results.append((coords, lattice))
            
        return results
        
    except Exception as e:
        print(f"WARNING: Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return [(None, None)] * num_seeds


def compute_uma_metrics_single_seed(pred_coords: torch.Tensor, pred_lattice: torch.Tensor, 
                                  atom_types: torch.Tensor, device: str = 'cuda', uma_relaxation_steps: int = 1000) -> Dict[str, float]:
    """
    Compute UMA metrics (relaxation steps, lattice energy) for a single predicted crystal.
    Uses an external script `calculate_uma_metrics.py` running in the fairchem environment.
    """
    uma_metrics = {
        'relaxation_steps': None,
        'lattice_energy': None,
        'crystal_energy_total': None,
        'molecule_energy': None,
        'z_value': None
    }
    
    # Path to the UMA script
    script_path = os.path.join(UMA_SCRIPTS_DIR, "workers", "uma_metrics.py")
    if not os.path.exists(script_path):
        print(f"WARNING: UMA script not found at {script_path}")
        return uma_metrics

    # Create temporary directory for structure files
    with tempfile.TemporaryDirectory() as temp_dir:
        crystal_path = os.path.join(temp_dir, "temp_crystal.cif")
        molecule_path = os.path.join(temp_dir, "temp_molecule.xyz")
        relaxed_crystal_path = os.path.join(temp_dir, "relaxed_crystal.cif")
        
        try:
            # Convert tensors to numpy
            coords_np = pred_coords.cpu().numpy()
            lattice_params = pred_lattice.cpu().numpy()
            atoms_np = atom_types.cpu().numpy()
            
            # 1. Create Pymatgen Structure (Crystal)
            # Lattice from parameters
            lat = Lattice.from_parameters(*lattice_params)
            
            # Map atomic numbers to species
            from pymatgen.core.periodic_table import Element
            species = [Element.from_Z(int(z)) for z in atoms_np]
            
            # Create structure (assuming coords are Cartesian)
            struct = Structure(lat, species, coords_np, coords_are_cartesian=True)
            struct.to(filename=crystal_path)
            
            # 2. Create Pymatgen Molecule (Isolated)
            # Just take species and coords
            mol = Molecule(species, coords_np)
            mol.to(filename=molecule_path)
            
            # 3. Call External Script
            # We assume 'conda run -n fairchem' works or 'source activate'
            # Let's try constructing a command that sources the environment
            # This is fragile but standard for SLURM/bash environments

            # Use 'source activate fairchem' logic similar to test script
            # Use absolute path to python in fairchem environment to avoid activation issues
            fairchem_python = FAIRCHEM_PYTHON
            if not os.path.exists(fairchem_python):
                 # Fallback/Error if not found, though we verified it exists
                 print(f"   WARNING: fairchem python not found at {fairchem_python}")
                 return uma_metrics

            # Pass the output path for relaxed crystal
            full_cmd = [
                fairchem_python, script_path,
                "--crystal", crystal_path,
                "--molecule", molecule_path,
                "--device", str(device),
                "--relaxation_steps", str(uma_relaxation_steps)
            ]
            print(f"DEBUG: Running UMA command: {full_cmd}", flush=True)

            # Let's execute the command directly without bash -c wrapper for simplicity and robustness
            result = subprocess.run(full_cmd, capture_output=True, text=True)

            # Always print DEBUG lines from stderr
            if result.stderr:
                debug_lines = [l for l in result.stderr.splitlines() if "DEBUG" in l]
                for l in debug_lines:
                    print(f"   {l}")

            if result.returncode != 0:
                print(f"   WARNING: UMA script failed (code {result.returncode})")
                print(f"   STDERR: {result.stderr}") 
                return uma_metrics

            # Parse JSON output (last line)
            lines = result.stdout.strip().splitlines()
            if not lines:
                 print("   WARNING: UMA script produced no output")
                 return uma_metrics

            try:
                metrics_json = json.loads(lines[-1])
            except json.JSONDecodeError:
                print(f"   WARNING: Could not decode JSON from UMA script output: {lines[-1]}")
                return uma_metrics

            if metrics_json.get("error"):
                print(f"   WARNING: UMA script reported error: {metrics_json['error']}")
                return uma_metrics

            # Update metrics
            # The script returns {"single_entry": {metrics...}} when run in single mode
            if "single_entry" in metrics_json:
                uma_metrics.update(metrics_json["single_entry"])
            else:
                # Fallback or batch mode behavior (though this function is for single seed)
                uma_metrics.update(metrics_json)
            
            # If relaxation was enabled, ensure we have the trajectory
            if uma_relaxation_steps > 0:
                 if 'relaxation_trajectory' not in uma_metrics:
                      # Maybe script failed significantly?
                      pass

            # Reconstruct relaxed structure from JSON data if available (preserves atom order)
            if "relaxed_frac_coords" in metrics_json and "relaxed_lattice_matrix" in metrics_json:
                try:
                    relaxed_frac_coords = metrics_json["relaxed_frac_coords"]
                    relaxed_lattice_matrix = metrics_json["relaxed_lattice_matrix"]
                    
                    # Create Lattice from matrix
                    relaxed_lat = Lattice(relaxed_lattice_matrix)
                    
                    # Create Structure using ORIGINAL species to guarantee order
                    relaxed_struct = Structure(relaxed_lat, species, relaxed_frac_coords, coords_are_cartesian=False)
                    uma_metrics['relaxed_crystal_structure'] = relaxed_struct
                    
                    # Verify atom count just in case
                    if len(relaxed_struct) != len(species):
                         print(f"   WARNING: Constructed relaxed structure has {len(relaxed_struct)} atoms, expected {len(species)}")
                         
                except Exception as e:
                    print(f"   WARNING: Failed to reconstruct relaxed structure from JSON: {e}")
                    # Fallback to file load
                    if os.path.exists(relaxed_crystal_path):
                        try:
                            relaxed_struct = Structure.from_file(relaxed_crystal_path)
                            uma_metrics['relaxed_crystal_structure'] = relaxed_struct
                        except Exception as e2:
                             print(f"   WARNING: Failed to load relaxed crystal from file fallback: {e2}")

            # Fallback: Load relaxed crystal from file if JSON data missing
            elif os.path.exists(relaxed_crystal_path):
                try:
                    relaxed_struct = Structure.from_file(relaxed_crystal_path)
                    uma_metrics['relaxed_crystal_structure'] = relaxed_struct
                    
                    # Verify atom order (since we fell back to file)
                    if len(relaxed_struct) != len(species):
                         print(f"   WARNING: Relaxed structure has {len(relaxed_struct)} atoms, expected {len(species)}")
                    else:
                        match_count = sum([1 for i in range(len(species)) if relaxed_struct.species[i] == species[i]])
                        if match_count != len(species):
                             print(f"   WARNING: Atom order mismatch in file load! Only {match_count}/{len(species)} atoms match species.")
                except Exception as e:
                    print(f"   WARNING: Failed to load relaxed crystal: {e}")

        except Exception as e:
            print(f"   WARNING: Failed to compute UMA metrics: {e}")
            import traceback
            traceback.print_exc()
            
    return uma_metrics


def evaluate_single_crystal(flow_matching, crystal_data_obj, 
                           atomic_masses: Dict[int, float], covalent_radii: Dict[int, float],
                           n_steps: int = 500, 
                           visualize: bool = False, output_dir: str = None, 
                           crystal_idx: int = 0, num_seeds: int = 1, lambda_val: float = 1.0,
                           compute_uma_metrics: bool = False,
                           uma_relaxation_steps: int = 0, defer_uma_execution: bool = False, uma_staging_dir: str = None,
                           device: Optional[str] = None, seed_batch_size: Optional[int] = None) -> Optional[Dict]:
    """Evaluate metrics for a single crystal structure with multiple seeds, average metrics across seeds, and return all data."""
    
    # Get device from flow_matching model or default or argument
    if device is None:
        if hasattr(flow_matching, 'model'):
            device = next(flow_matching.model.parameters()).device
        else:
            # Genarris wrapper or other mock
            device = getattr(flow_matching, 'device', 'cpu')
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
    else:
        # Use provided device
        device = torch.device(device)

    
    # Convert to dictionary format and move to correct device
    crystal_data = {
        'whole_cartesian_coords': crystal_data_obj.whole_cartesian_coords.to(device),
        'atom_types': crystal_data_obj.atom_types.to(device),
        'edge_index': crystal_data_obj.edge_index.to(device),
        'lattice_1': crystal_data_obj.lattice_1.to(device),
        # Optional features used by the Cartesian model
        'node_features': getattr(crystal_data_obj, 'node_features', None),
        'bond_features': getattr(crystal_data_obj, 'bond_features', None),
        'refcode': getattr(crystal_data_obj, 'refcode', 'Unknown'),
        'smiles': getattr(crystal_data_obj, 'smiles', None)
    }
    # Move optional features to device if present
    if crystal_data['node_features'] is not None:
        crystal_data['node_features'] = crystal_data['node_features'].to(device)
    if crystal_data['bond_features'] is not None:
        crystal_data['bond_features'] = crystal_data['bond_features'].to(device)
    
    refcode = crystal_data['refcode']
    
    # Get true data (already on correct device)
    true_coords = crystal_data['whole_cartesian_coords']
    true_lattice = crystal_data['lattice_1']
    atomic_numbers = crystal_data['atom_types']
    edge_index = crystal_data['edge_index']
    
    # Store all seeds' data
    all_seeds_data = []
    successful_samples = 0

    # Pre-calculate Ground Truth UMA metrics (if enabled)
    # GT will be included in the batch manifest and relaxed along with predictions
    gt_uma_metrics = {}
    # Note: GT relaxation is now done in batch with predictions (see below)
    
    # 1. Batched Sampling
    # Generate all seeds in parallel
    t_start_sample = time.time()
    try:
        # Check if using Genarris wrapper
        if hasattr(flow_matching, 'sample') and isinstance(flow_matching, GenarrisWrapper):
            # Genarris wrapper handles num_seeds internally
            samples_list = flow_matching.sample(crystal_data, num_seeds=num_seeds)
        else:
            # Flow matching model
            samples_list = sample_crystal_with_lambda(
                flow_matching, crystal_data, n_steps, lambda_val, num_seeds=num_seeds,
                seed_batch_size=seed_batch_size
            )
    except Exception as e:
        print(f"Failed to sample crystal {refcode}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    t_end_sample = time.time()
    inference_time = t_end_sample - t_start_sample
    
    # Check for Genarris normalized time override
    if hasattr(flow_matching, 'sample') and isinstance(flow_matching, GenarrisWrapper):
        if hasattr(flow_matching, 'last_inference_time'):
            inference_time = flow_matching.last_inference_time
            
    inference_time_per_seed = inference_time / max(1, num_seeds)


    all_seeds_data = []
    successful_samples = 0
    
    # Prepare Context Manager for Temp Dir (if needed)
    # If deferring, we MUST use a persistent uma_staging_dir provided by caller.
    # If not deferring, we use a temp dir that autocloses.
    
    class DummyContext:
        def __enter__(self): 
            if uma_staging_dir:
                os.makedirs(uma_staging_dir, exist_ok=True)
            return uma_staging_dir
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    cm = DummyContext() if (defer_uma_execution and uma_staging_dir) else tempfile.TemporaryDirectory()
        
    with cm as temp_dir:
        uma_manifest = []
        
        # Add Ground Truth to UMA manifest (will be relaxed once per crystal, not per seed)
        gt_entry = None
        if compute_uma_metrics:
            gt_id = f"{refcode}_gt"
            gt_crystal_path = os.path.join(temp_dir, f"{gt_id}_in.cif")
            gt_molecule_path = os.path.join(temp_dir, f"{gt_id}_mol.xyz")
            
            # Create GT CIF and XYZ files
            gt_lattice_mat = lattice_params_to_matrix_torch(true_lattice).cpu().numpy()
            gt_coords_np = true_coords.cpu().numpy()
            gt_atoms_np = atomic_numbers.cpu().numpy()
            
            species = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in gt_atoms_np]
            if gt_lattice_mat.shape == (3, 3):
                try:
                    gt_lat = Lattice(gt_lattice_mat)
                except:
                    gt_lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
            else:
                gt_lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
            
            # Wrap coords for file saving
            inv_matrix = np.linalg.inv(gt_lat.matrix)
            gt_frac_coords = gt_coords_np @ inv_matrix
            gt_wrapped_frac_coords = gt_frac_coords % 1.0
            
            gt_struc = Structure(gt_lat, species, gt_wrapped_frac_coords, coords_are_cartesian=False)
            gt_struc.to(filename=gt_crystal_path)
            
            gt_mol = Molecule(species, gt_coords_np)
            gt_mol.to(filename=gt_molecule_path)
            
            gt_entry = {
                "id": gt_id,
                "crystal_path": gt_crystal_path,
                "molecule_path": gt_molecule_path,
                "refcode": refcode,
                "is_ground_truth": True,
                "relaxation_steps": uma_relaxation_steps  # Relax GT with same steps as predictions
            }
            uma_manifest.append(gt_entry)
        
        for seed_idx, sample_output in enumerate(samples_list):
            pred_atom_types = None
            pred_edge_index = None  # For Genarris bond visualization
            
            if len(sample_output) == 4:
                # Genarris: (coords, lattice, atom_types, edge_index)
                pred_coords, pred_lattice, pred_atom_types_tensor, pred_edge_index = sample_output
                if pred_atom_types_tensor is not None:
                     pred_atom_types = pred_atom_types_tensor.cpu().numpy()
            elif len(sample_output) == 3:
                pred_coords, pred_lattice, pred_atom_types_tensor = sample_output
                if pred_atom_types_tensor is not None:
                     pred_atom_types = pred_atom_types_tensor.cpu().numpy()
            else:
                pred_coords, pred_lattice = sample_output
                
            if pred_coords is None or pred_lattice is None:
                print(f"   WARNING: Seed {seed_idx+1} failed for {refcode}")
                continue
            
            try:
                # Compute metrics for this seed
                # Identify if model is Genarris (to skip contact metrics)
                is_genarris = False
                if isinstance(flow_matching, GenarrisWrapper):
                    is_genarris = True
                elif hasattr(flow_matching, 'mode') and "genarris" in str(flow_matching.mode).lower(): # Fallback check
                    is_genarris = True
                
                # Compute metrics for this seed
                # If Genarris, set contact_cutoff to negative/invalid or catch later?
                # compute_all_metrics_for_crystal takes contact_cutoff. 
                # If we pass contact_cutoff < 0, maybe we can hack it? 
                # Better to just modify the call or arguments.
                
                # For Genarris, use pred_atom_types (reordered); for others, use ground truth atomic_numbers
                if pred_atom_types is not None:
                    metrics_atom_types = torch.tensor(pred_atom_types, dtype=torch.long, device=pred_coords.device)
                else:
                    metrics_atom_types = atomic_numbers
                
                metrics = compute_all_metrics_for_crystal(
                    pred_coords=pred_coords,
                    pred_lattice=pred_lattice,
                    true_coords=true_coords,
                    true_lattice=true_lattice,
                    atomic_numbers=metrics_atom_types,
                    edge_index=edge_index,
                    atomic_masses_g_mol=atomic_masses,
                    covalent_radii_A=covalent_radii,
                    contact_cutoff= -1.0 if is_genarris else 6.0 # Disable contact metrics for Genarris
                )
                
                
                # Add inference time and resource cost to metrics
                metrics['inference_time_s'] = inference_time_per_seed
                
                # Determine resources
                if is_genarris:
                    resource_type = 'cpu_cores'
                    resource_count = flow_matching.n_procs if hasattr(flow_matching, 'n_procs') else 1
                else:
                    if torch.cuda.is_available() and device.type == 'cuda':
                        resource_type = 'gpu'
                        resource_count = torch.cuda.device_count()
                        # If we assume 1 GPU for this specific process in distributed, might need adjustment
                        # But typically evaluating on 1 GPU.
                        resource_count = 1 
                    else:
                        resource_type = 'cpu_cores'
                        resource_count = 1 # Serial evaluation unless specified otherwise

                metrics['resource_type'] = resource_type
                metrics['resource_count'] = resource_count
                metrics['inference_resource_time_s'] = inference_time_per_seed * resource_count


                
                # Prepare UMA input if requested
                entry = None
                if compute_uma_metrics:
                    # Compute UMA metrics
                    # BATCH OPTIMIZATION: Skip single seed computation
                    # We will run all seeds in a batch at the end of the loop
                    pred_uma = {} 
                    
                    seed_id = f"{refcode}_seed_{seed_idx}" # Restore seed_id definition
                    
                    # Store raw predicted UMA metrics
                    # Unrelaxed
                    metrics['pred_unrelaxed_energy'] = pred_uma.get('unrelaxed_energy')
                    metrics['pred_unrelaxed_max_force'] = pred_uma.get('unrelaxed_max_force')
                    metrics['pred_unrelaxed_mean_force'] = pred_uma.get('unrelaxed_mean_force')
                    
                    # Relaxed (if steps > 0)
                    if uma_relaxation_steps > 0:
                         metrics['pred_relaxed_energy'] = pred_uma.get('relaxed_energy')
                         metrics['pred_relaxed_max_force'] = pred_uma.get('relaxed_max_force')
                         metrics['pred_relaxed_mean_force'] = pred_uma.get('relaxed_mean_force')
                         metrics['relaxation_trajectory'] = pred_uma.get('relaxation_trajectory')
                         # Also copy actual relaxation steps taken (for verification)
                         metrics['relaxation_steps'] = pred_uma.get('relaxation_steps', 0)
                         
                         # Calculate Residuals (Relaxed Pred - Unrelaxed GT)
                         # Note: gt_uma_metrics['unrelaxed_energy'] is the reference
                         if pred_uma.get('relaxed_energy') is not None and gt_uma_metrics.get('unrelaxed_energy') is not None:
                             metrics['residual_relaxed_energy'] = pred_uma['relaxed_energy'] - gt_uma_metrics['unrelaxed_energy']
                         
                         if pred_uma.get('relaxed_max_force') is not None and gt_uma_metrics.get('unrelaxed_max_force') is not None:
                             metrics['residual_relaxed_max_force'] = pred_uma['relaxed_max_force'] - gt_uma_metrics['unrelaxed_max_force']

                         if pred_uma.get('relaxed_mean_force') is not None and gt_uma_metrics.get('unrelaxed_mean_force') is not None:
                             metrics['residual_relaxed_mean_force'] = pred_uma['relaxed_mean_force'] - gt_uma_metrics['unrelaxed_mean_force']

                    # Residuals (Unrelaxed Pred - Unrelaxed GT)
                    if pred_uma.get('unrelaxed_energy') is not None and gt_uma_metrics.get('unrelaxed_energy') is not None:
                         metrics['residual_unrelaxed_energy'] = pred_uma['unrelaxed_energy'] - gt_uma_metrics['unrelaxed_energy']

                    if pred_uma.get('unrelaxed_max_force') is not None and gt_uma_metrics.get('unrelaxed_max_force') is not None:
                         metrics['residual_unrelaxed_max_force'] = pred_uma['unrelaxed_max_force'] - gt_uma_metrics['unrelaxed_max_force']

                    if pred_uma.get('unrelaxed_mean_force') is not None and gt_uma_metrics.get('unrelaxed_mean_force') is not None:
                         metrics['residual_unrelaxed_mean_force'] = pred_uma['unrelaxed_mean_force'] - gt_uma_metrics['unrelaxed_mean_force']

                    if pred_uma.get('error'):
                        metrics['uma_error'] = pred_uma['error']
                    
                    # Legacy/Deferred Logic removed/skipped for now as we are doing immediate execution
                    relaxed_crystal_path = os.path.join(temp_dir, f"{seed_id}_relaxed.cif")
                    crystal_path = os.path.join(temp_dir, f"{seed_id}_in.cif")
                    molecule_path = os.path.join(temp_dir, f"{seed_id}_mol.xyz")
                    
                    # Create Pymatgen Structure and Molecule
                    # 1. Structure
                    pred_lattice_mat = lattice_params_to_matrix_torch(pred_lattice).cpu().numpy()
                    coords_np = pred_coords.cpu().numpy()
                    # For Genarris, use pred_atom_types (reordered); for others, use ground truth atomic_numbers
                    if pred_atom_types is not None:
                        species = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in pred_atom_types]
                    else:
                        species = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in atomic_numbers.cpu().numpy()]
                    if pred_lattice_mat.shape == (3, 3):
                         try:
                             lat = Lattice(pred_lattice_mat)
                         except:
                             # Fallback for collapsed lattice
                             lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)
                    else:
                         lat = Lattice.from_parameters(10, 10, 10, 90, 90, 90)

                    # Wrap coords for file saving (important for UMA)
                    # Convert Cartesian to Fractional
                    inv_matrix = np.linalg.inv(lat.matrix)
                    frac_coords = coords_np @ inv_matrix
                    wrapped_frac_coords = frac_coords % 1.0
                    
                    struc = Structure(lat, species, wrapped_frac_coords, coords_are_cartesian=False)
                    struc.to(filename=crystal_path)
                    
                    # 2. Molecule
                    mol = Molecule(species, coords_np)
                    mol.to(filename=molecule_path)
                    
                    entry = {
                        "id": seed_id, # Updated to use unique ID
                        "crystal_path": crystal_path,
                        "molecule_path": molecule_path,
                        # Pass metadata to help matching back later if needed, though ID is enough
                        "refcode": refcode,
                        "seed_idx": seed_idx,
                        "relaxation_steps": uma_relaxation_steps  # Ensure batch run knows about relaxation steps
                    }
                    
                    # Always add to manifest for batch execution
                    uma_manifest.append(entry)

                # Compute lattice matrix for predicted structure
                pred_lattice_matrix = lattice_params_to_matrix_torch(pred_lattice)

                seed_data = {
                    'seed_idx': seed_idx,
                    'refcode': refcode, # Add refcode as requested
                    'metrics': metrics.copy(),
                    # Predicted structure (unrelaxed) - coordinates and lattice
                    'pred_coords': pred_coords.clone(),
                    'pred_lattice': pred_lattice.clone(),
                    'pred_lattice_matrix': pred_lattice_matrix.cpu().clone(),
                    'pred_atom_types': pred_atom_types, # Store explicit atom types if available (Genarris reorders)
                    'pred_edge_index': pred_edge_index.clone() if pred_edge_index is not None else None,  # For Genarris (different from GT)
                    # Ground truth structure info (same for all seeds)
                    'gt_coords': true_coords.cpu().clone(),
                    'gt_lattice': true_lattice.cpu().clone(),
                    'gt_lattice_matrix': lattice_params_to_matrix_torch(true_lattice).cpu().clone(),
                    'gt_atom_types': atomic_numbers.cpu().clone(),
                    'gt_edge_index': edge_index.cpu().clone(),
                    # Relaxed structures (will be populated after UMA batch execution)
                    'pred_relaxed_coords': None,
                    'pred_relaxed_lattice_matrix': None,
                    'gt_relaxed_coords': None,
                    'gt_relaxed_lattice_matrix': None,
                    # Legacy fields
                    'atomic_numbers': atomic_numbers.cpu().clone(), # Keep for backward compatibility
                    'relaxed_structure': None,
                    'gt_uma_metrics': gt_uma_metrics, # Store GT metrics for residual calculation (will be updated)
                    'uma_pending_entry': entry if (compute_uma_metrics and defer_uma_execution) else None
                }
                all_seeds_data.append(seed_data)
                successful_samples += 1
                
            except Exception as e:
                print(f"   WARNING: Seed {seed_idx+1} metrics computation failed for {refcode}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # --- Batched UMA Execution (Immediate Mode) ---
        # This batch includes GT (1 entry) + all predictions (num_seeds entries)
        if compute_uma_metrics and uma_manifest and not defer_uma_execution:
            manifest_path = os.path.join(temp_dir, "manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(uma_manifest, f)
            
            script_path = os.path.join(UMA_SCRIPTS_DIR, "workers", "uma_metrics.py")
            fairchem_python = FAIRCHEM_PYTHON
            
            if os.path.exists(fairchem_python) and os.path.exists(script_path):
                print(f"   Running UMA batch execution for {len(uma_manifest)} structures (1 GT + {len(uma_manifest)-1} seeds)...")
                cmd = [fairchem_python, script_path, "--batch_manifest", manifest_path, "--device", "cuda" if torch.cuda.is_available() else "cpu"]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Print progress lines from stderr that were captured
                    if result.stderr:
                         progress_lines = [l for l in result.stderr.splitlines() if "Processing" in l or "DEBUG" in l]
                         for l in progress_lines:
                             print(f"   [UMA] {l}")
                    if result.returncode == 0:
                        lines = result.stdout.strip().splitlines()
                        if lines:
                            try:
                                batch_results = json.loads(lines[-1])
                                
                                # First, extract GT results
                                gt_id = f"{refcode}_gt"
                                if gt_id in batch_results:
                                    gt_res = batch_results[gt_id]
                                    gt_uma_metrics = {
                                        # Unrelaxed GT metrics
                                        'unrelaxed_energy': gt_res.get('unrelaxed_energy'),
                                        'unrelaxed_max_force': gt_res.get('unrelaxed_max_force'),
                                        'unrelaxed_mean_force': gt_res.get('unrelaxed_mean_force'),
                                        'unrelaxed_stress_voigt': gt_res.get('unrelaxed_stress_voigt'),
                                        'unrelaxed_stress_matrix': gt_res.get('unrelaxed_stress_matrix'),
                                        'unrelaxed_pressure': gt_res.get('unrelaxed_pressure'),
                                        'unrelaxed_lattice_matrix': gt_res.get('unrelaxed_lattice_matrix'),
                                        'unrelaxed_frac_coords': gt_res.get('unrelaxed_frac_coords'),
                                        'unrelaxed_cart_coords': gt_res.get('unrelaxed_cart_coords'),
                                        # Relaxed GT metrics (if relaxation was performed)
                                        'relaxed_energy': gt_res.get('relaxed_energy'),
                                        'relaxed_max_force': gt_res.get('relaxed_max_force'),
                                        'relaxed_mean_force': gt_res.get('relaxed_mean_force'),
                                        'relaxed_stress_voigt': gt_res.get('relaxed_stress_voigt'),
                                        'relaxed_stress_matrix': gt_res.get('relaxed_stress_matrix'),
                                        'relaxed_pressure': gt_res.get('relaxed_pressure'),
                                        'relaxed_lattice_matrix': gt_res.get('relaxed_lattice_matrix'),
                                        'relaxed_frac_coords': gt_res.get('relaxed_frac_coords'),
                                        'relaxed_cart_coords': gt_res.get('relaxed_cart_coords'),
                                        'energy_trajectory': gt_res.get('relaxation_trajectory'),
                                        'relaxation_converged': gt_res.get('relaxation_converged'),
                                        'error': gt_res.get('error'),
                                    }
                                else:
                                    print(f"   WARNING: No UMA results found for GT {gt_id}")
                                
                                # Update seed data with prediction results
                                for seed_data in all_seeds_data:
                                    # We used seed_id = "{refcode}_seed_{seed_idx}"
                                    sid = f"{refcode}_seed_{seed_data['seed_idx']}"
                                    
                                    # Update GT metrics for this seed (same for all seeds)
                                    seed_data['gt_uma_metrics'] = gt_uma_metrics
                                    
                                    # Store GT relaxed structure data
                                    if gt_uma_metrics.get('relaxed_cart_coords') is not None:
                                        seed_data['gt_relaxed_coords'] = torch.tensor(gt_uma_metrics['relaxed_cart_coords'], dtype=torch.float32)
                                    if gt_uma_metrics.get('relaxed_lattice_matrix') is not None:
                                        seed_data['gt_relaxed_lattice_matrix'] = torch.tensor(gt_uma_metrics['relaxed_lattice_matrix'], dtype=torch.float32)
                                    
                                    if sid in batch_results:
                                        res = batch_results[sid]
                                        
                                        metrics_update = {}
                                        
                                        # Unrelaxed prediction metrics
                                        metrics_update['pred_unrelaxed_energy'] = res.get('unrelaxed_energy')
                                        metrics_update['pred_unrelaxed_max_force'] = res.get('unrelaxed_max_force')
                                        metrics_update['pred_unrelaxed_mean_force'] = res.get('unrelaxed_mean_force')
                                        metrics_update['pred_unrelaxed_stress_voigt'] = res.get('unrelaxed_stress_voigt')
                                        metrics_update['pred_unrelaxed_stress_matrix'] = res.get('unrelaxed_stress_matrix')
                                        metrics_update['pred_unrelaxed_pressure'] = res.get('unrelaxed_pressure')
                                        
                                        # Relaxed prediction metrics
                                        if res.get('relaxation_steps', 0) > 0:
                                            metrics_update['pred_relaxed_energy'] = res.get('relaxed_energy')
                                            metrics_update['pred_relaxed_max_force'] = res.get('relaxed_max_force')
                                            metrics_update['pred_relaxed_mean_force'] = res.get('relaxed_mean_force')
                                            metrics_update['pred_relaxed_stress_voigt'] = res.get('relaxed_stress_voigt')
                                            metrics_update['pred_relaxed_stress_matrix'] = res.get('relaxed_stress_matrix')
                                            metrics_update['pred_relaxed_pressure'] = res.get('relaxed_pressure')
                                            metrics_update['pred_energy_trajectory'] = res.get('relaxation_trajectory')
                                            metrics_update['pred_relaxation_converged'] = res.get('relaxation_converged')
                                            metrics_update['relaxation_steps'] = res.get('relaxation_steps')
                                            
                                            # Store relaxed prediction structure data
                                            if res.get('relaxed_cart_coords') is not None:
                                                seed_data['pred_relaxed_coords'] = torch.tensor(res['relaxed_cart_coords'], dtype=torch.float32)
                                            if res.get('relaxed_lattice_matrix') is not None:
                                                seed_data['pred_relaxed_lattice_matrix'] = torch.tensor(res['relaxed_lattice_matrix'], dtype=torch.float32)
                                            
                                            # Reconstruct relaxed structure from UMA results (for visualization)
                                            if res.get('relaxed_frac_coords') is not None and res.get('relaxed_lattice_matrix') is not None:
                                                try:
                                                    relaxed_frac_coords = res['relaxed_frac_coords']
                                                    relaxed_lattice_matrix = res['relaxed_lattice_matrix']
                                                    
                                                    # Create Lattice from matrix
                                                    relaxed_lat = Lattice(relaxed_lattice_matrix)
                                                    
                                                    # Get species - use pred_atom_types for Genarris (reordered), atomic_numbers for others
                                                    if seed_data.get('pred_atom_types') is not None:
                                                        atomic_nums = np.array(seed_data['pred_atom_types'])
                                                    else:
                                                        atomic_nums = seed_data['atomic_numbers'].numpy()
                                                    species = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in atomic_nums]
                                                    
                                                    # Create Structure
                                                    relaxed_struct = Structure(relaxed_lat, species, relaxed_frac_coords, coords_are_cartesian=False)
                                                    seed_data['relaxed_structure'] = relaxed_struct
                                                except Exception as e:
                                                    print(f"   WARNING: Failed to reconstruct relaxed structure for {sid}: {e}")
                                            
                                        if res.get('error'):
                                            metrics_update['uma_error'] = res['error']

                                        seed_data['metrics'].update(metrics_update)
                                        
                                        # Compute residuals using GT metrics
                                        if gt_uma_metrics:
                                            # Unrelaxed Pred vs Unrelaxed GT
                                            if metrics_update.get('pred_unrelaxed_energy') is not None and gt_uma_metrics.get('unrelaxed_energy') is not None:
                                                seed_data['metrics']['residual_unrelaxed_energy'] = metrics_update['pred_unrelaxed_energy'] - gt_uma_metrics['unrelaxed_energy']
                                            if metrics_update.get('pred_unrelaxed_max_force') is not None and gt_uma_metrics.get('unrelaxed_max_force') is not None:
                                                seed_data['metrics']['residual_unrelaxed_max_force'] = metrics_update['pred_unrelaxed_max_force'] - gt_uma_metrics['unrelaxed_max_force']
                                            if metrics_update.get('pred_unrelaxed_mean_force') is not None and gt_uma_metrics.get('unrelaxed_mean_force') is not None:
                                                seed_data['metrics']['residual_unrelaxed_mean_force'] = metrics_update['pred_unrelaxed_mean_force'] - gt_uma_metrics['unrelaxed_mean_force']
                                            if metrics_update.get('pred_unrelaxed_pressure') is not None and gt_uma_metrics.get('unrelaxed_pressure') is not None:
                                                seed_data['metrics']['residual_unrelaxed_pressure'] = metrics_update['pred_unrelaxed_pressure'] - gt_uma_metrics['unrelaxed_pressure']
                                            
                                            # Relaxed Pred vs Relaxed GT (if both relaxed)
                                            if metrics_update.get('pred_relaxed_energy') is not None and gt_uma_metrics.get('relaxed_energy') is not None:
                                                seed_data['metrics']['residual_relaxed_energy'] = metrics_update['pred_relaxed_energy'] - gt_uma_metrics['relaxed_energy']
                                            if metrics_update.get('pred_relaxed_max_force') is not None and gt_uma_metrics.get('relaxed_max_force') is not None:
                                                seed_data['metrics']['residual_relaxed_max_force'] = metrics_update['pred_relaxed_max_force'] - gt_uma_metrics['relaxed_max_force']
                                            if metrics_update.get('pred_relaxed_mean_force') is not None and gt_uma_metrics.get('relaxed_mean_force') is not None:
                                                seed_data['metrics']['residual_relaxed_mean_force'] = metrics_update['pred_relaxed_mean_force'] - gt_uma_metrics['relaxed_mean_force']
                                            if metrics_update.get('pred_relaxed_pressure') is not None and gt_uma_metrics.get('relaxed_pressure') is not None:
                                                seed_data['metrics']['residual_relaxed_pressure'] = metrics_update['pred_relaxed_pressure'] - gt_uma_metrics['relaxed_pressure']
                                            
                                            # Also store GT metrics in the seed's metrics dict for easy access
                                            seed_data['metrics']['gt_unrelaxed_energy'] = gt_uma_metrics.get('unrelaxed_energy')
                                            seed_data['metrics']['gt_unrelaxed_max_force'] = gt_uma_metrics.get('unrelaxed_max_force')
                                            seed_data['metrics']['gt_unrelaxed_mean_force'] = gt_uma_metrics.get('unrelaxed_mean_force')
                                            seed_data['metrics']['gt_unrelaxed_pressure'] = gt_uma_metrics.get('unrelaxed_pressure')
                                            seed_data['metrics']['gt_relaxed_energy'] = gt_uma_metrics.get('relaxed_energy')
                                            seed_data['metrics']['gt_relaxed_max_force'] = gt_uma_metrics.get('relaxed_max_force')
                                            seed_data['metrics']['gt_relaxed_mean_force'] = gt_uma_metrics.get('relaxed_mean_force')
                                            seed_data['metrics']['gt_relaxed_pressure'] = gt_uma_metrics.get('relaxed_pressure')
                                            seed_data['metrics']['gt_energy_trajectory'] = gt_uma_metrics.get('energy_trajectory')
                                            seed_data['metrics']['gt_relaxation_converged'] = gt_uma_metrics.get('relaxation_converged')
                                            
                                    else:
                                        print(f"   WARNING: No UMA results found for seed {sid}")
                                        
                            except json.JSONDecodeError:
                                print(f"   WARNING: UMA Output JSON decode failed: {lines[-1]}")
                    else:
                        print(f"   WARNING: UMA batch script failed (code {result.returncode}): {result.stderr}")
                except Exception as e:
                    print(f"   WARNING: UMA batch execution error: {e}")
            else:
                 print(f"   WARNING: UMA script or python env not found.")
    
    if not all_seeds_data:
        print(f"ERROR: All {num_seeds} seeds failed for {refcode}")
        return None
    
    # Average metrics across all seeds
    averaged_data = average_metrics_across_seeds(all_seeds_data)
    averaged_metrics = averaged_data['metrics']
    representative_coords = averaged_data['pred_coords']
    representative_lattice = averaged_data['pred_lattice']
    representative_seed_idx = averaged_data['seed_idx']
    
    # Get representative relaxed structure if available
    representative_relaxed_struct = None
    for seed in all_seeds_data:
        if seed['seed_idx'] == representative_seed_idx:
            representative_relaxed_struct = seed['relaxed_structure']
            break
    
    # Add metadata to averaged metrics
    averaged_metrics['refcode'] = refcode
    averaged_metrics['num_atoms'] = len(atomic_numbers)
    
    # Calculate number of molecules
    num_molecules = 1
    if hasattr(crystal_data_obj, 'molecule_ids') and crystal_data_obj.molecule_ids is not None:
        num_molecules = len(torch.unique(crystal_data_obj.molecule_ids))
    elif hasattr(crystal_data_obj, 'edge_index') and crystal_data_obj.edge_index is not None:
        try:
            import scipy.sparse as sp
            from scipy.sparse.csgraph import connected_components
            
            # Create adjacency matrix
            rows = crystal_data_obj.edge_index[0].cpu().numpy()
            cols = crystal_data_obj.edge_index[1].cpu().numpy()
            data = np.ones(len(rows))
            n_nodes = len(atomic_numbers)
            
            adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            # connected_components returns (n_components, labels)
            num_molecules, _ = connected_components(adj, directed=False)
        except Exception as e:
            print(f"WARNING: Failed to calculate num_molecules from edge_index: {e}")
            num_molecules = 1
            
    averaged_metrics['num_molecules'] = int(num_molecules)
    averaged_metrics['representative_seed_idx'] = representative_seed_idx  # For visualization reference
    averaged_metrics['successful_seeds'] = successful_samples
    averaged_metrics['successful_seeds'] = successful_samples
    averaged_metrics['total_seeds'] = num_seeds
    averaged_metrics['inference_time_total_s'] = inference_time # Store total batch time

    
    # Create visualization if requested (using representative seed's structure)
    if visualize and output_dir is not None and representative_coords is not None:
        try:
            # Identify if model is Genarris (for visualization settings)
            is_genarris = False
            if isinstance(flow_matching, GenarrisWrapper):
                is_genarris = True
            elif hasattr(flow_matching, 'mode') and "genarris" in str(flow_matching.mode).lower():
                is_genarris = True
            # Note: GenarrisWrapper type check handles the main case for Genarris detection

            
            os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
            viz_path = os.path.join(output_dir, 'visualizations', f'{refcode}_comparison_seed{representative_seed_idx}.png')
            
            # Extract predicted atom types and edge_index for Genarris
            pred_atom_types = None
            pred_edge_index = None
            if is_genarris:
                 for seed in all_seeds_data:
                     if seed['seed_idx'] == representative_seed_idx:
                         pred_atom_types = seed.get('pred_atom_types')
                         pred_edge_index = seed.get('pred_edge_index')
                         break
            
            # Enable bond drawing for Genarris if we have edge_index, otherwise disable
            draw_bonds = True if (not is_genarris or pred_edge_index is not None) else False
            center_pred = True
            
            visualize_crystal_comparison(crystal_data, representative_coords, representative_lattice, refcode, viz_path, draw_pred_bonds=draw_bonds, center_pred=center_pred, pred_atom_types=pred_atom_types, pred_edge_index=pred_edge_index)
            
            # Additional visualization for relaxed structure
            if representative_relaxed_struct is not None:
                relaxed_viz_path = os.path.join(output_dir, 'visualizations', f'{refcode}_relaxed_seed{representative_seed_idx}.png')
                
                # Processing to make molecules whole and centered
                # For Genarris, use pred_edge_index; for Flow Matching, use crystal_data edge_index
                
                processed_relaxed = False
                
                # Determine which edge_index to use for making molecules whole
                if is_genarris and pred_edge_index is not None:
                    # Use Genarris's own edge_index (matches its atom ordering)
                    edge_index_for_bonds = pred_edge_index.cpu().numpy()
                elif not is_genarris:
                    # Use original crystal_data edge_index (matches Flow Matching atom ordering)
                    edge_index_for_bonds = crystal_data['edge_index'].cpu().numpy()
                else:
                    # Genarris without edge_index - skip make_molecules_whole
                    edge_index_for_bonds = None
                
                if edge_index_for_bonds is not None:
                    try:
                        # 1. Get connectivity from edge_index
                        bonds = []
                        # edge_index is [2, E], ensure we only get unique bonds (lower < upper)
                        for i in range(edge_index_for_bonds.shape[1]):
                            if edge_index_for_bonds[0, i] < edge_index_for_bonds[1, i]:
                                bonds.append({'atom1_idx': int(edge_index_for_bonds[0, i]), 'atom2_idx': int(edge_index_for_bonds[1, i])})
                                
                        # 2. Get fractional coords
                        # pymatgen structure stores fractional or cartesian. .frac_coords returns fractional.
                        relaxed_frac_coords = representative_relaxed_struct.frac_coords
                        
                        # 3. Make molecules whole
                        whole_frac_coords = make_molecules_whole(relaxed_frac_coords, bonds)
                        
                        # 4. Convert back to Cartesian and center
                        # We need lattice matrix for conversion
                        lat = representative_relaxed_struct.lattice
                        lattice_matrix = lat.matrix # 3x3 array
                        
                        whole_cart_coords = whole_frac_coords @ lattice_matrix
                        
                        # Center
                        centroid = whole_cart_coords.mean(axis=0)
                        centered_cart_coords = whole_cart_coords - centroid
                        
                        # Prepare for visualization
                        relaxed_coords = torch.tensor(centered_cart_coords, dtype=torch.float32, device=device)
                             
                        relaxed_lattice_params = torch.tensor([
                            lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma
                        ], dtype=torch.float32, device=device)
                        
                        # For Genarris, use pred_edge_index for drawing bonds
                        if is_genarris:
                            visualize_crystal_comparison(crystal_data, relaxed_coords, relaxed_lattice_params, f"{refcode} (Relaxed)", relaxed_viz_path, draw_pred_bonds=True, center_pred=False, pred_atom_types=pred_atom_types, pred_edge_index=pred_edge_index)
                        else:
                            visualize_crystal_comparison(crystal_data, relaxed_coords, relaxed_lattice_params, f"{refcode} (Relaxed)", relaxed_viz_path, draw_pred_bonds=True)
                        processed_relaxed = True
    
                    except Exception as e:
                        print(f"WARNING: Failed to process relaxed structure for visualization: {e}")
                        import traceback
                        traceback.print_exc()
                    
                if not processed_relaxed:
                    # Fallback to raw visualization if processing fails OR if no edge_index available
                    # Convert pymatgen Structure to tensors expected by visualize_crystal_comparison
                    relaxed_coords = torch.tensor(representative_relaxed_struct.cart_coords, dtype=torch.float32, device=device)
                    
                    lat = representative_relaxed_struct.lattice
                    relaxed_lattice_params = torch.tensor([
                        lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma
                    ], dtype=torch.float32, device=device)
                    
                    visualize_crystal_comparison(crystal_data, relaxed_coords, relaxed_lattice_params, f"{refcode} (Relaxed)", relaxed_viz_path, draw_pred_bonds=draw_bonds, center_pred=center_pred, pred_atom_types=pred_atom_types, pred_edge_index=pred_edge_index if is_genarris else None)
                
        except Exception as e:
            print(f"WARNING: Visualization failed for {refcode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Return both averaged metrics and all seeds data
    return {
        'best_metrics': averaged_metrics,  # Keep key name for compatibility with rest of code
        'all_seeds_data': all_seeds_data,
        # Store GT for re-evaluation if needed
        'true_coords': true_coords.detach().cpu(), 
        'true_lattice': true_lattice.detach().cpu(),
        'atomic_numbers': atomic_numbers.detach().cpu(),
        'edge_index': edge_index.detach().cpu()
    }


def average_metrics_across_seeds(all_seeds_data: List[Dict]) -> Dict:
    """Average metrics across all seeds for a crystal."""
    if not all_seeds_data:
        raise ValueError("No seeds data provided")
    
    # Get all metric keys from the first seed (assuming all seeds have same metrics)
    first_metrics = all_seeds_data[0]['metrics']
    averaged_metrics = {}
    
    # Average each metric across all seeds
    for key, value in first_metrics.items():
        if isinstance(value, (int, float)):
            # Numeric metric - average across seeds
            values = [seed_data['metrics'][key] for seed_data in all_seeds_data if key in seed_data['metrics']]
            if values:
                averaged_metrics[key] = np.mean(values)
        else:
            # Non-numeric field (e.g., strings) - take from first seed
            averaged_metrics[key] = value
    
    # For visualization, use the first seed's coordinates/lattice
    # (we could also average coordinates, but that's more complex and may not be meaningful)
    representative_seed = all_seeds_data[0]
    
    return {
        'metrics': averaged_metrics,
        'pred_coords': representative_seed['pred_coords'],
        'pred_lattice': representative_seed['pred_lattice'],
        'seed_idx': representative_seed['seed_idx']  # For visualization reference
    }


def find_best_seed_normalized(all_seeds_data: List[Dict]) -> Dict:
    """Find the best seed using normalized metrics for fair comparison."""
    if not all_seeds_data:
        raise ValueError("No seeds data provided")
    
    # Define which fields are actual metrics (not metadata)
    # Must match keys returned by crystal_metrics.compute_all_metrics_for_crystal
    METRIC_FIELDS = {
        # Density
        'density_rmse', 'density_mae', 'density_mape_pct',
        # Clash (percentage of atoms in clash)
        'clash_total_pct',
        # Global RDF statistics
        'rdf_jsd', 'rdf_overlap_pct', 'rdf_wasserstein',
        'rdf_overlap_short_range_pct', 'rdf_wasserstein_short_range',
        # AMD distance
        'amd_linf',
    }
    
    # Extract all numeric metrics from all seeds
    all_metrics_values = {}
    for seed_data in all_seeds_data:
        metrics = seed_data['metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key in METRIC_FIELDS:
                if key not in all_metrics_values:
                    all_metrics_values[key] = []
                all_metrics_values[key].append(value)
    
    # Calculate normalization statistics (mean and std for z-score normalization)
    norm_stats = {}
    for metric_name, values in all_metrics_values.items():
        values_array = np.array(values)
        norm_stats[metric_name] = {
            'mean': np.mean(values_array),
            'std': np.std(values_array) + 1e-8  # Add small epsilon to avoid division by zero
        }
    
    # Calculate normalized scores for each seed
    best_seed = None
    min_normalized_score = float('inf')
    
    for seed_data in all_seeds_data:
        metrics = seed_data['metrics']
        normalized_score = 0.0
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and metric_name in METRIC_FIELDS and metric_name in norm_stats:
                # Z-score normalization; all metrics are lower-is-better
                normalized_value = (value - norm_stats[metric_name]['mean']) / norm_stats[metric_name]['std']
                normalized_score += normalized_value  # Lower metrics = better score (more negative)
        
        if normalized_score < min_normalized_score:
            min_normalized_score = normalized_score
            best_seed = seed_data
    
    return best_seed


def compute_topk_analysis(all_crystals_data: List[Dict], num_monte_carlo: int = 1) -> Dict:
    """Compute top-k analysis based on optimal sum over all metrics."""
    
    # Define which fields are actual metrics (not metadata)
    METRIC_FIELDS = {
        'density_rmse', 'density_mae', 'density_mape_pct',
        'clash_total_pct',
        'rdf_jsd', 'rdf_overlap_pct', 'rdf_wasserstein',
        'rdf_overlap_short_range_pct', 'rdf_wasserstein_short_range',
        'amd_linf',
    }
    
    # Extract all metric names from the first successful crystal
    metric_names = []
    for crystal_data in all_crystals_data:
        if crystal_data['all_seeds_data']:
            # Only include fields that are actual metrics
            metric_names = [k for k, v in crystal_data['all_seeds_data'][0]['metrics'].items() 
                           if isinstance(v, (int, float)) and k in METRIC_FIELDS]
            break
    
    if not metric_names:
        return {}
    
    max_k = max(len(crystal_data['all_seeds_data']) for crystal_data in all_crystals_data 
                if crystal_data['all_seeds_data'])
    
    # Initialize results
    topk_results = {}
    for metric_name in metric_names:
        topk_results[metric_name] = {
            'k_values': list(range(1, max_k + 1)),
            'mean_topk_values': [],
            'std_topk_values': []
        }
    
    # For each k from 1 to max_k
    for k in range(1, max_k + 1):
        # For each metric, collect the values from the best overall seeds
        metric_values_dict = {metric_name: [] for metric_name in metric_names}
        
        # For each crystal
        for crystal_data in all_crystals_data:
            seeds_data = crystal_data['all_seeds_data']
            if len(seeds_data) < k:
                continue
            
            # Monte Carlo sampling
            for _ in range(num_monte_carlo):
                # Randomly select k seeds
                selected_seeds = random.sample(seeds_data, k)
                
                # Find the best seed among the selected k seeds based on normalized total score
                best_seed = find_best_seed_normalized(selected_seeds)
                
                # Extract all metric values from this best seed
                for metric_name in metric_names:
                    if metric_name in best_seed['metrics']:
                        metric_values_dict[metric_name].append(best_seed['metrics'][metric_name])
        
        # Compute statistics for each metric
        for metric_name in metric_names:
            all_topk_values = metric_values_dict[metric_name]
            
            if all_topk_values:
                mean_val = np.mean(all_topk_values)
                std_val = np.std(all_topk_values)
                topk_results[metric_name]['mean_topk_values'].append(mean_val)
                topk_results[metric_name]['std_topk_values'].append(std_val)
            else:
                topk_results[metric_name]['mean_topk_values'].append(None)
                topk_results[metric_name]['std_topk_values'].append(None)
    
    return topk_results


# Plotting functions moved to plot_evaluation_results.py


    # Determine if we should draw bonds
    # Heuristic: If refcode suggests Genarris or we passed a flag? 
    # The user asked to avoid just for genarris. 
    # We can check if 'pred_coords' matches what we expect or just use a flag?
    # This function doesn't know the model name.
    # We can rely on 'refcode' passed? No, refcode is from crystal.
    
    # We can add an optional argument `draw_bonds=True` to the function signature
    # and update the caller. 
    # But for now, let's look at the implementation plan: "In plotting logic, set argument..."
    # I need to update signature of visualize_crystal_comparison first.
    
    # Actually, I can just modify the signature here and update the caller below.
    # But for multi_replace, I should stick to existing chunks or include signature.
    # The signature is at line 1235.
    
    # Re-reading: "Avoid drawing bonds/edge_index in visualizations just for genarris."
    # I will modify the caller to pass edge_index=None for Genarris prediction plot.
    
    # Let's modify the function to accept `draw_pred_bonds`
    
    # Wait, I cannot change signature easily across multiple chunks if calls are scattered.
    # Call to this function is at 994 and 1035.
    
    # ALTERNATIVE: Use `edge_index` from `crystal_data`.
    # If I nullify `edge_index` for prediction drawing, it won't draw.
    
    # Let's modify the drawing loop to check a condition.
    # Or better: Add `draw_bonds` argument to `visualize_crystal_comparison`.
    pass # Placeholder for thought trace. Implementation below.
    
def visualize_crystal_comparison(crystal_data: Dict[str, torch.Tensor], 
                               pred_coords: torch.Tensor, pred_lattice: torch.Tensor,
                               refcode: str, save_path: str, draw_pred_bonds: bool = True,
                               center_pred: bool = True, pred_atom_types: Optional[np.ndarray] = None,
                               pred_edge_index: Optional[torch.Tensor] = None) -> None:
    """
    Visualize original vs predicted crystal structure side by side.
    
    Args:
        crystal_data: Original crystal data dictionary
        pred_coords: Predicted coordinates
        pred_lattice: Predicted lattice
        refcode: Crystal reference code
        save_path: Path to save the visualization
        draw_pred_bonds: Whether to draw bonds for the predicted structure
        center_pred: Whether to center the predicted coordinates (subtract mean)
        pred_atom_types: Optional explicit atom types for prediction (atomic numbers). 
                         If None, uses atom_types from crystal_data.
        pred_edge_index: Optional edge_index for predicted structure (for Genarris).
                         If provided, used for bond drawing and molecule unwrapping.
    """
    from packflow.utils.crystal_utils import lattice_params_to_matrix, make_molecules_whole, cart_to_frac, frac_to_cart
    from packflow.models.model import draw_unit_cell_box
    
    # Convert tensors to CPU and numpy for visualization
    orig_coords = crystal_data['whole_cartesian_coords'].cpu().numpy()
    orig_lattice = crystal_data['lattice_1'].cpu().numpy()
    atom_types = crystal_data['atom_types'].cpu().numpy()
    edge_index = crystal_data['edge_index'].cpu().numpy()
    
    pred_coords_np = pred_coords.cpu().numpy()
    pred_lattice_np = pred_lattice.cpu().numpy()
    
    # Convert lattice parameters to lattice matrix for both structures
    orig_lattice_matrix = lattice_params_to_matrix(
        orig_lattice[0], orig_lattice[1], orig_lattice[2],
        orig_lattice[3], orig_lattice[4], orig_lattice[5]
    )
    pred_lattice_matrix = lattice_params_to_matrix(
        pred_lattice_np[0], pred_lattice_np[1], pred_lattice_np[2],
        pred_lattice_np[3], pred_lattice_np[4], pred_lattice_np[5]
    )
    
    # Create figure with 2 subplots side by side
    fig = plt.figure(figsize=(12, 6))
    
    # Define colors for different atom types (Global map based on all present types)
    # Collect all types to ensure consistent coloring
    if pred_atom_types is not None:
        all_types = np.unique(np.concatenate([atom_types, pred_atom_types]))
    else:
        all_types = np.unique(atom_types)
        
    unique_atom_types = np.sort(all_types)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_atom_types)))
    atom_color_map = {atom_type: colors[i] for i, atom_type in enumerate(unique_atom_types)}
    
    atom_colors_orig = [atom_color_map[atom_type] for atom_type in atom_types]
    
    if pred_atom_types is not None:
        atom_colors_pred = [atom_color_map[atom_type] for atom_type in pred_atom_types]
    else:
        # Fallback to assumming same order/types if not provided (legacy behavior)
        # But handle length mismatch if prediction has different atom count
        if len(pred_coords_np) == len(atom_types):
             atom_colors_pred = atom_colors_orig
        else:
             # If no types provided and length mismatch, default to black or first color
             print(f"WARNING: Pred coords length {len(pred_coords_np)} != atom types {len(atom_types)} and no pred_atom_types provided.")
             atom_colors_pred = [colors[0]] * len(pred_coords_np)

    
    # Center coordinates
    orig_coords_centered = orig_coords - np.mean(orig_coords, axis=0) # Always center original
    
    # Process predicted coordinates: unwrap and center
    if pred_edge_index is not None and center_pred:
        # For Genarris: unwrap molecules using pred_edge_index, then center
        try:
            # Convert edge_index to bonds format for make_molecules_whole
            pred_edge_np = pred_edge_index.cpu().numpy() if torch.is_tensor(pred_edge_index) else pred_edge_index
            pred_bonds = []
            for i in range(pred_edge_np.shape[1]):
                a1, a2 = int(pred_edge_np[0, i]), int(pred_edge_np[1, i])
                if a1 < a2:  # Only add unique bonds (avoid duplicates)
                    pred_bonds.append({'atom1_idx': a1, 'atom2_idx': a2})
            
            # Convert to fractional, unwrap, convert back to cartesian
            pred_frac = cart_to_frac(pred_coords_np, pred_lattice_matrix)
            whole_frac = make_molecules_whole(pred_frac, pred_bonds)
            whole_cart = frac_to_cart(whole_frac, pred_lattice_matrix)
            
            # Center
            pred_coords_centered = whole_cart - np.mean(whole_cart, axis=0)
        except Exception as e:
            print(f"WARNING: Failed to unwrap molecules for visualization: {e}")
            # Fallback to just centering
            pred_coords_centered = pred_coords_np - np.mean(pred_coords_np, axis=0)
    elif center_pred:
        pred_coords_centered = pred_coords_np - np.mean(pred_coords_np, axis=0)
    else:
        pred_coords_centered = pred_coords_np
    
    # Original crystal
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Scatter plot of atoms
    ax1.scatter(orig_coords_centered[:, 0], 
               orig_coords_centered[:, 1], 
               orig_coords_centered[:, 2], 
               c=atom_colors_orig, s=80, alpha=0.8)
    
    # Draw lattice box centered around molecular center
    molecular_center = orig_coords_centered.mean(axis=0)
    draw_unit_cell_box(ax1, orig_lattice_matrix, center=molecular_center, color='black', alpha=0.5)
    
    # Draw bonds (Always for Original)
    for i in range(edge_index.shape[1]):
        atom1, atom2 = edge_index[:, i]
        if atom1 < len(orig_coords_centered) and atom2 < len(orig_coords_centered):
            ax1.plot([orig_coords_centered[atom1, 0], orig_coords_centered[atom2, 0]],
                   [orig_coords_centered[atom1, 1], orig_coords_centered[atom2, 1]],
                   [orig_coords_centered[atom1, 2], orig_coords_centered[atom2, 2]], 
                   'k-', alpha=0.6, linewidth=1.5)
    
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('Original', fontsize=14, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    
    # Predicted crystal
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Scatter plot of atoms
    ax2.scatter(pred_coords_centered[:, 0], 
               pred_coords_centered[:, 1], 
               pred_coords_centered[:, 2], 
               c=atom_colors_pred, s=80, alpha=0.8)
    
    # Draw lattice box centered around molecular center
    molecular_center = pred_coords_centered.mean(axis=0)
    draw_unit_cell_box(ax2, pred_lattice_matrix, center=molecular_center, color='black', alpha=0.5)
    
    # Draw bonds (Conditional for Predicted)
    if draw_pred_bonds:
        # Use pred_edge_index if available (Genarris), otherwise fallback to original edge_index
        bonds_to_draw = pred_edge_index.cpu().numpy() if pred_edge_index is not None else edge_index
        for i in range(bonds_to_draw.shape[1]):
            atom1, atom2 = bonds_to_draw[:, i]
            if atom1 < len(pred_coords_centered) and atom2 < len(pred_coords_centered):
                ax2.plot([pred_coords_centered[atom1, 0], pred_coords_centered[atom2, 0]],
                       [pred_coords_centered[atom1, 1], pred_coords_centered[atom2, 1]],
                       [pred_coords_centered[atom1, 2], pred_coords_centered[atom2, 2]], 
                       'k-', alpha=0.6, linewidth=1.5)
    
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_zlabel('Z (Å)')
    ax2.set_title('Predicted', fontsize=14, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    
    # Add overall title
    fig.suptitle(f'Crystal Structure Comparison: {refcode}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {save_path}")


def evaluate_test_set(flow_matching: CrystalFlowMatching, test_loader, 
                      atomic_masses: Dict[int, float], covalent_radii: Dict[int, float],
                      n_steps: int = 500, 
                      max_crystals: Optional[int] = None, visualize: bool = False,
                      output_dir: str = None, num_seeds: int = 1, 
                      num_monte_carlo: int = 1, lambda_val: float = 1.0,
                      compute_uma_metrics: bool = False,
                      uma_relaxation_steps: int = 0,
                      model_name: str = None, device: str = None,
                      seed_batch_size: Optional[int] = None,
                      save_crystal_data: bool = False,
                      save_uma_features: bool = False) -> Dict:
    """Evaluate metrics on the entire test set."""
    
    print(f"\nStarting test set evaluation...")
    print(f"   - Sampling steps: {n_steps}")
    print(f"   - Lambda (sampling temperature): {lambda_val}")
    print(f"   - Seeds per crystal: {num_seeds}")
    print(f"   - Seed batch size: {seed_batch_size if seed_batch_size else 'All at once'}")
    print(f"   - Monte Carlo samples: {num_monte_carlo}")
    print(f"   - Max crystals: {max_crystals if max_crystals else 'All'}")
    print(f"   - Save crystal data: {'Enabled' if save_crystal_data else 'Disabled'}")
    print(f"   - Save UMA features: {'Enabled' if save_uma_features else 'Disabled'}")
    
    all_crystals_data = []
    best_metrics_list = []
    successful_samples = 0
    failed_samples = 0
    
    # Global UMA Staging
    global_uma_manifest = []
    uma_staging_dir = None
    if compute_uma_metrics and output_dir:
        # Default: stage UMA intermediates under the evaluation output dir.
        uma_staging_dir = os.path.join(output_dir, "uma_staging")

        # Optional: stage on a fast node-local scratch disk if PACKFLOW_EVAL_STAGING_DIR
        # is set (useful on clusters to avoid hammering shared storage).
        from packflow import config as pf_config
        staging_override = pf_config.eval_staging_dir()
        if staging_override and model_name:
            try:
                model_staging_base = os.path.join(staging_override, f"evaluation_results_{model_name}")
                os.makedirs(model_staging_base, exist_ok=True)
                uma_staging_dir = os.path.join(model_staging_base, "uma_staging")
                print(f"   - Using staging dir: {uma_staging_dir}")
            except Exception as e:
                print(f"   - WARNING: failed to create staging dir {staging_override}: {e}")
                print(f"   - Falling back to: {uma_staging_dir}")

        os.makedirs(uma_staging_dir, exist_ok=True)
    
    crystal_count = 0
    start_time = time.time()
    
    # --- PHASE 1: GENERATION ---
    print("\n--- Phase 1: Generation & Basic Metrics ---")
    
    for batch_idx, batch in enumerate(test_loader):
        if max_crystals and crystal_count >= max_crystals:
            break
            
        # Process each crystal in the batch
        batch_split = batch.to_data_list()
        
        for crystal_data_obj in batch_split:
            if max_crystals and crystal_count >= max_crystals:
                break
                
            crystal_count += 1
            refcode = getattr(crystal_data_obj, 'refcode', f'Crystal_{crystal_count}')
            
            print(f"\nProcessing crystal {crystal_count}: {refcode}")
            
            # Evaluate this crystal (Defer UMA)
            # BATCH OPTIMIZATION: We now use immediate per-crystal batching (Phase 1),
            # so we disable global deferred execution (Phase 2).
            defer_uma = False 
            
            crystal_result = evaluate_single_crystal(
                flow_matching, crystal_data_obj, atomic_masses, covalent_radii, n_steps,
                visualize, output_dir, crystal_count, num_seeds, lambda_val,
                compute_uma_metrics=compute_uma_metrics,
                uma_relaxation_steps=uma_relaxation_steps,
                defer_uma_execution=defer_uma,
                uma_staging_dir=uma_staging_dir,
                device=device,
                seed_batch_size=seed_batch_size
            )
            
            if crystal_result is not None:
                all_crystals_data.append(crystal_result)
                
                # Collect pending UMA entries
                if defer_uma:
                    for seed_data in crystal_result['all_seeds_data']:
                        if seed_data.get('uma_pending_entry'):
                            global_uma_manifest.append(seed_data['uma_pending_entry'])
                
                # Check success (pre-UMA)
                successful_samples += 1
                
                # Print intermediate metrics (without UMA)
                best_metrics = crystal_result['best_metrics']
                print(
                    f"   Generated - "
                    f"Density RMSE: {best_metrics.get('density_rmse', 0.0):.4f}, "
                    f"Clash Total %: {best_metrics.get('clash_total_pct', 0.0):.2f}"
                )
            else:
                failed_samples += 1
                print(f"   Failed")
            
            # Progress update
            if crystal_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = crystal_count / elapsed
                eta = (len(test_loader.dataset) - crystal_count) / rate if rate > 0 else 0
                print(f"   Progress: {crystal_count} crystals, {rate:.1f} crystals/sec, ETA: {eta/60:.1f} min")

    # --- PHASE 2: GLOBAL UMA EXECUTION ---
    if compute_uma_metrics and global_uma_manifest:
        print(f"\n--- Phase 2: Global UMA Evaluation ({len(global_uma_manifest)} items) ---")
        
        manifest_path = os.path.join(uma_staging_dir, "global_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(global_uma_manifest, f)
            
        script_path = os.path.join(UMA_SCRIPTS_DIR, "workers", "uma_metrics.py")
        fairchem_python = FAIRCHEM_PYTHON
        
        # Determine device/args
        # We need to read the `uma_skip_relaxation` flag from the function signature?
        # NO, `evaluate_test_set` signature doesn't have it?
        # WAIT, `evaluate_test_set_metrics.py` argparse has it, but `evaluate_test_set` function signature 
        # needs to accept it.
        # I need to check `evaluate_test_set` signature. It does NOT have `uma_skip_relaxation`.
        # I should add it or infer it? 
        # Currently `compute_uma_metrics` is there.
        # The `uma_skip_relaxation` is in `evaluate_single_crystal` args but not `evaluate_test_set`.
        # I must have missed updating `evaluate_test_set` signature in previous steps?
        # Let's check `evaluate_test_set` call site in `main`.
        
        # Assuming I can get it from args passed to `evaluate_single_crystal`? No.
        # I will assume `uma_skip_relaxation` needs to be added to `evaluate_test_set` or accessed via a global arg (bad practice).
        # Better: Add it to `evaluate_test_set` signature in this edit.
        
        # Back to execution logic:
        cmd = [fairchem_python, script_path, "--batch_manifest", manifest_path, "--device", "cuda" if torch.cuda.is_available() else "cpu"]
            
        print(f"Running UMA script...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("UMA script completed successfully.")
                # The output might be huge, so we might want to read it from a file?
                # The script prints to stdout.
                # If stdout is huge (>buffer), subprocess might hang?
                # `capture_output=True` stores in memory. Should be fine for 100-500 crystals (few MBs).
                
                try:
                    lines = result.stdout.strip().splitlines()
                    batch_results = json.loads(lines[-1])
                except:
                    print(f"UMA Output JSON decode failed.")
                    batch_results = {}
            else:
                 print(f"UMA script failed: {result.stderr}")
                 batch_results = {}
        except Exception as e:
            print(f"UMA execution error: {e}")
            batch_results = {}
            
        # --- PHASE 3: UPDATE METRICS ---
        print("\n--- Phase 3: Updating Metrics & Saving ---")
        
        updated_crystal_count = 0

        # CORRECT LOOP:
        for crystal_data_res in all_crystals_data:
            seeds_data = crystal_data_res['all_seeds_data']
            seeds_updated = False
            
            for seed_data in seeds_data:
                # Reconstruct ID
                # seed_data['uma_pending_entry'] has the ID!
                if seed_data.get('uma_pending_entry'):
                    sid = seed_data['uma_pending_entry']['id']
                    if sid in batch_results:
                        res = batch_results[sid]
                        seed_data['metrics'].update(res)
                        
                        # Compute residuals if GT metrics available
                        gt_metrics = seed_data.get('gt_uma_metrics', {})
                        if gt_metrics and 'unrelaxed_energy' in res and 'unrelaxed_energy' in gt_metrics:
                            try:
                                # Energy residual
                                pred_e = res['unrelaxed_energy']
                                gt_e = gt_metrics['unrelaxed_energy']
                                if pred_e is not None and gt_e is not None:
                                    seed_data['metrics']['residual_energy'] = pred_e - gt_e
                                
                                # Force residual
                                pred_f = res['unrelaxed_mean_force']
                                gt_f = gt_metrics['unrelaxed_mean_force']
                                if pred_f is not None and gt_f is not None:
                                    seed_data['metrics']['residual_force'] = pred_f - gt_f
                            except Exception as e:
                                print(f"   WARNING: Failed to compute residuals (deferred): {e}")

                        seeds_updated = True
                        
            if seeds_updated:
                 # Re-average metrics
                 new_avg = average_metrics_across_seeds(seeds_data)
                 # Preserve original metadata that average_metrics_across_seeds might overwrite/miss
                 # Actually average_metrics_across_seeds creates a new dict.
                 # We should re-add metadata if needed, but crystal_data_res['best_metrics'] was updated earlier 
                 # in evaluate_single_crystal with extra metadata like refcode, num_molecules...
                 # We must ensure we don't lose them.
                 
                 # Let's merge new metrics into old best_metrics to preserve metadata
                 old_best = crystal_data_res['best_metrics']
                 new_metrics = new_avg['metrics']
                 old_best.update(new_metrics) # Update values
                 
                 # crystal_data_res['best_metrics'] = old_best # Already updated in place
                 
                 updated_crystal_count += 1

                 
        print(f"Updated metrics for {updated_crystal_count} crystals.")
        
        # Cleanup staging directory
        if uma_staging_dir and os.path.exists(uma_staging_dir):
            import shutil
            try:
                shutil.rmtree(uma_staging_dir)
                print(f"Cleaned up UMA staging directory: {uma_staging_dir}")
            except Exception as e:
                print(f"Warning: Failed to cleanup UMA staging directory: {e}")

    # Final Summary & Top-K (Existing logic)
    elapsed_total = time.time() - start_time
    
    # Re-extract best_metrics_list for final summary
    best_metrics_list = [c['best_metrics'] for c in all_crystals_data]
    
    print(f"\nEvaluation Summary:")
    print(f"   - Total crystals processed: {crystal_count}")
    print(f"   - Successful samples: {successful_samples}")
    print(f"   - Failed samples: {failed_samples}")
    print(f"   - Success rate: {successful_samples/crystal_count*100:.1f}%" if crystal_count > 0 else "   - Success rate: 0%")
    print(f"   - Total time: {elapsed_total/60:.1f} minutes")
    print(f"   - Average time per crystal: {elapsed_total/crystal_count:.1f} seconds" if crystal_count > 0 else "   - Average time per crystal: N/A")
    
    # Perform top-k analysis if we have multiple seeds (results saved in JSON, plotting done separately)
    if num_seeds > 1 and all_crystals_data:
        print(f"\nComputing top-k analysis...")
        topk_results = compute_topk_analysis(all_crystals_data, num_monte_carlo)
    else:
        topk_results = {}
    
    return {
        'best_metrics_list': best_metrics_list,
        'all_crystals_data': all_crystals_data,
        'topk_results': topk_results
    }


# Plotting functions moved to plot_evaluation_results.py
# def plot_metrics_vs_rotatable_bonds(best_metrics_list: List[Dict[str, float]],
    """Plot each metric as a function of number of rotatable bonds."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    
    def _filter_outliers_y(y: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
        """Filter extreme outliers in y using quantiles; returns boolean mask."""
        if len(y) < 10:
            return np.ones(len(y), dtype=bool)
        lo = np.quantile(y, lo_q)
        hi = np.quantile(y, hi_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.ones(len(y), dtype=bool)
        mask = (y >= lo) & (y <= hi)
        if mask.sum() < max(5, int(0.5 * len(y))):
            return np.ones(len(y), dtype=bool)
        return mask
    
    def _set_robust_ylim(ax, y: np.ndarray, pad_frac: float = 0.05) -> None:
        """Set y-limits based on robust quantiles (ignores extremes)."""
        if len(y) < 2:
            return
        lo = np.quantile(y, 0.01) if len(y) >= 10 else np.min(y)
        hi = np.quantile(y, 0.99) if len(y) >= 10 else np.max(y)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return
        pad = (hi - lo) * pad_frac
        ax.set_ylim(lo - pad, hi + pad)
    
    # Define which fields are actual metrics (not metadata)
    # Excluding density_rmse, density_mae, and RDF metrics as requested
    METRIC_FIELDS = {
        'density_mape_pct',
        'clash_total_pct',
        'amd_linf',
    }
    
    # Collect data: (rotatable_bonds, metric_value) pairs
    metric_data = {metric_name: {'rot_bonds': [], 'values': []} 
                   for metric_name in METRIC_FIELDS}
    
    for idx, metrics in enumerate(best_metrics_list):
        if idx not in rotatable_bonds_dict:
            continue  # Skip if no rotatable bond data for this crystal
        
        n_rot = rotatable_bonds_dict[idx]
        
        for metric_name in METRIC_FIELDS:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_data[metric_name]['rot_bonds'].append(n_rot)
                    metric_data[metric_name]['values'].append(value)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'rotatable_bonds_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Metric display names
    metric_display_names = {
        'amd_linf': 'AMD L∞',
        'density_mape_pct': 'Density MAPE (%)',
        'clash_total_pct': 'Clash Total (%)',
    }
    
    # Create individual plots for each metric
    for metric_name, data in metric_data.items():
        if not data['rot_bonds']:  # Skip if no data
                    continue
                    
        rot_bonds = np.array(data['rot_bonds'])
        values = np.array(data['values'])
        
        # Calculate Spearman correlation on FULL data (before filtering)
        if len(rot_bonds) > 1:
            spearman_corr, spearman_p = spearmanr(rot_bonds, values)
        else:
            spearman_corr = np.nan
        
        # Filter outliers for plotting only
        mask = _filter_outliers_y(values)
        rot_bonds_plot = rot_bonds[mask]
        values_plot = values[mask]
        
        # Create scatter plot with trend line
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot (only non-outliers)
        ax.scatter(rot_bonds_plot, values_plot, alpha=0.6, s=50, color='#2E8B57', edgecolors='white', linewidth=1.5)
        
        # Add trend line (linear regression on filtered data)
        if len(rot_bonds_plot) > 1:
            z = np.polyfit(rot_bonds_plot, values_plot, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(rot_bonds_plot.min(), rot_bonds_plot.max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
            
            # Display Spearman correlation (calculated on full data)
            if not np.isnan(spearman_corr):
                ax.text(0.98, 0.98, f'ρ = {spearman_corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(fontsize=12, loc='lower right')
        
        # Labels and title
        display_name = metric_display_names.get(metric_name, metric_name.replace('_', ' ').title())
        ax.set_xlabel('Number of Rotatable Bonds', fontsize=16)
        ax.set_ylabel(display_name, fontsize=16)
        
        # Remove spines and add grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f'{metric_name}_vs_rotatable_bonds.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved plot: {plot_path}")
    
    # Create combined plot with all 3 metrics in a single row
    valid_metrics = {name: data for name, data in metric_data.items() if data['rot_bonds']}
    if not valid_metrics:
        print("WARNING: No valid metrics found for rotatable bonds plots")
        return
    
    # Filter to only the 3 requested metrics in order
    ordered_metrics = ['density_mape_pct', 'clash_total_pct', 'amd_linf']
    valid_metrics = {k: v for k, v in valid_metrics.items() if k in ordered_metrics}
    
    n_metrics = len(valid_metrics)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(ordered_metrics):
        if metric_name not in valid_metrics:
            continue
        data = valid_metrics[metric_name]
        ax = axes[i]
        
        rot_bonds = np.array(data['rot_bonds'])
        values = np.array(data['values'])
        
        # Calculate Spearman correlation on FULL data (before filtering)
        if len(rot_bonds) > 1:
            spearman_corr, spearman_p = spearmanr(rot_bonds, values)
        else:
            spearman_corr = np.nan
        
        # Filter outliers for plotting only
        mask = _filter_outliers_y(values)
        rot_bonds_plot = rot_bonds[mask]
        values_plot = values[mask]
        
        # Scatter plot (only non-outliers)
        ax.scatter(rot_bonds_plot, values_plot, alpha=0.6, s=50, color='#2E8B57', edgecolors='white', linewidth=1.5)
        
        # Trend line (on filtered data)
        if len(rot_bonds_plot) > 1:
            z = np.polyfit(rot_bonds_plot, values_plot, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(rot_bonds_plot.min(), rot_bonds_plot.max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8)
            
            # Spearman correlation (calculated on full data)
            if not np.isnan(spearman_corr):
                ax.text(0.98, 0.98, f'ρ = {spearman_corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Labels
        display_name = metric_display_names.get(metric_name, metric_name.replace('_', ' ').title())
        ax.set_xlabel('Rotatable Bonds', fontsize=14)
        ax.set_ylabel(display_name, fontsize=14)
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Set robust y-limits
        _set_robust_ylim(ax, values_plot)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(plots_dir, 'all_metrics_vs_rotatable_bonds.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved combined plot: {combined_plot_path}")


def plot_metrics_vs_num_molecules(best_metrics_list: List[Dict[str, float]],
                                  num_molecules_dict: Dict[int, int],
                                  output_dir: str) -> None:
    """Scatter plots of metrics vs number of molecules (components) per crystal."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    
    def _filter_outliers_y(y: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
        """Filter extreme outliers in y using quantiles; returns boolean mask."""
        if len(y) < 10:
            return np.ones(len(y), dtype=bool)
        lo = np.quantile(y, lo_q)
        hi = np.quantile(y, hi_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.ones(len(y), dtype=bool)
        mask = (y >= lo) & (y <= hi)
        if mask.sum() < max(5, int(0.5 * len(y))):
            return np.ones(len(y), dtype=bool)
        return mask
    
    def _set_robust_ylim(ax, y: np.ndarray, pad_frac: float = 0.05) -> None:
        """Set y-limits based on robust quantiles (ignores extremes)."""
        if len(y) < 2:
            return
        lo = np.quantile(y, 0.01) if len(y) >= 10 else np.min(y)
        hi = np.quantile(y, 0.99) if len(y) >= 10 else np.max(y)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return
        pad = (hi - lo) * pad_frac
        ax.set_ylim(lo - pad, hi + pad)
    
    # Excluding density_rmse, density_mae, and RDF metrics as requested
    METRIC_FIELDS = {
        'density_mape_pct',
        'clash_total_pct',
        'amd_linf',
    }

    metric_display_names = {
        'amd_linf': 'AMD L∞',
        'density_mape_pct': 'Density MAPE (%)',
        'clash_total_pct': 'Clash Total (%)',
    }

    metric_data = {metric_name: {'num_mols': [], 'values': []}
                   for metric_name in METRIC_FIELDS}

    for idx, metrics in enumerate(best_metrics_list):
        if idx not in num_molecules_dict:
            continue
        n_mols = num_molecules_dict[idx]
        for metric_name in METRIC_FIELDS:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_data[metric_name]['num_mols'].append(n_mols)
                    metric_data[metric_name]['values'].append(value)

    plots_dir = os.path.join(output_dir, 'num_molecules_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create combined plot with all 3 metrics in a single row
    valid_metrics = {name: data for name, data in metric_data.items() if data['num_mols']}
    if not valid_metrics:
        print("WARNING: No valid metrics found for num molecules plots")
        return
    
    # Filter to only the 3 requested metrics in order
    ordered_metrics = ['density_mape_pct', 'clash_total_pct', 'amd_linf']
    valid_metrics = {k: v for k, v in valid_metrics.items() if k in ordered_metrics}
    
    n_metrics = len(valid_metrics)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(ordered_metrics):
        if metric_name not in valid_metrics:
            continue
        data = valid_metrics[metric_name]
        ax = axes[i]
        
        x = np.array(data['num_mols'])
        y = np.array(data['values'])
        
        # Calculate Spearman correlation on FULL data (before filtering)
        if len(x) > 1:
            spearman_corr, spearman_p = spearmanr(x, y)
        else:
            spearman_corr = np.nan
        
        # Filter outliers for plotting only
        mask = _filter_outliers_y(y)
        x_plot = x[mask]
        y_plot = y[mask]
        
        ax.scatter(x_plot, y_plot, alpha=0.6, s=50, color='#2E8B57', edgecolors='white', linewidth=1.5)
        if len(x_plot) > 1:
            z = np.polyfit(x_plot, y_plot, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_plot.min(), x_plot.max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8)
            
            # Spearman correlation (calculated on full data)
            if not np.isnan(spearman_corr):
                ax.text(0.98, 0.98, f'ρ = {spearman_corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        display_name = metric_display_names.get(metric_name, metric_name.replace('_', ' ').title())
        ax.set_xlabel('Number of Molecules', fontsize=14)
        ax.set_ylabel(display_name, fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Set robust y-limits
        _set_robust_ylim(ax, y_plot)
    
    plt.tight_layout()
    combined_path = os.path.join(plots_dir, 'all_metrics_vs_num_molecules.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined plot: {combined_path}")


def plot_metrics_vs_num_atoms(best_metrics_list: List[Dict[str, float]],
                              num_atoms_dict: Dict[int, int],
                              output_dir: str) -> None:
    """Scatter plots of metrics vs number of atoms per crystal."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    
    def _filter_outliers_y(y: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
        """Filter extreme outliers in y using quantiles; returns boolean mask."""
        if len(y) < 10:
            return np.ones(len(y), dtype=bool)
        lo = np.quantile(y, lo_q)
        hi = np.quantile(y, hi_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.ones(len(y), dtype=bool)
        mask = (y >= lo) & (y <= hi)
        if mask.sum() < max(5, int(0.5 * len(y))):
            return np.ones(len(y), dtype=bool)
        return mask
    
    def _set_robust_ylim(ax, y: np.ndarray, pad_frac: float = 0.05) -> None:
        """Set y-limits based on robust quantiles (ignores extremes)."""
        if len(y) < 2:
            return
        lo = np.quantile(y, 0.01) if len(y) >= 10 else np.min(y)
        hi = np.quantile(y, 0.99) if len(y) >= 10 else np.max(y)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return
        pad = (hi - lo) * pad_frac
        ax.set_ylim(lo - pad, hi + pad)
    
    # Excluding density_rmse, density_mae, and RDF metrics as requested
    METRIC_FIELDS = {
        'density_mape_pct',
        'clash_total_pct',
        'amd_linf',
    }

    metric_display_names = {
        'amd_linf': 'AMD L∞',
        'density_mape_pct': 'Density MAPE (%)',
        'clash_total_pct': 'Clash Total (%)',
    }

    metric_data = {metric_name: {'num_atoms': [], 'values': []}
                   for metric_name in METRIC_FIELDS}

    for idx, metrics in enumerate(best_metrics_list):
        if idx not in num_atoms_dict:
            continue
        n_atoms = num_atoms_dict[idx]
        for metric_name in METRIC_FIELDS:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metric_data[metric_name]['num_atoms'].append(n_atoms)
                    metric_data[metric_name]['values'].append(value)

    plots_dir = os.path.join(output_dir, 'num_atoms_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create combined plot with all 3 metrics in a single row
    valid_metrics = {name: data for name, data in metric_data.items() if data['num_atoms']}
    if not valid_metrics:
        print("WARNING: No valid metrics found for num atoms plots")
        return
    
    # Filter to only the 3 requested metrics in order
    ordered_metrics = ['density_mape_pct', 'clash_total_pct', 'amd_linf']
    valid_metrics = {k: v for k, v in valid_metrics.items() if k in ordered_metrics}
    
    n_metrics = len(valid_metrics)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(ordered_metrics):
        if metric_name not in valid_metrics:
            continue
        data = valid_metrics[metric_name]
        ax = axes[i]
        
        x = np.array(data['num_atoms'])
        y = np.array(data['values'])
        
        # Calculate Spearman correlation on FULL data (before filtering)
        if len(x) > 1:
            spearman_corr, spearman_p = spearmanr(x, y)
        else:
            spearman_corr = np.nan
        
        # Filter outliers for plotting only
        mask = _filter_outliers_y(y)
        x_plot = x[mask]
        y_plot = y[mask]
        
        ax.scatter(x_plot, y_plot, alpha=0.6, s=50, color='#2E8B57', edgecolors='white', linewidth=1.5)
        if len(x_plot) > 1:
            z = np.polyfit(x_plot, y_plot, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_plot.min(), x_plot.max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8)
            
            # Spearman correlation (calculated on full data)
            if not np.isnan(spearman_corr):
                ax.text(0.98, 0.98, f'ρ = {spearman_corr:.3f}', 
                       transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        display_name = metric_display_names.get(metric_name, metric_name.replace('_', ' ').title())
        ax.set_xlabel('Number of Atoms', fontsize=14)
        ax.set_ylabel(display_name, fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Set robust y-limits
        _set_robust_ylim(ax, y_plot)
    
    plt.tight_layout()
    combined_path = os.path.join(plots_dir, 'all_metrics_vs_num_atoms.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
# Plotting functions moved to plot_evaluation_results.py


def save_results(best_metrics_list: List[Dict[str, float]], all_crystals_data: List[Dict], 
                output_dir: str, checkpoint_path: str, n_steps: int, data_dir: str,
                save_crystal_data: bool = False, save_uma_features: bool = False,
                chunk: bool = False) -> Dict[str, float]:
    """Save results and compute summary statistics.
    
    Args:
        best_metrics_list: List of best metrics per crystal
        all_crystals_data: All crystal data including seeds
        output_dir: Output directory
        checkpoint_path: Path to model checkpoint
        n_steps: Number of sampling steps
        data_dir: Data directory path
        save_crystal_data: If True, save crystal data (coords, lattice, atom_types) to .pt files
        save_uma_features: If True, include full UMA features (forces, per-atom energies) in .pt files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute mean metrics from averaged seed metrics (averaged per crystal, then across crystals)
    mean_metrics_result = mean_metrics(best_metrics_list)
    
    # Save detailed results (averaged metrics per crystal)
    detailed_results_path = os.path.join(output_dir, "detailed_metrics_best_seeds.json")
    with open(detailed_results_path, 'w') as f:
        json.dump(best_metrics_list, f, indent=2)
    
    # Save all seeds data
    all_seeds_results_path = os.path.join(output_dir, "all_seeds_data.json")

    # Helper function to convert tensor to list
    def tensor_to_list(t):
        if t is None:
            return None
        if torch.is_tensor(t):
            return t.cpu().numpy().tolist()
        if isinstance(t, np.ndarray):
            return t.tolist()
        return t

    if chunk:
        # Stream-write each crystal to avoid a single huge end-of-run json.dump().
        appender = _JsonArrayAppender(all_seeds_results_path)
        for crystal_data in all_crystals_data:
            crystal_serializable = {
                "best_metrics": crystal_data["best_metrics"],
                "all_seeds_data": [],
            }
            for seed_data in crystal_data["all_seeds_data"]:
                seed_serializable = {
                    "seed_idx": seed_data["seed_idx"],
                    "refcode": seed_data.get("refcode", "Unknown"),
                    "metrics": seed_data["metrics"],
                    # Predicted structure (unrelaxed)
                    "pred_coords": tensor_to_list(seed_data.get("pred_coords")),
                    "pred_lattice": tensor_to_list(seed_data.get("pred_lattice")),
                    "pred_lattice_matrix": tensor_to_list(seed_data.get("pred_lattice_matrix")),
                    "pred_atom_types": tensor_to_list(seed_data.get("pred_atom_types")),
                    "pred_edge_index": tensor_to_list(seed_data.get("pred_edge_index")),
                    # Predicted structure (relaxed)
                    "pred_relaxed_coords": tensor_to_list(seed_data.get("pred_relaxed_coords")),
                    "pred_relaxed_lattice_matrix": tensor_to_list(seed_data.get("pred_relaxed_lattice_matrix")),
                    # Ground truth structure (unrelaxed)
                    "gt_coords": tensor_to_list(seed_data.get("gt_coords")),
                    "gt_lattice": tensor_to_list(seed_data.get("gt_lattice")),
                    "gt_lattice_matrix": tensor_to_list(seed_data.get("gt_lattice_matrix")),
                    "gt_atom_types": tensor_to_list(seed_data.get("gt_atom_types")),
                    "gt_edge_index": tensor_to_list(seed_data.get("gt_edge_index")),
                    # Ground truth structure (relaxed)
                    "gt_relaxed_coords": tensor_to_list(seed_data.get("gt_relaxed_coords")),
                    "gt_relaxed_lattice_matrix": tensor_to_list(seed_data.get("gt_relaxed_lattice_matrix")),
                }
                crystal_serializable["all_seeds_data"].append(seed_serializable)
            appender.append(crystal_serializable)
    else:
        with open(all_seeds_results_path, "w") as f:
            # Convert tensors to lists for JSON serialization
            serializable_data = []
            for crystal_data in all_crystals_data:
                crystal_serializable = {
                    "best_metrics": crystal_data["best_metrics"],
                    "all_seeds_data": [],
                }
                for seed_data in crystal_data["all_seeds_data"]:
                    seed_serializable = {
                        "seed_idx": seed_data["seed_idx"],
                        "refcode": seed_data.get("refcode", "Unknown"),
                        "metrics": seed_data["metrics"],
                        # Predicted structure (unrelaxed)
                        "pred_coords": tensor_to_list(seed_data.get("pred_coords")),
                        "pred_lattice": tensor_to_list(seed_data.get("pred_lattice")),
                        "pred_lattice_matrix": tensor_to_list(seed_data.get("pred_lattice_matrix")),
                        "pred_atom_types": tensor_to_list(seed_data.get("pred_atom_types")),
                        "pred_edge_index": tensor_to_list(seed_data.get("pred_edge_index")),
                        # Predicted structure (relaxed)
                        "pred_relaxed_coords": tensor_to_list(seed_data.get("pred_relaxed_coords")),
                        "pred_relaxed_lattice_matrix": tensor_to_list(seed_data.get("pred_relaxed_lattice_matrix")),
                        # Ground truth structure (unrelaxed)
                        "gt_coords": tensor_to_list(seed_data.get("gt_coords")),
                        "gt_lattice": tensor_to_list(seed_data.get("gt_lattice")),
                        "gt_lattice_matrix": tensor_to_list(seed_data.get("gt_lattice_matrix")),
                        "gt_atom_types": tensor_to_list(seed_data.get("gt_atom_types")),
                        "gt_edge_index": tensor_to_list(seed_data.get("gt_edge_index")),
                        # Ground truth structure (relaxed)
                        "gt_relaxed_coords": tensor_to_list(seed_data.get("gt_relaxed_coords")),
                        "gt_relaxed_lattice_matrix": tensor_to_list(seed_data.get("gt_relaxed_lattice_matrix")),
                    }
                    crystal_serializable["all_seeds_data"].append(seed_serializable)
                serializable_data.append(crystal_serializable)
            json.dump(serializable_data, f, indent=2)
    
    # Save summary results
    summary_results_path = os.path.join(output_dir, "summary_metrics.json")
    summary_data = {
        "checkpoint_path": checkpoint_path,
        "n_steps": n_steps,
        "num_crystals": len(best_metrics_list),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mean_metrics": mean_metrics_result
    }
    
    with open(summary_results_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # --- Rotatable Bonds Analysis ---
    rotatable_bonds_dict = {}
    num_molecules_dict = {}
    num_atoms_dict = {}
    rotatable_bonds_path = os.path.join(data_dir, "test_set_rotatable_bonds.pt")
    num_molecules_path = os.path.join(data_dir, "test_set_num_molecules.pt")
    num_atoms_path = os.path.join(data_dir, "test_set_num_atoms.pt")
    
    if os.path.exists(rotatable_bonds_path):
        print(f"\nLoading rotatable bond counts from {rotatable_bonds_path}")
        try:
            rotatable_bonds_dict = torch.load(rotatable_bonds_path, map_location='cpu')
            # Convert to regular dict if needed
            if isinstance(rotatable_bonds_dict, dict):
                rotatable_bonds_dict = {int(k): int(v) for k, v in rotatable_bonds_dict.items()}
            
            # Add rotatable bond count to each crystal's metrics
            for idx, metrics in enumerate(best_metrics_list):
                if idx in rotatable_bonds_dict:
                    metrics['num_rotatable_bonds'] = rotatable_bonds_dict[idx]
            
            # Note: Plotting is done separately using plot_evaluation_results.py
            
        except Exception as e:
            print(f"WARNING: Failed to load rotatable bond counts or create plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"No rotatable bond counts found at {rotatable_bonds_path}, skipping rotatable bonds analysis.")
    # --------------------------

    # --- Number of Molecules Analysis ---
    if os.path.exists(num_molecules_path):
        print(f"\nLoading number of molecules from {num_molecules_path}")
        try:
            num_molecules_dict = torch.load(num_molecules_path, map_location='cpu')
            if isinstance(num_molecules_dict, dict):
                num_molecules_dict = {int(k): int(v) for k, v in num_molecules_dict.items()}
            # Note: Plotting is done separately using plot_evaluation_results.py
        except Exception as e:
            print(f"WARNING: Failed to load num molecules counts or create plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"No num molecules counts found at {num_molecules_path}, skipping num molecules analysis.")
    # --------------------------

    # --- Number of Atoms Analysis ---
    if os.path.exists(num_atoms_path):
        print(f"\nLoading number of atoms from {num_atoms_path}")
        try:
            num_atoms_dict = torch.load(num_atoms_path, map_location='cpu')
            if isinstance(num_atoms_dict, dict):
                num_atoms_dict = {int(k): int(v) for k, v in num_atoms_dict.items()}
            # Note: Plotting is done separately using plot_evaluation_results.py
        except Exception as e:
            print(f"WARNING: Failed to load num atoms counts or create plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"No num atoms counts found at {num_atoms_path}, skipping num atoms analysis.")
    # --------------------------
    
    # Save CSV format for easy analysis (averaged metrics per crystal)
    import pandas as pd
    df = pd.DataFrame(best_metrics_list)
    csv_path = os.path.join(output_dir, "metrics_best_seeds.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved:")
    print(f"   - Detailed metrics (averaged per crystal): {detailed_results_path}")
    print(f"   - All seeds data: {all_seeds_results_path}")
    print(f"   - Summary metrics: {summary_results_path}")
    print(f"   - CSV format (averaged per crystal): {csv_path}")
    
    # Save crystal data and UMA features to .pt files (structured format)
    if save_crystal_data:
        print(f"\nSaving crystal data to .pt files...")
        
        # Structure: crystal_data.pt contains ground truth data keyed by refcode
        # predictions.pt contains predictions keyed by refcode with seeds
        crystal_data_dict = {}
        predictions_dict = {}
        uma_features_dict = {} if save_uma_features else None
        
        for crystal_data in all_crystals_data:
            refcode = crystal_data['best_metrics'].get('refcode', 'Unknown')
            
            # Ground truth data (same for all seeds)
            crystal_data_dict[refcode] = {
                'atom_types': crystal_data.get('atomic_numbers'),
                'edge_index': crystal_data.get('edge_index'),
                'true_coords': crystal_data.get('true_coords'),
                'true_lattice_params': crystal_data.get('true_lattice'),
                'true_lattice_matrix': lattice_params_to_matrix_torch(crystal_data['true_lattice']).cpu() if crystal_data.get('true_lattice') is not None else None,
            }
            
            # Predictions (per seed)
            seeds_list = []
            uma_seeds_list = [] if save_uma_features else None
            
            for seed_data in crystal_data.get('all_seeds_data', []):
                pred_entry = {
                    'seed_idx': seed_data.get('seed_idx'),
                    'pred_coords': seed_data.get('pred_coords'),
                    'pred_lattice_params': seed_data.get('pred_lattice'),
                    'pred_lattice_matrix': seed_data.get('pred_lattice_matrix'),
                    'pred_atom_types': seed_data.get('pred_atom_types'),
                    'pred_edge_index': seed_data.get('pred_edge_index'),
                    # Relaxed structures (if available)
                    'pred_relaxed_coords': seed_data.get('pred_relaxed_coords'),
                    'pred_relaxed_lattice_matrix': seed_data.get('pred_relaxed_lattice_matrix'),
                }
                seeds_list.append(pred_entry)
                
                # UMA features (if requested)
                if save_uma_features:
                    metrics = seed_data.get('metrics', {})
                    uma_entry = {
                        'seed_idx': seed_data.get('seed_idx'),
                        # Unrelaxed UMA features
                        'unrelaxed_energy': metrics.get('pred_unrelaxed_energy'),
                        'unrelaxed_forces': metrics.get('unrelaxed_forces'),  # Full force vectors [N, 3]
                        'unrelaxed_force_norms': metrics.get('force_norms'),  # Per-atom norms
                        'unrelaxed_max_force': metrics.get('pred_unrelaxed_max_force'),
                        'unrelaxed_mean_force': metrics.get('pred_unrelaxed_mean_force'),
                        'unrelaxed_pressure': metrics.get('pred_unrelaxed_pressure'),
                        'unrelaxed_stress_voigt': metrics.get('unrelaxed_stress_voigt'),
                        # Relaxed UMA features (if relaxation was performed)
                        'relaxed_energy': metrics.get('pred_relaxed_energy'),
                        'relaxed_forces': metrics.get('relaxed_forces'),  # Full force vectors after relaxation
                        'relaxed_force_norms': metrics.get('relaxed_force_norms'),
                        'relaxed_max_force': metrics.get('pred_relaxed_max_force'),
                        'relaxed_mean_force': metrics.get('pred_relaxed_mean_force'),
                        'relaxed_pressure': metrics.get('pred_relaxed_pressure'),
                        'relaxed_stress_voigt': metrics.get('relaxed_stress_voigt'),
                        # AMD descriptors
                        'amd_descriptor': metrics.get('amd_descriptor'),
                        'relaxed_amd_descriptor': metrics.get('relaxed_amd_descriptor'),
                        # Energy trajectory
                        'relaxation_trajectory': metrics.get('relaxation_trajectory'),
                        'relaxation_converged': metrics.get('relaxation_converged'),
                    }
                    uma_seeds_list.append(uma_entry)
            
            predictions_dict[refcode] = {'seeds': seeds_list}
            
            if save_uma_features:
                # Also save ground truth UMA features
                gt_uma = crystal_data.get('all_seeds_data', [{}])[0].get('gt_uma_metrics', {}) if crystal_data.get('all_seeds_data') else {}
                uma_features_dict[refcode] = {
                    'ground_truth': {
                        'unrelaxed_energy': gt_uma.get('unrelaxed_energy'),
                        'unrelaxed_forces': gt_uma.get('unrelaxed_forces'),
                        'unrelaxed_force_norms': gt_uma.get('force_norms'),
                        'unrelaxed_max_force': gt_uma.get('unrelaxed_max_force'),
                        'unrelaxed_mean_force': gt_uma.get('unrelaxed_mean_force'),
                        'unrelaxed_pressure': gt_uma.get('unrelaxed_pressure'),
                        'relaxed_energy': gt_uma.get('relaxed_energy'),
                        'relaxed_forces': gt_uma.get('relaxed_forces'),
                        'relaxed_force_norms': gt_uma.get('relaxed_force_norms'),
                        'relaxed_max_force': gt_uma.get('relaxed_max_force'),
                        'relaxed_mean_force': gt_uma.get('relaxed_mean_force'),
                        'relaxed_pressure': gt_uma.get('relaxed_pressure'),
                        'amd_descriptor': gt_uma.get('amd_descriptor'),
                        'relaxed_amd_descriptor': gt_uma.get('relaxed_amd_descriptor'),
                        # Ground truth relaxed structures
                        'gt_relaxed_coords': crystal_data.get('all_seeds_data', [{}])[0].get('gt_relaxed_coords') if crystal_data.get('all_seeds_data') else None,
                        'gt_relaxed_lattice_matrix': crystal_data.get('all_seeds_data', [{}])[0].get('gt_relaxed_lattice_matrix') if crystal_data.get('all_seeds_data') else None,
                    },
                    'seeds': uma_seeds_list
                }
        
        # Save crystal data (ground truth)
        crystal_data_path = os.path.join(output_dir, "crystal_data.pt")
        torch.save(crystal_data_dict, crystal_data_path)
        print(f"   - Crystal data (ground truth): {crystal_data_path}")
        
        # Save predictions
        predictions_path = os.path.join(output_dir, "predictions.pt")
        torch.save(predictions_dict, predictions_path)
        print(f"   - Predictions: {predictions_path}")
        
        # Save UMA features (if requested)
        if save_uma_features:
            uma_features_path = os.path.join(output_dir, "uma_features.pt")
            torch.save(uma_features_dict, uma_features_path)
            print(f"   - UMA features: {uma_features_path}")
    
    return mean_metrics_result


def print_summary(mean_metrics_result: Dict[str, float]):
    """Print a formatted summary of the results."""
    
    def print_metrics_dict(name, metrics):
        print(f"\n{name.upper()} SUMMARY")
        print("=" * 50)
        
        # Group metrics by category
        density_metrics = {k: v for k, v in metrics.items() if 'density' in k}
        clash_metrics = {k: v for k, v in metrics.items() if 'clash' in k}
        rdf_metrics = {k: v for k, v in metrics.items() if 'rdf' in k}
        
        if density_metrics:
            print("\nDENSITY METRICS:")
            for k, v in density_metrics.items():
                print(f"   {k}: {v:.6f}")
        
        if clash_metrics:
            print("\nCLASH METRICS:")
            for k, v in clash_metrics.items():
                print(f"   {k}: {v:.6f}")
        
        if rdf_metrics:
            print("\nRDF METRICS:")
            for k, v in rdf_metrics.items():
                print(f"   {k}: {v:.6f}")
        
        # AMD metrics
        amd_metrics = {k: v for k, v in metrics.items() if 'amd' in k}
        if amd_metrics:
            print("\nAMD METRICS:")
            for k, v in amd_metrics.items():
                print(f"   {k}: {v:.6f}")
        
        print(f"\nDATASET INFO:")
        print(f"   num_crystals: {metrics.get('num_crystals', 'N/A')}")
        
        # Seed information (only for overall mean usually)
        if 'mean_best_seed_idx' in metrics:
            print(f"   mean_best_seed_idx: {metrics['mean_best_seed_idx']:.2f}")

    # Print Overall
    print_metrics_dict("OVERALL", mean_metrics_result)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate crystal structure prediction model on test set")
    parser.add_argument(
        "--model",
        type=str,
        default="packflow-60M",
        help=(
            "Model type/name (default: packflow-60M). If --checkpoint_path is not "
            "provided, will use default path for this model."
        ),
    )
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to the model checkpoint (optional if --model is specified)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing test.pt")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--n_steps", type=int, default=500,
                       help="Number of sampling steps")
    parser.add_argument("--max_crystals", type=int, default=None,
                       help="Maximum number of crystals to evaluate (for testing)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for data loading")
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="If set, stream-write large JSON outputs incrementally (avoids one huge final write).",
    )
    parser.add_argument("--schedule_type", type=str, default=None,
                       help="Override schedule type from checkpoint")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations comparing original vs predicted crystals")
    parser.add_argument("--num_seeds_per_crystal", type=int, default=1,
                       help="Number of seeds to sample per crystal (metrics will be averaged across all seeds)")
    parser.add_argument("--seed_batch_size", type=int, default=None,
                       help="Batch size for processing seeds (default: None = all seeds at once). Use smaller values to reduce GPU memory usage with large num_seeds.")
    parser.add_argument("--num_monte_carlo_samples", type=int, default=1,
                       help="Number of Monte Carlo samples for top-k analysis")
    parser.add_argument("--lambda_val", type=float, default=1.0,
                       help="Lambda (sampling temperature) for crystal sampling (1.0 = full sampling, 0.0 = no sampling)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for deterministic behavior (default: 42)")
    parser.add_argument("--compute_uma_metrics", action="store_true",
                       help="Compute UMA metrics (energy, forces) (requires fairchem env)")
    parser.add_argument("--uma_relaxation_steps", type=int, default=0, help="Number of UMA relaxation steps for PREDICTED crystals (default: 0)")
    parser.add_argument(
        "--genarris_mode",
        type=str,
        choices=["plain", "rigid_press"],
        default=None,
        help="Run generation using Genarris instead of Flow Matching. Choices: 'plain' or 'rigid_press'."
    )
    parser.add_argument(
        "--genarris_python_path",
        type=str,
        default=os.environ.get("GENARRIS_PYTHON", "python"),
        help="Path to the python executable for the Genarris environment "
             "(or set the GENARRIS_PYTHON env var)."
    )
    parser.add_argument(
        "--genarris_n_procs",
        type=int,
        default=4,
        help="Number of MPI processes to use for Genarris execution."
    )
    parser.add_argument(
        "--csp_blind_test",
        action="store_true",
        help="Run evaluation only on CSP Blind Test refcodes (7 specific crystals from test set)."
    )
    parser.add_argument(
        "--save_crystal_data",
        action="store_true",
        help="Save ground truth and predicted crystal data (atom_types, edge_index, coords, lattice) to .pt files."
    )
    parser.add_argument(
        "--save_uma_features",
        action="store_true",
        help="Save additional UMA features including per-atom energies, full force vectors, and embeddings (requires --compute_uma_metrics)."
    )
    
    args = parser.parse_args()
    
    # Set deterministic seed first
    set_deterministic_seed(args.seed)
    
    print("Crystal Structure Prediction Test Set Evaluation")
    print("=" * 60)
    print(f"Model type: {args.model}")
    
    # Load model
    if args.genarris_mode:
        print(f"Running in Genarris mode: {args.genarris_mode}")
        print(f"Using Genarris Python: {args.genarris_python_path}")
        print(f"Using MPI processes: {args.genarris_n_procs}")
        # Note: GenarrisWrapper is imported at top level
        flow_matching = GenarrisWrapper(mode=args.genarris_mode, python_path=args.genarris_python_path, n_procs=args.genarris_n_procs)
        # Mock device for Genarris (CPU-based mainly, but we use tensors)
        # If user specified cuda, tensors will be on cuda.
        if args.device == "cuda" and not torch.cuda.is_available():
             print("Warning: CUDA requested but not available. Using CPU.")
             args.device = "cpu"
        # In Genarris mode, checkpoint_path is not directly used for model loading,
        # but we might want to log it or use it for output naming.
        # If not provided, set a placeholder.
        if args.checkpoint_path is None:
            args.checkpoint_path = f"genarris_{args.genarris_mode}"
    else:
        # Standard Flow Matching Load
        if not args.checkpoint_path:
            # Try to infer from model name
            if not args.model:
                 raise ValueError("Must provide either --checkpoint_path or --model (or --genarris_mode)")
            args.checkpoint_path = get_default_checkpoint_path(args.model)
            
        flow_matching = load_pretrained_model(
            args.checkpoint_path, 
            device=args.device,
            schedule_type=args.schedule_type,
            model_type=args.model if args.model else "unknown"
        )
    
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Sampling steps: {args.n_steps}")
    print(f"Lambda (sampling temperature): {args.lambda_val}")
    print(f"Seeds per crystal: {args.num_seeds_per_crystal}")
    print(f"Seed batch size: {args.seed_batch_size if args.seed_batch_size else 'All at once'}")
    print(f"Random seed: {args.seed}")
    print(f"Max crystals: {args.max_crystals if args.max_crystals else 'All'}")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled'}")
    print(f"CSP Blind Test mode: {'Enabled (' + str(len(CSP_BLIND_TEST_REFCODES)) + ' refcodes)' if args.csp_blind_test else 'Disabled'}")
    print(f"Save crystal data: {'Enabled' if args.save_crystal_data else 'Disabled'}")
    print(f"Save UMA features: {'Enabled' if args.save_uma_features else 'Disabled'}")
    
    try:
        # Ensure deterministic behavior for model loading
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Load model (if not already loaded in Genarris mode)
        if not args.genarris_mode:
            flow_matching = load_pretrained_model(args.checkpoint_path, args.device, args.schedule_type, args.model)
            
            # Ensure model is in eval mode for deterministic sampling
            flow_matching.model.eval()
        

        
        # Load test data
        refcode_filter = CSP_BLIND_TEST_REFCODES if args.csp_blind_test else None
        data_module, test_loader = load_test_data(args.data_dir, args.batch_size, args.max_crystals, refcode_filter=refcode_filter)
        
        # Get default constants for metrics
        atomic_masses, covalent_radii = get_default_constants()
        print(f"\nUsing default atomic properties for {len(atomic_masses)} elements")
        
        # Evaluate test set
        evaluation_results = evaluate_test_set(
            flow_matching, test_loader, atomic_masses, covalent_radii,
            args.n_steps, args.max_crystals, args.visualize, args.output_dir, 
            args.num_seeds_per_crystal, args.num_monte_carlo_samples, args.lambda_val,
            args.compute_uma_metrics,
            uma_relaxation_steps=args.uma_relaxation_steps,
            model_name=args.model,
            device=args.device,
            seed_batch_size=args.seed_batch_size,
            save_crystal_data=args.save_crystal_data,
            save_uma_features=args.save_uma_features
        )
        
        if not evaluation_results['best_metrics_list']:
            print("ERROR: No successful evaluations!")
            return
        
        # Save results
        mean_metrics_result = save_results(
            evaluation_results['best_metrics_list'], evaluation_results['all_crystals_data'],
            args.output_dir, args.checkpoint_path, args.n_steps, args.data_dir,
            save_crystal_data=args.save_crystal_data,
            save_uma_features=args.save_uma_features,
            chunk=args.chunk,
        )
        
        # Print summary
        print_summary(mean_metrics_result)
        
        print(f"\nEvaluation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
