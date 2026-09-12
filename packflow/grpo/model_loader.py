#!/usr/bin/env python3
"""
Model Loader for PackFlow PPO Interface

Loads any packflow checkpoint and returns a CrystalFlowMatching wrapper
ready for sampling.

Extracted from evaluate_test_set_metrics.py
"""

import os
import torch
from typing import Dict, Optional

from packflow.models.model import (
    CrystalTransformerEncoder as CartesianCrystalTransformerEncoder,
    CrystalFlowMatching,
)


def get_model_config(model_type: str) -> Dict:
    """Get model configuration based on model type."""
    
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
        'attn_bias_combine': 'sum',
        'attn_bias_baseline': 0.0,
        'bias_scale': 1.0,
        'learnable_bias_scale': False,
        'lattice_token': False,
        'periodic_coord_emb': False,
        'periodic_nmax': 5,
        'periodic_topk': 512,
        'use_fractional_coords': False,
        'use_rdkit_features': True,
        'use_positional_embeddings': False,
        'use_attention_bias_from_graph': True,
    }
    
    configs = {
        "packflow-2M": {'d_model': 256, 'nhead': 8, 'dim_feedforward': 512, 'num_layers': 4},
        "packflow-20M": {'d_model': 320, 'nhead': 8, 'dim_feedforward': 1280, 'num_layers': 12},
        "packflow-60M": {'d_model': 640, 'nhead': 10, 'dim_feedforward': 2560, 'num_layers': 12},
        "packflow-60M-no-bonds": {'d_model': 640, 'nhead': 10, 'dim_feedforward': 2560, 'num_layers': 12, 
                                   'use_attention_bias_from_graph': False},
        "packflow-60M-bond-aux-loss": {'d_model': 640, 'nhead': 10, 'dim_feedforward': 2560, 'num_layers': 12},
        "packflow-60M-shared-time": {'d_model': 640, 'nhead': 10, 'dim_feedforward': 2560, 'num_layers': 12},
        "packflow-60M-periodic-aux-loss": {'d_model': 640, 'nhead': 10, 'dim_feedforward': 2560, 'num_layers': 12},
        "packflow-80M": {'d_model': 800, 'nhead': 10, 'dim_feedforward': 3072, 'num_layers': 10},
        "packflow-ddp": {'d_model': 640, 'nhead': 10, 'dim_feedforward': 2560, 'num_layers': 12},
        "packflow-ddp-4n": {'d_model': 640, 'nhead': 10, 'dim_feedforward': 2560, 'num_layers': 12},
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {list(configs.keys())}")
    
    config = base_config.copy()
    config.update(configs[model_type])
    return config


def get_default_checkpoint_path(model_type: str) -> str:
    """Get default checkpoint path based on model type.

    Looks first in the curated package model-zoo
    (``packflow/checkpoints/<model_type>/best_model.pt``), then in a configurable
    raw training-run directory (``$PACKFLOW_EXPERIMENTS_DIR``).
    """
    # packflow/grpo/model_loader.py -> package dir is the grandparent of this file.
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_root = os.path.join(package_dir, "checkpoints")

    # Check if local (model-zoo) checkpoint exists
    local_path = os.path.join(checkpoint_root, model_type, "best_model.pt")
    if os.path.exists(local_path):
        return local_path

    # Fallback to raw training-run directories (for ablation variants).
    experiments_dir = os.environ.get(
        "PACKFLOW_EXPERIMENTS_DIR",
        os.path.join(os.path.dirname(package_dir), "experiments"),
    )
    run_folders = {
        "packflow-2M": "20251130-044730_packflow-2M_jid3797662",
        "packflow-20M": "20251130-044907_packflow-20M_jid3797673",
        "packflow-60M": "20251130-052224_packflow-60M_jid3797682",
        "packflow-80M": "20251130-052224_packflow-80M_jid3797683",
        "packflow-ddp": "20251201-011524_packflow-ddp_jid3871200",
    }
    
    if model_type in run_folders:
        ckpt_path = os.path.join(experiments_dir, run_folders[model_type], "checkpoints", "best_model.pt")
        if os.path.exists(ckpt_path):
            return ckpt_path
    
    raise ValueError(f"Checkpoint for model type '{model_type}' not found locally or in experiments dir.")


def load_pretrained_model(
    checkpoint_path: str, 
    device: str = 'cuda',
    model_type: str = "packflow-60M"
) -> CrystalFlowMatching:
    """
    Load a pretrained PackFlow model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load model on ('cuda' or 'cpu')
        model_type: Model type for architecture config (used as fallback)
        
    Returns:
        CrystalFlowMatching wrapper ready for sampling
    """
    print(f"Loading pretrained model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or fallback
    checkpoint_model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    
    valid_encoder_args = {
        'd_model', 'nhead', 'dim_feedforward', 'activation', 'dropout', 'norm_first', 'bias',
        'num_layers', 'time_embed_dim', 'cart_coords_dim', 'lattice_dim',
        'use_attention_bias_from_graph', 'attn_bias_heads', 'attn_bias_combine',
        'attn_bias_baseline', 'bias_scale', 'learnable_bias_scale',
        'use_positional_embeddings', 'use_rdkit_features', 'lattice_token',
        'rdkit_bond_feat_dim', 'rdkit_node_feat_dim',
        'periodic_coord_emb', 'periodic_nmax', 'periodic_topk',
        'use_fractional_coords'
    }
    
    if checkpoint_model_config:
        print("Using model configuration from checkpoint")
        final_config = {k: v for k, v in checkpoint_model_config.items() if k in valid_encoder_args}
        fallback_config = get_model_config(model_type)
        for k, v in fallback_config.items():
            if k not in final_config:
                final_config[k] = v
    else:
        print(f"Using get_model_config for model type: {model_type}")
        final_config = get_model_config(model_type)
    
    print(f"Creating model with config: d_model={final_config.get('d_model')}, "
          f"num_layers={final_config.get('num_layers')}")
    
    # Create model
    model = CartesianCrystalTransformerEncoder(**final_config)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed, trying flexible: {e}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Create flow matching wrapper. Sampling-relevant flags are restored from the
    # checkpoint's training config so the model behaves exactly as it did in training.
    lattice_loss_weight = checkpoint.get('lattice_loss_weight', 1.0)
    shared_time = checkpoint.get('shared_time', False)
    use_k_basis = training_config.get('use_k_basis_representation', False)

    flow_matching = CrystalFlowMatching(
        model=model,
        device=device,
        lattice_loss_weight=lattice_loss_weight,
        shared_time=shared_time,
        use_k_basis_representation=use_k_basis,
        add_periodic_edges=training_config.get('add_periodic_edges', False),
        periodic_edge_cutoff=training_config.get('periodic_edge_cutoff', 5.0),
        periodic_edge_periodic=training_config.get('periodic_edge_periodic', True),
        periodic_edge_time_cutoff=training_config.get('periodic_edge_time_cutoff', 0.5),
    )
    
    print(f"Model loaded! Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Device: {device}")
    print(f"  - Shared time: {shared_time}")
    
    return flow_matching


def load_model_by_type(model_type: str, device: str = 'cuda') -> CrystalFlowMatching:
    """
    Convenience function to load model by type name.
    
    Args:
        model_type: One of 'packflow-2M', 'packflow-20M', 'packflow-60M', etc.
        device: Device to load on
        
    Returns:
        CrystalFlowMatching wrapper
    """
    checkpoint_path = get_default_checkpoint_path(model_type)
    return load_pretrained_model(checkpoint_path, device, model_type)
