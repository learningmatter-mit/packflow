#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Trainer for PackFlow

This script implements GRPO finetuning for flow matching models using
UMA energy as the reward signal. The goal is to maximize rewards
(minimize crystal energy) by training the model to generate more
stable crystal structures.

GRPO Algorithm:
1. For each crystal template, sample K structures from the policy
2. Compute rewards (reward = -energy) for all K samples
3. Compute group-relative advantages: A_i = (r_i - mean(r)) / std(r)
4. Update the policy using advantage-weighted flow matching loss

Usage:
    python grpo_trainer.py --experiment_number exp001 --model_type packflow-2M --num_seeds 4 --epochs 10
    # Checkpoints will be saved to grpo_output/exp001/
"""

import os
import sys
import math
import json
import argparse
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

# Setup environment for offline UMA model loading (must be before fairchem import).
# Honors pre-set HF_HOME / FAIRCHEM_CACHE_DIR; otherwise defaults to a ``uma_cache``
# directory under the current working directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_uma_cache = os.environ.get('FAIRCHEM_CACHE_DIR', os.path.join(os.getcwd(), 'uma_cache'))
os.environ.setdefault('HF_HOME', _uma_cache)
os.environ.setdefault('FAIRCHEM_CACHE_DIR', _uma_cache)
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packflow.grpo.model_loader import load_pretrained_model, get_default_checkpoint_path, get_model_config
from packflow.models.model import (
    CrystalTransformerEncoder as CartesianCrystalTransformerEncoder,
    CrystalFlowMatching,
    LatticeTransform,
)
from packflow.data.crystal_datamodule import CrystalDataModule, collate_fn
from packflow.grpo.sampler import sample_batch
from packflow.relaxation.uma import UMAEnergyCalculator, samples_to_crystal_list, CrystalSample
from packflow.grpo.checkpoint_utils import save_packflow_checkpoint, get_checkpoint_configs
from packflow.utils.crystal_utils import lattice_params_to_matrix_torch


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Experiment identifier (required)
    experiment_number: str = ""  # Required: unique experiment identifier
    
    # Model
    model_type: str = "packflow-2M"
    checkpoint_path: Optional[str] = None
    
    # Data
    data_dir: str = "./data"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = 50
    
    # GRPO Hyperparameters
    num_seeds: int = 4  # Number of samples per crystal (K in GRPO)
    advantage_eps: float = 1e-8  # Epsilon for advantage normalization
    advantage_clip: float = 5.0  # Clip advantages to prevent extreme updates
    kl_coef: float = 0.01  # KL penalty coefficient (optional regularization)
    entropy_coef: float = 0.0  # Entropy bonus coefficient
    reward_baseline: str = "group_mean"  # "group_mean", "running_mean", or "none"
    reward_type: str = "e"  # "e" = energy only, "f" = forces only, "ef" = both
    ef_lambda: Optional[float] = None  # Required when reward_type="ef", range [0,1]
    
    # PPO Hyperparameters
    ppo_epochs: int = 4  # Number of optimization epochs per batch
    ppo_clip_range: float = 0.1  # PPO clipping range (epsilon)
    
    # Sampling
    n_steps: int = 100  # ODE integration steps for sampling
    lambda_val: float = 1.0  # Temperature for sampling (1.0 = standard)
    
    # Training
    epochs: int = 10
    batch_size: int = 1  # Crystals per batch (each gets num_seeds samples)
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_epochs: float = 0.5
    
    # Flow matching loss weights
    coords_loss_weight: float = 1.0
    lattice_loss_weight: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    output_dir: str = "./grpo_output"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "packflow-grpo"
    wandb_entity: Optional[str] = None  # Your wandb username or team
    wandb_run_name: Optional[str] = None  # Auto-generated if None
    wandb_tags: Optional[List[str]] = None
    
    # Device
    device: str = "cuda"
    gpu_id: int = 0  # GPU device ID to use
    
    # Reproducibility
    seed: int = 42


class GRPOTrainer:
    """
    GRPO Trainer for PackFlow flow matching models.
    
    Implements Group Relative Policy Optimization where:
    - Multiple samples are generated per input
    - Advantages are computed relative to the group
    - Policy is updated to favor high-reward samples
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        
        # Setup device with specific GPU ID
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = f"cuda:{config.gpu_id}"
            torch.cuda.set_device(config.gpu_id)
            print(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        # Validate required experiment_number
        if not config.experiment_number:
            raise ValueError("experiment_number is required. Use --experiment_number <NAME>")
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Setup output directory using experiment_number
        self.run_name = config.experiment_number
        self.output_dir = os.path.join(config.output_dir, self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        # Save config
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Initialize wandb
        self._setup_wandb()
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_reward_calculator()
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_reward = float('-inf')
        self.running_reward_mean = None
        self.running_reward_std = None
        
        # Logging
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        print(f"\n{'='*60}")
        print(f"GRPO Trainer Initialized")
        print(f"{'='*60}")
        print(f"Experiment: {config.experiment_number}")
        print(f"Model: {config.model_type}")
        print(f"Num seeds per crystal: {config.num_seeds}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Output: {self.output_dir}")
        if self.use_wandb:
            print(f"WandB: {config.wandb_project}/{self.wandb_run_name}")
        print(f"{'='*60}\n")
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        config = self.config
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        
        if config.use_wandb and not WANDB_AVAILABLE:
            print("WARNING: wandb requested but not installed. Run: pip install wandb")
            self.use_wandb = False
            return
        
        if not self.use_wandb:
            self.wandb_run_name = None
            return
        
        # Set wandb to offline mode (no internet access on compute nodes)
        os.environ["WANDB_MODE"] = "offline"
        
        # Store wandb files inside the experiment output folder
        wandb_dir = os.path.join(self.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        
        # Determine run name - FORCE use of experiment_number as requested
        self.wandb_run_name = config.experiment_number
        
        # Use default entity if not specified
        entity = config.wandb_entity if config.wandb_entity else os.environ.get("WANDB_ENTITY")
        
        # Initialize wandb in offline mode
        wandb.init(
            project=config.wandb_project,
            entity=entity,
            name=self.wandb_run_name,
            config=asdict(config),
            tags=config.wandb_tags or [config.model_type, "grpo"],
            dir=wandb_dir,
            resume="allow",
            mode="offline",
        )
        
        # Log code
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        
        print(f"WandB initialized in OFFLINE mode")
        print(f"  Project: {config.wandb_project}")
        print(f"  Entity: {entity}")
        print(f"  Run name: {self.wandb_run_name}")
        print(f"  Offline files: {wandb_dir}")
    
    def _setup_model(self):
        """Load and setup the flow matching model."""
        config = self.config
        
        # Get checkpoint path
        if config.checkpoint_path:
            checkpoint_path = config.checkpoint_path
        else:
            checkpoint_path = get_default_checkpoint_path(config.model_type)
        
        print(f"Loading model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model config
        self.model_config = checkpoint.get('model_config', {})
        self.training_config = checkpoint.get('training_config', {})
        
        if not self.model_config:
            self.model_config = get_model_config(config.model_type)
        
        # Create model
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
        
        encoder_config = {k: v for k, v in self.model_config.items() if k in valid_encoder_args}
        fallback = get_model_config(config.model_type)
        for k, v in fallback.items():
            if k not in encoder_config:
                encoder_config[k] = v
        
        self.model = CartesianCrystalTransformerEncoder(**encoder_config)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.train()
        
        # Store reference model for KL penalty (frozen copy)
        if config.kl_coef > 0:
            print(f"Loading reference model for KL penalty (kl_coef={config.kl_coef})...")
            self.ref_model = CartesianCrystalTransformerEncoder(**encoder_config)
            self.ref_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            print("Reference model loaded and frozen.")
        else:
            self.ref_model = None
            print("KL penalty disabled (kl_coef=0).")
        
        # Create flow matching wrapper for sampling
        self.flow_matching = CrystalFlowMatching(
            model=self.model,
            device=self.device,
            lattice_loss_weight=config.lattice_loss_weight,
            shared_time=self.training_config.get('shared_time', False),
            use_k_basis_representation=self.training_config.get('use_k_basis_representation', False)
        )
        
        self.lattice_transform = LatticeTransform()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _setup_data(self):
        """Setup data loaders."""
        config = self.config
        
        train_path = os.path.join(config.data_dir, "train.pt")
        val_path = os.path.join(config.data_dir, "val.pt")
        
        self.data_module = CrystalDataModule(
            train_path=train_path,
            val_path=val_path if os.path.exists(val_path) else None,
            batch_size=config.batch_size,
            num_workers=0,
            max_samples=config.max_train_samples,
            shuffle_train=True
        )
        self.data_module.setup('fit')
        
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader() if self.data_module.val_dataset else None
        
        print(f"Training samples: {len(self.data_module.train_dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.data_module.val_dataset)}")
    
    def _setup_reward_calculator(self):
        """Setup UMA energy calculator for rewards."""
        print("Loading UMA energy calculator...")
        # UMA/fairchem only accepts "cuda" or "cpu", not "cuda:X"
        uma_device = "cuda" if "cuda" in self.device else "cpu"
        self.uma_calc = UMAEnergyCalculator(device=uma_device)
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        config = self.config
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Warmup + cosine decay schedule
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = int(len(self.train_loader) * config.warmup_epochs)
        
        if warmup_steps > 0:
            # Use warmup + cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_steps
            )
            decay_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps - warmup_steps,
                eta_min=config.learning_rate * 0.01
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps]
            )
        else:
            # No warmup - just cosine decay from full learning rate
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps,
                eta_min=config.learning_rate * 0.01
            )
    
    def _sample_and_compute_rewards(
        self, 
        batch,
        num_seeds: int
    ) -> Tuple[List[List[Tuple[torch.Tensor, torch.Tensor]]], List[List[float]], List[List[float]], List[str]]:
        """
        Sample multiple structures per crystal and compute their rewards.
        
        Returns:
            all_samples: List of samples per crystal, each containing num_seeds (coords, lattice) tuples
            all_energy_rewards: List of energy rewards per crystal (reward = -energy)
            all_force_rewards: List of force rewards per crystal (reward = -force_norm)
            refcodes: List of crystal reference codes
        """
        device = self.device
        config = self.config
        all_samples = []
        all_energy_rewards = []
        all_force_rewards = []
        refcodes = []
        
        # Get batch info - keep on CPU for indexing
        batch_vector = batch.batch  # Keep on CPU
        num_crystals = batch_vector.max().item() + 1
        
        for crystal_idx in range(num_crystals):
            # Extract single crystal from batch (mask on CPU for indexing)
            mask = (batch_vector == crystal_idx)
            
            # Build template - index on CPU, then move to device
            template = {
                'atom_types': batch.atom_types[mask].to(device),
                'edge_index': self._extract_subgraph_edges(batch.edge_index, mask, device),
            }
            
            if hasattr(batch, 'node_features') and batch.node_features is not None:
                template['node_features'] = batch.node_features[mask].to(device)
            
            if hasattr(batch, 'bond_features') and batch.bond_features is not None:
                edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                template['bond_features'] = batch.bond_features[edge_mask].to(device)
            elif hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                template['bond_features'] = batch.edge_attr[edge_mask].to(device)
            
            refcode = batch.refcode[crystal_idx] if hasattr(batch, 'refcode') else f"crystal_{crystal_idx}"
            refcodes.append(refcode)
            
            # Sample multiple structures
            with torch.no_grad():
                samples = sample_batch(
                    self.flow_matching,
                    template,
                    n_steps=config.n_steps,
                    num_seeds=num_seeds,
                    lambda_val=config.lambda_val
                )
            
            # Compute rewards (always compute both for logging purposes)
            crystals = samples_to_crystal_list(samples, template['atom_types'])
            energies, force_norms = self.uma_calc.compute_energies_and_forces_batch(crystals)
            
            energy_rewards = [-e for e in energies]  # reward = -energy
            force_rewards = [-f for f in force_norms]  # reward = -force_norm
            
            all_samples.append(samples)
            all_energy_rewards.append(energy_rewards)
            all_force_rewards.append(force_rewards)
        
        return all_samples, all_energy_rewards, all_force_rewards, refcodes
    
    def _extract_subgraph_edges(self, edge_index: torch.Tensor, node_mask: torch.Tensor, 
                                device: str) -> torch.Tensor:
        """Extract edge_index for a subgraph defined by node_mask."""
        # Do all operations on CPU first, then move to device
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        sub_edges = edge_index[:, edge_mask]
        
        node_indices = torch.where(node_mask)[0]
        old_to_new = torch.full((node_mask.numel(),), -1, dtype=torch.long)
        old_to_new[node_indices] = torch.arange(len(node_indices))
        
        remapped_edges = old_to_new[sub_edges]
        return remapped_edges.to(device)
    
    def _compute_advantages(
        self, 
        energy_rewards: List[List[float]], 
        force_rewards: List[List[float]]
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Compute group-relative advantages for GRPO.
        
        Returns:
            advantages: List of combined advantage tensors
            adv_metrics: Dictionary of advantage statistics for logging
        """
        config = self.config
        advantages = []
        adv_metrics = defaultdict(list)
        
        for i, (e_rewards, f_rewards) in enumerate(zip(energy_rewards, force_rewards)):
            # Compute energy advantages
            if config.reward_type in ["e", "ef"]:
                energy_advs = self._normalize_rewards(e_rewards)
                adv_metrics['energy_advantage'].append(energy_advs.mean().item())
            else:
                energy_advs = None
            
            # Compute force advantages
            if config.reward_type in ["f", "ef"]:
                force_advs = self._normalize_rewards(f_rewards)
                adv_metrics['force_advantage'].append(force_advs.mean().item())
            else:
                force_advs = None
            
            # Combine based on reward_type
            if config.reward_type == "e":
                advs = energy_advs
            elif config.reward_type == "f":
                advs = force_advs
            else:  # "ef"
                # Mixed advantages: MA = ef_lambda * EA + (1 - ef_lambda) * FA
                advs = config.ef_lambda * energy_advs + (1 - config.ef_lambda) * force_advs
            
            # Clip advantages
            advs = torch.clamp(advs, -config.advantage_clip, config.advantage_clip)
            advantages.append(advs)
            adv_metrics['combined_advantage'].append(advs.mean().item())
        
        # Aggregate metrics
        agg_metrics = {k: np.mean(v) for k, v in adv_metrics.items()}
        
        return advantages, agg_metrics
    
    def _normalize_rewards(self, rewards: List[float]) -> torch.Tensor:
        """Normalize rewards using the configured baseline method."""
        config = self.config
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        if config.reward_baseline == "group_mean":
            # Standard GRPO: normalize within group
            mean = rewards_tensor.mean()
            std = rewards_tensor.std(unbiased=False) + config.advantage_eps
            advs = (rewards_tensor - mean) / std
        elif config.reward_baseline == "running_mean":
            # Use running statistics
            if self.running_reward_mean is None:
                self.running_reward_mean = rewards_tensor.mean().item()
                self.running_reward_std = rewards_tensor.std().item() + config.advantage_eps
            else:
                # EMA update
                alpha = 0.1
                self.running_reward_mean = (1 - alpha) * self.running_reward_mean + alpha * rewards_tensor.mean().item()
                self.running_reward_std = (1 - alpha) * self.running_reward_std + alpha * (rewards_tensor.std().item() + config.advantage_eps)
            
            advs = (rewards_tensor - self.running_reward_mean) / self.running_reward_std
        else:
            # No baseline (just use rewards directly)
            advs = rewards_tensor
        
        return advs
    
    def _compute_ppo_loss(
        self,
        batch,
        all_samples: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        advantages: List[torch.Tensor],
        fixed_times: List[List[torch.Tensor]],
        fixed_noises_coords: List[List[torch.Tensor]],
        fixed_noises_lattice: List[List[torch.Tensor]],
        old_losses: Optional[List[List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with importance sampling and clipping.
        
        Args:
            batch: The batch of crystals
            all_samples: Generated samples
            advantages: Computed advantages
            fixed_times: Frozen time steps for each sample
            fixed_noises_coords: Frozen noise for coords
            fixed_noises_lattice: Frozen noise for lattice
            old_losses: Losses from the old policy (for importance sampling).
                        If None, assumes old_loss = new_loss (first epoch).
        """
        device = self.device
        config = self.config
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = defaultdict(float)
        num_samples = 0
        
        # Keep batch_vector on CPU for indexing
        batch_vector = batch.batch
        num_crystals = batch_vector.max().item() + 1
        
        for crystal_idx in range(num_crystals):
            mask = (batch_vector == crystal_idx)
            crystal_advs = advantages[crystal_idx]
            crystal_samples = all_samples[crystal_idx]
            
            # Get fixed conditions
            c_times = fixed_times[crystal_idx]
            c_noises_coords = fixed_noises_coords[crystal_idx]
            c_noises_lattice = fixed_noises_lattice[crystal_idx]
            c_old_losses = old_losses[crystal_idx] if old_losses is not None else None
            
            # Get crystal info - index on CPU, then move to device
            atom_types = batch.atom_types[mask].to(device)
            edge_index = self._extract_subgraph_edges(batch.edge_index, mask, device)
            n_atoms = len(atom_types)
            
            node_features = None
            bond_features = None
            if hasattr(batch, 'node_features') and batch.node_features is not None:
                node_features = batch.node_features[mask].to(device)
            if hasattr(batch, 'bond_features') and batch.bond_features is not None:
                edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                bond_features = batch.bond_features[edge_mask].to(device)
            elif hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                bond_features = batch.edge_attr[edge_mask].to(device)
            
            # Process each sample
            for i, ((coords, lattice), adv) in enumerate(zip(crystal_samples, crystal_advs)):
                # Get conditions for this sample
                t = c_times[i]
                noise_coords = c_noises_coords[i]
                noise_lattice = c_noises_lattice[i]
                
                # Compute FLOW MATCHING loss (our proxy for negative log likelihood)
                # Lower loss = higher probability
                new_flow_loss, kl_penalty, sample_metrics = self._compute_flow_matching_loss_for_sample(
                    coords=coords.to(device),
                    lattice=lattice.to(device),
                    atom_types=atom_types,
                    edge_index=edge_index,
                    node_features=node_features,
                    bond_features=bond_features,
                    n_atoms=n_atoms,
                    fixed_t=t,
                    fixed_noise_coords=noise_coords,
                    fixed_noise_lattice=noise_lattice
                )
                
                # Retrieve old loss (or use current if first step)
                old_flow_loss = c_old_losses[i] if c_old_losses is not None else new_flow_loss.detach()
                
                # Compute Ratio: exp(old_loss - new_loss)
                # Explanation:
                # P = exp(-Energy) ~ exp(-Loss)
                # Ratio = P_new / P_old = exp(-L_new) / exp(-L_old) = exp(L_old - L_new)
                # CRITICAL: This ratio must ONLY be based on the flow matching loss (probability),
                # NOT including the KL penalty term.
                log_ratio = old_flow_loss - new_flow_loss
                
                # Clamp log_ratio to prevent numerical explosion
                # [-5, 5] ensures ratio in range [~0.0067, ~148.4], which is already very wide
                log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
                
                ratio = torch.exp(log_ratio)
                
                # PPO Clipping
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip_range, 1.0 + config.ppo_clip_range) * adv
                
                # Maximize objective => Minimize negative objective
                # PPO objective is to MAXIMIZE min(surr1, surr2)
                # So we minimize -min(surr1, surr2) = max(-surr1, -surr2)
                pg_loss = -torch.min(surr1, surr2)
                
                # Add KL penalty (additive, not weighted by advantage)
                # Loss = PG_Loss + Beta * KL
                policy_loss = pg_loss + config.kl_coef * kl_penalty
                
                total_loss = total_loss + policy_loss
                num_samples += 1
                
                # Accumulate metrics
                metrics['ppo_loss'] += policy_loss.item()
                metrics['ratio'] += ratio.item()
                metrics['clip_frac'] += (1.0 if (ratio.item() < 1.0 - config.ppo_clip_range or ratio.item() > 1.0 + config.ppo_clip_range) else 0.0)
                
                # Add base flow metrics
                for k, v in sample_metrics.items():
                    metrics[k] += v

        # Average over samples
        if num_samples > 0:
            total_loss = total_loss / num_samples
            for k in metrics:
                metrics[k] /= num_samples
        
        metrics['total_loss'] = total_loss.item()
        metrics['num_samples'] = num_samples
        
        return total_loss, dict(metrics)
    
    def _compute_flow_matching_loss_for_sample(
        self,
        coords: torch.Tensor,
        lattice: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        node_features: Optional[torch.Tensor],
        bond_features: Optional[torch.Tensor],
        n_atoms: int,
        fixed_t: Optional[torch.Tensor] = None,
        fixed_noise_coords: Optional[torch.Tensor] = None,
        fixed_noise_lattice: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute flow matching loss for a single sample.
        
        Returns:
            flow_loss: Weighted sum of coords and lattice flow matching loss (NO KL)
            kl_penalty: KL divergence penalty (0 if disabled)
            metrics: Dictionary of individual metrics
        """
        device = self.device
        config = self.config
        
        # Ensure lattice is 2D for batch processing
        if lattice.dim() == 1:
            lattice = lattice.unsqueeze(0)
        
        # Convert lattice to unconstrained representation
        lattice_unconstrained = self.lattice_transform.constrained_to_unconstrained(lattice)
        
        # Sample random time or use fixed time
        if fixed_t is not None:
            t = fixed_t
        else:
            t = torch.rand(1, 1, device=device)
        
        # Sample noise or use fixed noise
        if fixed_noise_coords is not None:
            noise_coords = fixed_noise_coords
        else:
            noise_coords = torch.randn_like(coords)
            
        if fixed_noise_lattice is not None:
            noise_lattice = fixed_noise_lattice
        else:
            noise_lattice = torch.randn_like(lattice_unconstrained)
        
        # # Create noisy samples using OT interpolation: x_t = (1-t)*x_1 + t*noise
        # # Note: in flow matching, we go from noise (t=0) to data (t=1)
        # # So at time t, we have: x_t = (1-t)*noise + t*data
        # alpha = t
        # sigma = 1.0 - t

        sigma = t # swapped from the above because of diffusion formulation
        alpha = 1.0 - t # swapped from the above because of diffusion formulation
        
        noisy_coords = alpha * coords + sigma * noise_coords
        noisy_lattice = alpha * lattice_unconstrained + sigma * noise_lattice
        
        # Target vector field: v = (data - noise) = direction from noise to data
        # target_v_coords = coords - noise_coords # archived
        # target_v_lattice = lattice_unconstrained - noise_lattice # archived
        target_v_coords = noise_coords - coords # reversed from the above because of diffusion formulation
        target_v_lattice = noise_lattice - lattice_unconstrained # reversed from the above because of diffusion formulation
        
        # Create batch vector for single crystal
        batch_vec = torch.zeros(n_atoms, dtype=torch.long, device=device)
        
        # Forward pass
        t_cart = t.expand(1, 1)
        t_lattice = t.expand(1, 1)
        
        pred_v_coords, pred_v_lattice = self.model(
            atom_types=atom_types,
            cart_coords=noisy_coords,
            lattice=noisy_lattice,
            batch=batch_vec,
            t_cart=t_cart,
            t_lattice=t_lattice,
            edge_index=edge_index,
            node_features=node_features,
            bond_features=bond_features
        )
        
        # Compute flow matching losses
        coords_loss = F.mse_loss(pred_v_coords, target_v_coords)
        lattice_loss = F.mse_loss(pred_v_lattice, target_v_lattice)
        
        flow_loss = config.coords_loss_weight * coords_loss + config.lattice_loss_weight * lattice_loss
        
        # Compute KL penalty if enabled (prevents policy from diverging too far from reference)
        kl_penalty = torch.tensor(0.0, device=device)
        if config.kl_coef > 0 and self.ref_model is not None:
            with torch.no_grad():
                ref_v_coords, ref_v_lattice = self.ref_model(
                    atom_types=atom_types,
                    cart_coords=noisy_coords,
                    lattice=noisy_lattice,
                    batch=batch_vec,
                    t_cart=t_cart,
                    t_lattice=t_lattice,
                    edge_index=edge_index,
                    node_features=node_features,
                    bond_features=bond_features
                )
                
                # Compute reference losses
                ref_coords_loss = F.mse_loss(ref_v_coords, target_v_coords)
                ref_lattice_loss = F.mse_loss(ref_v_lattice, target_v_lattice)
                ref_flow_loss = config.coords_loss_weight * ref_coords_loss + config.lattice_loss_weight * ref_lattice_loss
            
            # KL Estimator: d(Delta) = e^Delta - Delta - 1
            # Where Delta = log(pi_ref) - log(pi_theta)
            # Since log(pi) ~ -Loss, Delta = (-Loss_ref) - (-Loss_theta) = Loss_theta - Loss_ref
            delta = flow_loss - ref_flow_loss
            
            # Using the estimator from the prompt: d(Delta) = e^Delta - Delta - 1
            kl_penalty = torch.exp(delta) - delta - 1.0
        
        metrics = {
            'coords_loss': coords_loss.item(),
            'lattice_loss': lattice_loss.item(),
            'kl_penalty': kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty,
            'flow_loss': flow_loss.item()
        }
        
        return flow_loss, kl_penalty, metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of GRPO training."""
        self.model.train()
        config = self.config
        
        epoch_metrics = defaultdict(list)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            self.global_step += 1
            
            # 1. Sample and compute rewards (Generation Phase)
            with torch.no_grad():
                all_samples, all_e_rewards, all_f_rewards, refcodes = self._sample_and_compute_rewards(
                    batch, config.num_seeds
                )
            
            # 2. Compute advantages
            advantages, adv_metrics = self._compute_advantages(all_e_rewards, all_f_rewards)
            
            # 3. Prepare PPO batch (fix time and noise)
            fixed_times = []
            fixed_noises_coords = []
            fixed_noises_lattice = []
            
            for crystal_samples in all_samples:
                c_times = []
                c_n_coords = []
                c_n_lattice = []
                for coords, lattice in crystal_samples:
                    # Generate fixed conditions for this sample
                    # Ensure lattice is 2D
                    if lattice.dim() == 1:
                        lat = lattice.unsqueeze(0)
                    else:
                        lat = lattice
                    
                    lat_unconstrained = self.lattice_transform.constrained_to_unconstrained(lat)
                    
                    c_times.append(torch.rand(1, 1, device=self.device))
                    c_n_coords.append(torch.randn_like(coords))
                    c_n_lattice.append(torch.randn_like(lat_unconstrained))
                
                fixed_times.append(c_times)
                fixed_noises_coords.append(c_n_coords)
                fixed_noises_lattice.append(c_n_lattice)
            
            # 4. Compute "Old" Log Probs (Losses)
            # We do this once before the PPO epochs
            with torch.no_grad():
                # We perform a dedicated pass to get old_losses_vals (flow matching loss only)
                old_losses_vals = []
                batch_vector = batch.batch
                num_crystals = batch_vector.max().item() + 1
                
                for c_idx in range(num_crystals):
                    mask = (batch_vector == c_idx)
                    atom_types = batch.atom_types[mask].to(self.device)
                    edge_index = self._extract_subgraph_edges(batch.edge_index, mask, self.device)
                    n_atoms = len(atom_types)
                    
                    node_features = None
                    bond_features = None
                    if hasattr(batch, 'node_features') and batch.node_features is not None:
                        node_features = batch.node_features[mask].to(self.device)
                    if hasattr(batch, 'bond_features') and batch.bond_features is not None:
                        edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                        bond_features = batch.bond_features[edge_mask].to(self.device)
                    elif hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                        edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                        bond_features = batch.edge_attr[edge_mask].to(self.device)
                    
                    c_losses = []
                    for i, (coords, lattice) in enumerate(all_samples[c_idx]):
                        # Computes flow_loss, kl_penalty, metrics
                        flow_loss, _, _ = self._compute_flow_matching_loss_for_sample(
                            coords.to(self.device), lattice.to(self.device),
                            atom_types, edge_index, node_features, bond_features, n_atoms,
                            fixed_times[c_idx][i], 
                            fixed_noises_coords[c_idx][i], 
                            fixed_noises_lattice[c_idx][i]
                        )
                        c_losses.append(flow_loss.detach())
                    old_losses_vals.append(c_losses)
            
            # 5. PPO Inner Loop
            for ppo_epoch in range(config.ppo_epochs):
                loss, metrics = self._compute_ppo_loss(
                    batch, all_samples, advantages,
                    fixed_times, fixed_noises_coords, fixed_noises_lattice,
                    old_losses=old_losses_vals
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.max_grad_norm
                    )
                    metrics['grad_norm'] = grad_norm.item()
                
                self.optimizer.step()
                
                # Accumulate metrics (average over PPO epochs)
                # Note: this might skew logging if ppo_epochs is large
                if ppo_epoch == config.ppo_epochs - 1:
                     # Log metrics from the LAST PPO step (most relevant)
                    all_e_flat = [r for rewards in all_e_rewards for r in rewards]
                    all_f_flat = [r for rewards in all_f_rewards for r in rewards]
                    
                    # Choose which reward to show as the primary "mean_reward"
                    if config.reward_type == "e":
                        primary_rewards = all_e_flat
                    elif config.reward_type == "f":
                        primary_rewards = all_f_flat
                    else: # "ef"
                        # For mixed, let's use energy as primary for consistency, 
                        # but ideally we log both
                        primary_rewards = all_e_flat
                    
                    metrics['mean_reward'] = np.mean(primary_rewards)
                    metrics['max_reward'] = np.max(primary_rewards)
                    metrics['min_reward'] = np.min(primary_rewards)
                    metrics['reward_std'] = np.std(primary_rewards)
                    metrics['mean_energy'] = -np.mean(all_e_flat)
                    metrics['mean_force_norm'] = -np.mean(all_f_flat)
                    metrics['lr'] = self.scheduler.get_last_lr()[0]
                    
                    # Add advantage metrics
                    for k, v in adv_metrics.items():
                        metrics[k] = v
                    
                    for k, v in metrics.items():
                        epoch_metrics[k].append(v)
            
            self.scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'reward': f"{metrics['mean_reward']:.2f}",
                'lr': f"{metrics['lr']:.2e}"
            })
            
            # Logging
            if self.global_step % config.log_interval == 0:
                self._log_metrics(metrics, prefix="train")
            
            # Evaluation
            if self.val_loader and self.global_step % config.eval_interval == 0:
                val_metrics = self.evaluate()
                self._log_metrics(val_metrics, prefix="val")
                self.model.train()
            
            # Checkpointing
            if self.global_step % config.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step}")
        
        # Aggregate epoch metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        config = self.config
        
        all_e_rewards = []
        all_f_rewards = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            samples, e_rewards, f_rewards, _ = self._sample_and_compute_rewards(
                batch, config.num_seeds
            )
            
            for group_e, group_f in zip(e_rewards, f_rewards):
                all_e_rewards.extend(group_e)
                all_f_rewards.extend(group_f)
        
        # Determine primary rewards for validation tracking
        if config.reward_type == "e":
            primary_rewards = all_e_rewards
        elif config.reward_type == "f":
            primary_rewards = all_f_rewards
        else: # "ef"
            primary_rewards = all_e_rewards
            
        metrics = {
            'mean_reward': np.mean(primary_rewards),
            'max_reward': np.max(primary_rewards),
            'min_reward': np.min(primary_rewards),
            'std_reward': np.std(primary_rewards),
            'mean_energy': -np.mean(all_e_rewards),
            'mean_force_norm': -np.mean(all_f_rewards),
        }
        
        # Track best model
        if metrics['mean_reward'] > self.best_val_reward:
            self.best_val_reward = metrics['mean_reward']
            self._save_checkpoint("best_model")
            print(f"  ★ New best validation reward: {self.best_val_reward:.4f}")
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to console, history, and wandb."""
        # Log to local history
        for k, v in metrics.items():
            self.train_history[f"{prefix}/{k}"].append((self.global_step, v))
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics["global_step"] = self.global_step
            wandb_metrics["epoch"] = self.epoch
            wandb.log(wandb_metrics, step=self.global_step)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.output_dir, "checkpoints", f"{name}.pt")
        
        save_packflow_checkpoint(
            model=self.flow_matching,
            optimizer=self.optimizer,
            epoch=self.epoch,
            loss=self.train_history.get('train/total_loss', [(0, 0)])[-1][1],
            model_config=self.model_config,
            training_config={
                **self.training_config,
                'grpo_config': asdict(self.config)
            },
            save_path=save_path
        )
    
    def train(self):
        """Main training loop."""
        config = self.config
        
        print(f"\n{'='*60}")
        print("Starting GRPO Training")
        print(f"{'='*60}")
        
        for epoch in range(config.epochs):
            self.epoch = epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch summary
            print(f"\nEpoch {epoch + 1}/{config.epochs} Summary:")
            print(f"  Loss: {epoch_metrics.get('total_loss', 0):.4f}")
            print(f"  Mean Reward: {epoch_metrics.get('mean_reward', 0):.4f}")
            print(f"  Max Reward: {epoch_metrics.get('max_reward', 0):.4f}")
            
            # Log epoch metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/loss": epoch_metrics.get('total_loss', 0),
                    "epoch/mean_reward": epoch_metrics.get('mean_reward', 0),
                    "epoch/max_reward": epoch_metrics.get('max_reward', 0),
                    "epoch/min_reward": epoch_metrics.get('min_reward', 0),
                    "epoch/reward_std": epoch_metrics.get('reward_std', 0),
                }, step=self.global_step)
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")
            
            # Save training history
            self._save_history()
        
        # Final save
        self._save_checkpoint("final_model")
        
        # Log final summary to wandb
        if self.use_wandb:
            wandb.run.summary["best_val_reward"] = self.best_val_reward
            wandb.run.summary["total_steps"] = self.global_step
            wandb.run.summary["total_epochs"] = config.epochs
            wandb.finish()
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best validation reward: {self.best_val_reward:.4f}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "train": dict(self.train_history),
                "val": dict(self.val_history),
                "best_val_reward": self.best_val_reward
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for PackFlow")
    
    # Required experiment identifier
    parser.add_argument("--experiment_number", type=str, required=True,
                        help="Experiment identifier (required). Checkpoints saved to grpo_output/<experiment_number>/")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="packflow-2M",
                        help="Model type (packflow-2M, packflow-20M, etc.)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint (optional, uses default if not provided)")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing train.pt, val.pt")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Max training samples (for debugging)")
    
    # GRPO arguments
    parser.add_argument("--num_seeds", type=int, default=4,
                        help="Number of samples per crystal (K in GRPO)")
    parser.add_argument("--advantage_eps", type=float, default=1e-8,
                        help="Epsilon for advantage normalization")
    parser.add_argument("--advantage_clip", type=float, default=5.0,
                        help="Clip advantages to prevent extreme updates")
    parser.add_argument("--kl_coef", type=float, default=0.01,
                        help="KL penalty coefficient (0 to disable)")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="Entropy bonus coefficient")
    parser.add_argument("--reward_baseline", type=str, default="group_mean",
                        choices=["group_mean", "running_mean", "none"],
                        help="Reward baseline type: group_mean, running_mean, or none")
    parser.add_argument("--reward_type", type=str, default="e",
                        choices=["e", "f", "ef"],
                        help="Reward type: 'e'=energy only (default), 'f'=forces only, 'ef'=both")
    parser.add_argument("--ef_lambda", type=float, default=None,
                        help="Energy-force mixing weight (0-1). Required when --reward_type='ef'. "
                             "1.0 = energy only, 0.0 = forces only")
    # PPO arguments
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="Number of PPO optimization epochs per batch")
    parser.add_argument("--ppo_clip_range", type=float, default=0.2,
                        help="PPO clipping range (epsilon)")
    
    # Sampling arguments
    parser.add_argument("--n_steps", type=int, default=100,
                        help="ODE integration steps for sampling")
    parser.add_argument("--lambda_val", type=float, default=1.0,
                        help="Temperature for sampling (1.0 = standard)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (crystals per batch)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--warmup_epochs", type=float, default=0.5,
                        help="Number of warmup epochs for learning rate scheduler")
    
    # Flow matching loss weights
    parser.add_argument("--coords_loss_weight", type=float, default=1.0,
                        help="Weight for coordinate flow matching loss")
    parser.add_argument("--lattice_loss_weight", type=float, default=1.0,
                        help="Weight for lattice flow matching loss")
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Evaluate on validation set every N steps")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="./grpo_output",
                        help="Output directory for checkpoints and logs")
    
    # Weights & Biases arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="packflow-grpo",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None,
                        help="WandB tags (space-separated)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID to use (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Validate reward_type and ef_lambda
    if args.reward_type == "ef":
        if args.ef_lambda is None:
            parser.error("--ef_lambda is required when --reward_type='ef'")
        if not (0 <= args.ef_lambda <= 1):
            parser.error("--ef_lambda must be between 0 and 1")
    
    # Create config from args
    config = GRPOConfig(
        experiment_number=args.experiment_number,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        num_seeds=args.num_seeds,
        advantage_eps=args.advantage_eps,
        advantage_clip=args.advantage_clip,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        reward_baseline=args.reward_baseline,
        reward_type=args.reward_type,
        ef_lambda=args.ef_lambda,
        ppo_epochs=args.ppo_epochs,
        ppo_clip_range=args.ppo_clip_range,
        n_steps=args.n_steps,
        lambda_val=args.lambda_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=args.warmup_epochs,
        coords_loss_weight=args.coords_loss_weight,
        lattice_loss_weight=args.lattice_loss_weight,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        device=args.device,
        gpu_id=args.gpu_id,
        seed=args.seed
    )
    
    # Create trainer and start training
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

