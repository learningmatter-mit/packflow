#!/usr/bin/env python3
"""
Example: GRPO/PPO interface for PackFlow.

This script demonstrates the minimal rollout-and-reward loop used to fine-tune
PackFlow with a policy-gradient method (GRPO/PPO):

1. Load processed crystal data from the dataloader.
2. For each crystal, sample ``num_seeds`` structures from the flow model.
3. Compute UMA energies for all samples and turn them into rewards (reward = -energy).

It is meant as a starting template: plug these rollouts and rewards into your
preferred policy-update step (see the notes printed at the end of the run).
"""

import os
import sys
import torch
from tqdm import tqdm

# Ensure we import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packflow.grpo.model_loader import load_pretrained_model, get_default_checkpoint_path
from packflow.data.crystal_datamodule import CrystalDataset, CrystalDataModule, collate_fn
from packflow.grpo.sampler import prepare_crystal_template, sample_batch
from packflow.relaxation.uma import UMAEnergyCalculator, samples_to_crystal_list


def main():
    # ========== Configuration ==========
    
    # Paths - MODIFY THESE
    MODEL_TYPE = "packflow-ddp"  # Any model in the zoo: packflow-2M, packflow-20M, packflow-ddp, packflow-pa
    
    # Use this data path for training
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(script_dir, "data")
    
    # Sampling parameters
    NUM_SEEDS = 4  # Number of samples per crystal (for PPO, this is rollout size)
    N_STEPS = 500  # ODE integration steps
    LAMBDA_VAL = 1.0  # Temperature (1.0 = normal, >1.0 = colder)
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # How many crystals to process (set to None for all)
    MAX_CRYSTALS = 5  # Small number for demo
    
    print("=" * 60)
    print("PackFlow PPO Interface Example")
    print("=" * 60)
    
    # ========== 1. Load Flow Model ==========
    print("\n[1/4] Loading flow matching model...")
    
    checkpoint_path = get_default_checkpoint_path(MODEL_TYPE)
    flow_model = load_pretrained_model(checkpoint_path, device=DEVICE, model_type=MODEL_TYPE)
    
    # ========== 2. Load Data ==========
    print("\n[2/4] Loading crystal data...")
    
    train_path = os.path.join(DATA_DIR, "train.pt")
    data_module = CrystalDataModule(
        train_path=train_path,
        batch_size=1,  # Process one crystal at a time
        num_workers=0,
        max_samples=MAX_CRYSTALS
    )
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    
    print(f"Loaded {len(data_module.train_dataset)} crystals")
    
    # ========== 3. Load UMA Calculator ==========
    print("\n[3/4] Loading UMA energy calculator...")
    
    try:
        uma_calc = UMAEnergyCalculator(device=DEVICE)
    except Exception as e:
        print(f"ERROR: Could not load UMA calculator: {e}")
        print("Make sure you're running in the fairchem environment!")
        sys.exit(1)
    
    # ========== 4. Main Loop: Sample + Compute Rewards ==========
    print("\n[4/4] Sampling and computing rewards...")
    print(f"  - Processing {MAX_CRYSTALS or 'all'} crystals")
    print(f"  - Generating {NUM_SEEDS} samples per crystal")
    print(f"  - Using {N_STEPS} ODE integration steps")
    print()
    
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing crystals")):
        # Get single crystal from batch
        crystal_data = batch  # batch_size=1, so batch IS the crystal
        
        # Prepare template for sampling
        # Note: batch is a PyG Batch object, we need the first (only) crystal
        crystal_template = {
            'atom_types': crystal_data.atom_types.to(DEVICE),
            'edge_index': crystal_data.edge_index.to(DEVICE),
        }
        if hasattr(crystal_data, 'node_features') and crystal_data.node_features is not None:
            crystal_template['node_features'] = crystal_data.node_features.to(DEVICE)
        if hasattr(crystal_data, 'bond_features') and crystal_data.bond_features is not None:
            crystal_template['bond_features'] = crystal_data.bond_features.to(DEVICE)
        elif hasattr(crystal_data, 'edge_attr') and crystal_data.edge_attr is not None:
            crystal_template['bond_features'] = crystal_data.edge_attr.to(DEVICE)
        
        refcode = crystal_data.refcode[0] if hasattr(crystal_data, 'refcode') else f"crystal_{batch_idx}"
        
        # Sample multiple structures
        try:
            samples = sample_batch(
                flow_model,
                crystal_template,
                n_steps=N_STEPS,
                num_seeds=NUM_SEEDS,
                lambda_val=LAMBDA_VAL
            )
        except Exception as e:
            print(f"  Sampling failed for {refcode}: {e}")
            continue
        
        # Convert to CrystalSample objects and compute energies
        atom_types = crystal_template['atom_types']
        crystal_samples = samples_to_crystal_list(samples, atom_types)
        
        # Compute energies (this is the reward signal: reward = -energy)
        try:
            energies = uma_calc.compute_energies_batch(crystal_samples)
            rewards = [-e for e in energies]  # reward = -energy
        except Exception as e:
            print(f"  Energy computation failed for {refcode}: {e}")
            continue
        
        # Store results
        result = {
            'refcode': refcode,
            'num_atoms': len(atom_types),
            'energies': energies,
            'rewards': rewards,
            'best_reward': max(rewards),
            'mean_reward': sum(rewards) / len(rewards),
        }
        all_results.append(result)
        
        # Print summary for this crystal
        print(f"\n  {refcode} ({len(atom_types)} atoms):")
        print(f"    Energies: {[f'{e:.2f}' for e in energies]} eV")
        print(f"    Rewards:  {[f'{r:.2f}' for r in rewards]}")
        print(f"    Best reward: {max(rewards):.2f}")
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_results:
        all_rewards = [r for res in all_results for r in res['rewards']]
        print(f"Total samples generated: {len(all_rewards)}")
        print(f"Mean reward across all: {sum(all_rewards)/len(all_rewards):.2f}")
        print(f"Best reward overall: {max(all_rewards):.2f}")
        print(f"Worst reward overall: {min(all_rewards):.2f}")
    else:
        print("No results - check errors above")
    
    print("\n" + "=" * 60)
    print("PPO Training Integration Notes:")
    print("=" * 60)
    print("""
For PPO/GRPO training:

1. The samples from sample_batch() are your policy rollouts
2. The rewards (= -energy) from UMA are your reward signal
3. Lower energy = better structure = higher reward

Key components:
- flow_model.sample() is differentiable through the ODE integrator
- For policy gradient, you'll need to track the sampling trajectory
- Consider using the 'whole_cartesian_coords' from the template as baseline

Example PPO loop structure:
    for epoch in range(num_epochs):
        for batch in loader:
            # 1. Sample and Compute rewards
            samples = sample_batch(flow_model, template, num_seeds=K)
            rewards = -uma_calc.compute_energies_batch(samples, atom_types)
            
            # 2. Update model (PPO implementation)
            # update_policy(samples, rewards, ...)
            
        # 3. Save standardized checkpoint (IMPORTANT for evaluations)
        from packflow.grpo.checkpoint_utils import save_packflow_checkpoint, get_checkpoint_configs
        
        # Get original configs from existing checkpoint to maintain consistency
        model_cfg, train_cfg = get_checkpoint_configs(checkpoint_path)
        
        save_packflow_checkpoint(
            model=flow_model,
            optimizer=None, # pass your optimizer here
            epoch=epoch,
            loss=avg_loss,
            model_config=model_cfg,
            training_config=train_cfg,
            save_path=f"checkpoints/ppo_trained_epoch_{epoch}.pt"
        )
""")


if __name__ == "__main__":
    main()
