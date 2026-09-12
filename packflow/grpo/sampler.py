#!/usr/bin/env python3
"""
Sampler for PackFlow PPO Interface

Provides batched sampling functions for generating multiple crystal structures
from a flow matching model.

This is the primary interface for PPO policy rollouts.
"""

import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional

from packflow.models.model import CrystalFlowMatching


def prepare_crystal_template(crystal_data_obj, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Convert a PyG Data object (from CrystalDataset) to a template dict for sampling.
    
    Args:
        crystal_data_obj: PyG Data object from CrystalDataset
        device: Target device
        
    Returns:
        Dict with keys needed for sampling
    """
    template = {
        'atom_types': crystal_data_obj.atom_types.to(device),
        'edge_index': crystal_data_obj.edge_index.to(device),
    }
    
    # Optional features
    if hasattr(crystal_data_obj, 'node_features') and crystal_data_obj.node_features is not None:
        template['node_features'] = crystal_data_obj.node_features.to(device)
    
    if hasattr(crystal_data_obj, 'bond_features') and crystal_data_obj.bond_features is not None:
        template['bond_features'] = crystal_data_obj.bond_features.to(device)
    elif hasattr(crystal_data_obj, 'edge_attr') and crystal_data_obj.edge_attr is not None:
        template['bond_features'] = crystal_data_obj.edge_attr.to(device)
    
    # Metadata for reference
    if hasattr(crystal_data_obj, 'refcode'):
        template['refcode'] = crystal_data_obj.refcode
    if hasattr(crystal_data_obj, 'lattice_1'):
        template['ground_truth_lattice'] = crystal_data_obj.lattice_1.to(device)
    if hasattr(crystal_data_obj, 'whole_cartesian_coords'):
        template['ground_truth_coords'] = crystal_data_obj.whole_cartesian_coords.to(device)
    
    return template


def sample_single(
    flow_model: CrystalFlowMatching,
    crystal_template: Dict[str, torch.Tensor],
    n_steps: int = 500,
    lambda_val: float = 1.0,
    coords_time_grid_type: str = "linear",
    lattice_time_grid_type: str = "linear"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a single crystal structure.
    
    Args:
        flow_model: Loaded CrystalFlowMatching model
        crystal_template: Template dict from prepare_crystal_template
        n_steps: Number of ODE integration steps
        lambda_val: Temperature parameter (1.0 = normal, >1.0 = colder/more deterministic)
        
    Returns:
        cart_coords: [N, 3] Cartesian coordinates
        lattice_params: [6] Lattice parameters [a, b, c, alpha, beta, gamma]
    """
    return flow_model.sample(
        crystal_template,
        n_steps=n_steps,
        low_temperature_lambda=lambda_val,
        coords_time_grid_type=coords_time_grid_type,
        lattice_time_grid_type=lattice_time_grid_type
    )


def sample_batch(
    flow_model: CrystalFlowMatching,
    crystal_template: Dict[str, torch.Tensor],
    n_steps: int = 500,
    num_seeds: int = 8,
    lambda_val: float = 1.0,
    coords_time_grid_type: str = "linear",
    lattice_time_grid_type: str = "linear"
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample multiple crystal structures (seeds) for the same template in parallel.
    
    This is the primary function for PPO rollouts - generates multiple samples
    from the policy (flow model) for reward computation.
    
    Args:
        flow_model: Loaded CrystalFlowMatching model
        crystal_template: Template dict from prepare_crystal_template
        n_steps: Number of ODE integration steps
        num_seeds: Number of samples to generate
        lambda_val: Temperature parameter
        
    Returns:
        List of (cart_coords, lattice_params) tuples, one per seed
    """
    device = flow_model.device
    
    # Create batched data by replicating template
    data_obj = Data(
        atom_types=crystal_template['atom_types'],
        edge_index=crystal_template['edge_index'],
        num_nodes=len(crystal_template['atom_types'])
    )
    
    if 'node_features' in crystal_template and crystal_template['node_features'] is not None:
        data_obj.node_features = crystal_template['node_features']
    if 'bond_features' in crystal_template and crystal_template['bond_features'] is not None:
        data_obj.bond_features = crystal_template['bond_features']
    
    # Replicate for num_seeds
    data_list = [data_obj] * num_seeds
    batch_data = Batch.from_data_list(data_list).to(device)
    
    # Build batched template
    batch_template = {
        'atom_types': batch_data.atom_types,
        'edge_index': batch_data.edge_index,
        'batch': batch_data.batch,
    }
    if hasattr(batch_data, 'node_features'):
        batch_template['node_features'] = batch_data.node_features
    if hasattr(batch_data, 'bond_features'):
        batch_template['bond_features'] = batch_data.bond_features
    
    # Sample
    with torch.no_grad():
        cart_coords_all, lattice_all = flow_model.sample(
            batch_template,
            n_steps=n_steps,
            low_temperature_lambda=lambda_val,
            coords_time_grid_type=coords_time_grid_type,
            lattice_time_grid_type=lattice_time_grid_type
        )
    
    # Unbatch results
    results = []
    batch_vector = batch_data.batch
    
    # Handle lattice shape
    if lattice_all.dim() == 1:
        lattice_all = lattice_all.unsqueeze(0)
    
    for i in range(num_seeds):
        mask = (batch_vector == i)
        coords = cart_coords_all[mask]
        lattice = lattice_all[i]
        results.append((coords, lattice))
    
    return results


def sample_from_dataloader_batch(
    flow_model: CrystalFlowMatching,
    batch_data,
    n_steps: int = 500,
    num_seeds_per_crystal: int = 4,
    lambda_val: float = 1.0
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Sample from a batch of crystals from the dataloader.
    
    For each crystal in the batch, generates num_seeds_per_crystal samples.
    
    Args:
        flow_model: Loaded CrystalFlowMatching model
        batch_data: PyG Batch object from dataloader
        n_steps: ODE integration steps
        num_seeds_per_crystal: Seeds per crystal
        lambda_val: Temperature
        
    Returns:
        List of lists: outer list is per crystal, inner list is per seed
        Each element is (cart_coords, lattice_params)
    """
    device = flow_model.device
    
    # Get number of crystals in batch
    batch_vector = batch_data.batch.to(device)
    num_crystals = batch_vector.max().item() + 1
    
    all_results = []
    
    for crystal_idx in range(num_crystals):
        # Extract single crystal from batch
        mask = (batch_vector == crystal_idx)
        
        crystal_template = {
            'atom_types': batch_data.atom_types[mask].to(device),
            'edge_index': _extract_subgraph_edges(batch_data.edge_index, mask, device),
        }
        
        if hasattr(batch_data, 'node_features') and batch_data.node_features is not None:
            crystal_template['node_features'] = batch_data.node_features[mask].to(device)
        if hasattr(batch_data, 'bond_features') and batch_data.bond_features is not None:
            # Need to filter bond features by edge mask
            edge_mask = mask[batch_data.edge_index[0]] & mask[batch_data.edge_index[1]]
            crystal_template['bond_features'] = batch_data.bond_features[edge_mask].to(device)
        elif hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
            edge_mask = mask[batch_data.edge_index[0]] & mask[batch_data.edge_index[1]]
            crystal_template['bond_features'] = batch_data.edge_attr[edge_mask].to(device)
        
        # Sample multiple seeds for this crystal
        seeds = sample_batch(
            flow_model, crystal_template, n_steps, num_seeds_per_crystal, lambda_val
        )
        all_results.append(seeds)
    
    return all_results


def _extract_subgraph_edges(edge_index: torch.Tensor, node_mask: torch.Tensor, 
                            device: str) -> torch.Tensor:
    """Extract edge_index for a subgraph defined by node_mask."""
    # Get edges where both endpoints are in the mask
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    sub_edges = edge_index[:, edge_mask]
    
    # Remap node indices to be contiguous
    node_indices = torch.where(node_mask)[0]
    old_to_new = torch.full((node_mask.numel(),), -1, dtype=torch.long, device=device)
    old_to_new[node_indices] = torch.arange(len(node_indices), device=device)
    
    remapped_edges = old_to_new[sub_edges]
    return remapped_edges
