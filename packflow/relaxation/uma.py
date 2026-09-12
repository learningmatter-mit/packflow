#!/usr/bin/env python3
"""
UMA Energy Calculator for PackFlow PPO Interface

Computes energies for crystal structures using the UMA (Universal Materials Approximator)
potential from fairchem. Energy is used as the reward signal for PPO training:
    reward = -energy

NOTE: Requires the packflow_ppo conda environment with both fairchem and torch_geometric.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

# These imports require the combined environment
try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from ase import Atoms
    FAIRCHEM_AVAILABLE = True
except ImportError:
    FAIRCHEM_AVAILABLE = False
    print("WARNING: fairchem not available. Install with: conda env create -f environment.yml")

from packflow.utils.crystal_utils import lattice_params_to_matrix_torch


@dataclass
class CrystalSample:
    """Container for a crystal sample with coordinates and lattice."""
    cart_coords: torch.Tensor  # [N, 3] Cartesian coordinates
    lattice_params: torch.Tensor  # [6] a, b, c, alpha, beta, gamma
    atom_types: torch.Tensor  # [N] atomic numbers


def samples_to_crystal_list(
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    atom_types: torch.Tensor
) -> List[CrystalSample]:
    """
    Convert list of (coords, lattice) tuples to CrystalSample list.
    
    Args:
        samples: List of (cart_coords, lattice_params) from sampler
        atom_types: Atomic numbers for the crystal template
        
    Returns:
        List of CrystalSample objects
    """
    return [
        CrystalSample(
            cart_coords=coords,
            lattice_params=lattice,
            atom_types=atom_types
        )
        for coords, lattice in samples
    ]


class UMAEnergyCalculator:
    """
    Calculator for computing crystal energies using UMA potential.
    
    The UMA model is loaded once and reused for all computations.
    For PPO: reward = -energy (lower energy = better structure)
    """
    
    def __init__(self, device: str = 'cuda', model_name: str = 'uma-s-1p1'):
        """
        Initialize UMA calculator.
        
        Args:
            device: 'cuda' or 'cpu'
            model_name: UMA model variant (default: 'uma-s-1p1')
        """
        if not FAIRCHEM_AVAILABLE:
            raise RuntimeError("fairchem not available. Create env with: conda env create -f environment.yml")
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading UMA model '{model_name}' on {device}...")
        self.predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        self.task_name = "omc"  # organic molecular crystals
        print("UMA model loaded successfully!")
    
    def crystal_sample_to_ase(self, sample: CrystalSample) -> 'Atoms':
        """Convert CrystalSample to ASE Atoms object."""
        coords = sample.cart_coords.cpu().numpy()
        lattice_params = sample.lattice_params.cpu().numpy()
        atom_numbers = sample.atom_types.cpu().numpy()
        
        # Build lattice matrix from params
        lattice_matrix = lattice_params_to_matrix_torch(
            torch.tensor(lattice_params).unsqueeze(0)
        ).squeeze(0).numpy()
        
        atoms = Atoms(
            numbers=atom_numbers,
            positions=coords,
            cell=lattice_matrix,
            pbc=True
        )
        
        return atoms
    
    def compute_energy(self, sample: CrystalSample) -> float:
        """
        Compute energy for a single crystal sample.
        
        Args:
            sample: CrystalSample object
            
        Returns:
            energy: Total potential energy (eV)
        """
        atoms = self.crystal_sample_to_ase(sample)
        calc = FAIRChemCalculator(self.predictor, task_name=self.task_name)
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        return float(energy)
    
    def compute_energy_and_forces(self, sample: CrystalSample) -> Tuple[float, np.ndarray]:
        """
        Compute energy and forces for a single crystal sample.
        
        Args:
            sample: CrystalSample object
            
        Returns:
            energy: Total potential energy (eV)
            forces: [N, 3] Force vectors (eV/Angstrom)
        """
        atoms = self.crystal_sample_to_ase(sample)
        calc = FAIRChemCalculator(self.predictor, task_name=self.task_name)
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        return float(energy), forces
    
    def compute_energies_batch(
        self,
        samples: List[Union[CrystalSample, Tuple[torch.Tensor, torch.Tensor]]],
        atom_types: Optional[torch.Tensor] = None
    ) -> List[float]:
        """
        Compute energies for a batch of crystal samples.
        
        This is the primary function for PPO reward computation.
        The reward should be: reward = -energy
        
        Args:
            samples: List of CrystalSample objects OR list of (coords, lattice) tuples
            atom_types: Required if samples are tuples (provides atomic numbers)
            
        Returns:
            List of energies (one per sample)
        """
        # Convert tuples to CrystalSample if needed
        if samples and isinstance(samples[0], tuple):
            if atom_types is None:
                raise ValueError("atom_types required when samples are tuples")
            samples = samples_to_crystal_list(samples, atom_types)
        
        energies = []
        for sample in samples:
            try:
                energy = self.compute_energy(sample)
                energies.append(energy)
            except Exception as e:
                print(f"WARNING: Energy computation failed: {e}")
                # Return high energy as penalty for failed structures
                energies.append(1e10)
        
        return energies
    
    def compute_energies_and_forces_batch(
        self,
        samples: List[Union[CrystalSample, Tuple[torch.Tensor, torch.Tensor]]],
        atom_types: Optional[torch.Tensor] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Compute energies AND force norms for a batch of crystal samples.
        
        Args:
            samples: List of CrystalSample objects OR list of (coords, lattice) tuples
            atom_types: Required if samples are tuples (provides atomic numbers)
            
        Returns:
            energies: List of energies (one per sample)
            force_norms: List of L2 force norms (one per sample) - computed as 
                         mean of per-atom force magnitudes
        """
        # Convert tuples to CrystalSample if needed
        if samples and isinstance(samples[0], tuple):
            if atom_types is None:
                raise ValueError("atom_types required when samples are tuples")
            samples = samples_to_crystal_list(samples, atom_types)
        
        energies = []
        force_norms = []
        for sample in samples:
            try:
                energy, forces = self.compute_energy_and_forces(sample)
                energies.append(energy)
                # Compute L2 norm: mean of per-atom force magnitudes
                # forces shape: [N_atoms, 3]
                per_atom_force_mag = np.linalg.norm(forces, axis=1)  # [N_atoms]
                force_norm = float(np.mean(per_atom_force_mag))  # scalar
                force_norms.append(force_norm)
            except Exception as e:
                print(f"WARNING: Energy/forces computation failed: {e}")
                # Return high energy and force as penalty for failed structures
                energies.append(1e10)
                force_norms.append(1e10)
        
        return energies, force_norms
    
    def compute_rewards_batch(
        self,
        samples: List[Union[CrystalSample, Tuple[torch.Tensor, torch.Tensor]]],
        atom_types: Optional[torch.Tensor] = None,
        normalize: bool = False,
        normalization_offset: float = 0.0
    ) -> torch.Tensor:
        """
        Compute rewards for PPO training.
        
        reward = -(energy - normalization_offset)
        
        Args:
            samples: List of samples
            atom_types: Atomic numbers (required if samples are tuples)
            normalize: If True, subtract normalization_offset from energies
            normalization_offset: Energy offset for normalization
            
        Returns:
            rewards: Tensor of rewards [num_samples]
        """
        energies = self.compute_energies_batch(samples, atom_types)
        energies_tensor = torch.tensor(energies, dtype=torch.float32)
        
        if normalize:
            energies_tensor = energies_tensor - normalization_offset
        
        rewards = -energies_tensor
        return rewards


def create_uma_calculator(device: str = 'cuda') -> UMAEnergyCalculator:
    """Factory function to create UMA calculator."""
    return UMAEnergyCalculator(device=device)


# Standalone function for simple usage
def compute_uma_energy(
    cart_coords: torch.Tensor,
    lattice_params: torch.Tensor,
    atom_types: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Compute UMA energy for a single crystal structure.
    
    Args:
        cart_coords: [N, 3] Cartesian coordinates
        lattice_params: [6] a, b, c, alpha, beta, gamma
        atom_types: [N] atomic numbers
        device: Device for UMA model
        
    Returns:
        energy: Potential energy in eV
    """
    calc = UMAEnergyCalculator(device=device)
    sample = CrystalSample(cart_coords, lattice_params, atom_types)
    return calc.compute_energy(sample)
