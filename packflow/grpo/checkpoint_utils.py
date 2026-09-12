import torch
import os
from typing import Dict, Any, Optional

def save_packflow_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    loss: float,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    save_path: str,
    lattice_loss_weight: Optional[float] = None,
    shared_time: Optional[bool] = None
):
    """
    Save a checkpoint in the exact format used by original PackFlow experiments.
    This ensures compatibility with the standard evaluation scripts.
    
    Args:
        model: CrystalFlowMatching wrapper or underlying transformer model
        optimizer: The optimizer used for training
        epoch: Current epoch number
        loss: Current loss value
        model_config: Model architecture parameters
        training_config: Training hyperparameters
        save_path: Where to save the checkpoint
        lattice_loss_weight: (Optional) Weight for lattice loss. Auto-extracted if model is wrapper.
        shared_time: (Optional) Whether time is shared. Auto-extracted if model is wrapper.
    """
    # Extract state dict and parameters if model is the wrapper
    if hasattr(model, 'model'):
        # It's a CrystalFlowMatching wrapper
        inner_model = model.model
        # Handle DistributedDataParallel unwrap
        if hasattr(inner_model, 'module'):
            state_dict = inner_model.module.state_dict()
        else:
            state_dict = inner_model.state_dict()
            
        if lattice_loss_weight is None:
            lattice_loss_weight = getattr(model, 'lattice_loss_weight', 1.0)
        if shared_time is None:
            shared_time = getattr(model, 'shared_time', False)
    else:
        # It's likely the transformer directly
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        if lattice_loss_weight is None:
            lattice_loss_weight = 1.0
        if shared_time is None:
            shared_time = False
        
    # Standard PackFlow checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss,
        'lattice_loss_weight': lattice_loss_weight,
        'shared_time': shared_time,
        'model_config': model_config,
        'training_config': training_config
    }
    
    # Ensure lattice params are also in training_config if provided
    if training_config is not None:
        if 'lattice_loss_weight' not in training_config:
            training_config['lattice_loss_weight'] = lattice_loss_weight
        if 'shared_time' not in training_config:
            training_config['shared_time'] = shared_time
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"💾 Saved standardized checkpoint: {save_path}")

def get_checkpoint_configs(checkpoint_path: str) -> tuple:
    """Helper to extract configs from an existing checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return ckpt.get('model_config', {}), ckpt.get('training_config', {})
