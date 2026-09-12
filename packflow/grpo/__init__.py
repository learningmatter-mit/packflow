"""GRPO (Group Relative Policy Optimization) post-training for PackFlow.

Finetunes a pretrained PackFlow flow-matching model using UMA energy as the
reward signal. The "PA" (preference-aligned) model in the paper is the GRPO
finetune of ``packflow-ddp`` (experiment E025).

Key modules:
- ``trainer``          -- the GRPO training loop (run as ``python -m packflow.grpo.trainer``).
- ``sampler``          -- batched ODE sampling used to draw policy rollouts.
- ``model_loader``     -- load any PackFlow checkpoint into a flow-matching wrapper.
- ``checkpoint_utils`` -- save checkpoints in the standardized PackFlow format.

Reward (UMA energy) lives in ``packflow.relaxation.uma``.

Note: ``trainer``, ``sampler`` and the reward modules import torch / fairchem and
are intentionally NOT imported here, so ``import packflow.grpo`` stays lightweight.
"""

from .model_loader import (
    load_pretrained_model,
    load_model_by_type,
    get_default_checkpoint_path,
    get_model_config,
)
from .checkpoint_utils import save_packflow_checkpoint

__all__ = [
    "load_pretrained_model",
    "load_model_by_type",
    "get_default_checkpoint_path",
    "get_model_config",
    "save_packflow_checkpoint",
]
