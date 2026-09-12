"""PackFlow base-model training (flow matching).

Compact, visualization-free re-implementation of the original training script.
The optimizer, validation/best-model selection and checkpoint format are
preserved exactly, so runs reproduce the released base checkpoints.

    from packflow.training import TrainConfig, train

    cfg = TrainConfig(processed_data_dir="data/processed", d_model=640,
                      nhead=10, dim_feedforward=2560, num_transformer_layers=12,
                      coordinate_system="cartesian", use_attention_bias_from_graph=True,
                      use_rdkit_features=True)
    model = train(cfg)
"""

from .config import TrainConfig
from .trainer import train, build_flow_matching

__all__ = ["TrainConfig", "train", "build_flow_matching"]
