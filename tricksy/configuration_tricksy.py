import dataclasses
import torch
from transformers.models.opt.configuration_opt import OPTConfig

@dataclasses.dataclass(frozen=True)
class TricksyConfig:
    opt_config: OPTConfig

    # Percentage of weights to keep on each device
    # e.g. 30% of each MLP layer on GPU
    min_mlp_sparsity_gpu: float = .3
    # e.g. 100% of each MLP layer on CPU
    min_mlp_sparsity_cpu: float = 1

    # If true, cleans up layer's weights after computing forward pass
    full_offload: bool = False

    dtype: torch.dtype = torch.float16