import dataclasses
import torch
from transformers.models.opt.configuration_opt import OPTConfig

@dataclasses.dataclass(frozen=True)
class TricksyConfig:
    opt_config: OPTConfig

    min_embedding_sparsity: float = 0.05
    min_embedding_probability: float = 0.05

    # Percentage of weights to keep in each layer
    min_mlp_sparsity_gpu: float = .2
    min_mlp_sparsity_cpu: float = .65
    min_mlp_probability: float = 1
    # Testing
    adjacent_mlp_sparsity: float = 0

    min_head_sparsity_gpu: float = .5
    min_head_sparsity_cpu: float = 1
    min_head_probability: float = 1
    # Testing
    adjacent_head_sparsity: float = 0

    full_offload: bool = True
    dtype: torch.dtype = torch.float16