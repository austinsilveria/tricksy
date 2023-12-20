import dataclasses
import torch
from transformers.models.opt.configuration_opt import OPTConfig

@dataclasses.dataclass(frozen=True)
class TricksyConfig:
    opt_config: OPTConfig

    min_embedding_sparsity: float = 0.05
    min_embedding_probability: float = 0.05

    min_mlp_sparsity: float = .2
    min_mlp_probability: float = 1
    # Testing
    adjacent_mlp_sparsity: float = 0

    min_head_sparsity: float = .5
    min_head_probability: float = 1
    # Testing
    adjacent_head_sparsity: float = 0

    dtype: torch.dtype = torch.float16