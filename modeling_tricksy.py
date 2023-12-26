from typing import Any, Optional, Callable, List, Tuple
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from accelerate import init_empty_weights
from transformers.activations import ACT2FN
from transformers.generation import GenerationConfig
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoder,
    OPTDecoderLayer,
    OPTForCausalLM,
    OPTModel,
)
from transformers.models.opt.configuration_opt import OPTConfig
from huggingface_hub import snapshot_download

from configuration_tricksy import TricksyConfig
from util import batch_copy, compute_index_diffs, load_mlp_sparsity_predictor, mmap_to_tensor, topk_and_threshold

TRICKSY_WEIGHTS_PATH = 'tricksy-weights/'

class SparseMLPCache:
    def __init__(
        self,
        indexed_fc1_weight: Optional[torch.Tensor] = None,
        indexed_fc1_bias: Optional[torch.Tensor] = None,
        indexed_fc2_weight: Optional[torch.Tensor] = None,
        gpu_cached_mlp_indices: Optional[torch.Tensor] = None,
    ):
        # [ffn_embed_dim * min_mlp_sparsity, hidden_size]
        self.indexed_fc1_weight = indexed_fc1_weight
        # [ffn_embed_dim * min_mlp_sparsity]
        self.indexed_fc1_bias = indexed_fc1_bias
        # [ffn_embed_dim * min_mlp_sparsity, hidden_size] (stored in transpose for efficient indexing)
        self.indexed_fc2_weight = indexed_fc2_weight

        # Indices that are already on GPU (this tensor is stored on the CPU)
        # [ffn_embed_dim * min_mlp_sparsity]
        self.gpu_cached_mlp_indices = gpu_cached_mlp_indices

class SparseIndices:
    def __init__(self, tricksy_config: TricksyConfig, opt_config: OPTConfig):
        self.mlp_indices_buffer_gpu = torch.empty(
            (int(opt_config.ffn_dim * tricksy_config.min_mlp_sparsity_gpu),),
            dtype=torch.int32,
            device='cuda'
        )
        self.mlp_indices_buffer_cpu = torch.empty(
            (int(opt_config.ffn_dim * tricksy_config.min_mlp_sparsity_gpu),),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )

        # Default stream blocks until indices are copied to CPU
        self.index_copy_stream = torch.cuda.default_stream()
    
    def copy_mlp_indices_to_cpu(self):
        self.mlp_indices_buffer_cpu = batch_copy([self.mlp_indices_buffer_gpu], self.index_copy_stream, device='cpu')[0]

class OPTDiskWeights:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_suffix = model_name.split('/')[-1]
        self.config = OPTConfig.from_pretrained(model_name)

        try:
            print(f'downloading from austinsilveria/tricksy-{self.model_suffix}')
            self.weight_path = snapshot_download(repo_id=f'austinsilveria/tricksy-{self.model_suffix}') + '/'
        except:
            print(f'failed to download from austinsilveria/tricksy-{self.model_suffix}')
            self.weight_path = f'{TRICKSY_WEIGHTS_PATH}{self.model_suffix}/'

        with init_empty_weights():
            model = OPTModel(self.config)
        self.state_dict = model.state_dict()

        if not os.path.exists(f'{self.weight_path}decoder.embed_tokens.weight'):
            # Download original weights and write memmap files
            print(f'downloading and preprocessing original weights')
            self.cache_weights()
        
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f'decoder.layers.{i}.'
            self.delete_weights([
                f'{layer_prefix}self_attn.q_proj.weight',
                f'{layer_prefix}self_attn.k_proj.weight',
                f'{layer_prefix}self_attn.v_proj.weight',
                f'{layer_prefix}self_attn.out_proj.weight',
                f'{layer_prefix}self_attn.q_proj.bias',
                f'{layer_prefix}self_attn.k_proj.bias',
                f'{layer_prefix}self_attn.v_proj.bias'
            ])
            self.add_weights([
                (f'{layer_prefix}fc2.weight', (self.config.ffn_dim, self.config.hidden_size)),
                (f'{layer_prefix}self_attn.catted_head_weights', (self.config.num_attention_heads, head_dim * 4, self.config.hidden_size)),
                (f'{layer_prefix}self_attn.catted_head_biases', (self.config.num_attention_heads, 3, head_dim)),
            ])

        self.memmap_weights = { key: self.load_memmap_weight(key) for key in self.state_dict.keys() }

    def load_memmap_weight(self, key: str):
        return torch.from_numpy(np.memmap(f'{self.weight_path}{key}', dtype='float16', mode='r', shape=(self.state_dict[key].shape)))

    def add_weights(self, weights: List[Tuple[str, torch.Size]]):
        for key, shape in weights:
            self.state_dict[key] = torch.empty(shape, dtype=torch.float16, device='meta')

    def delete_weights(self, keys: List[str]):
        for key in keys:
            if key in self.state_dict:
                del self.state_dict[key]
            path = f'{self.weight_path}{key}'
            if os.path.exists(path):
                os.remove(path)

    def cache_weights(self):
        os.makedirs(self.weight_path, exist_ok=True)
        weights_location = snapshot_download(repo_id=self.model_name, ignore_patterns=['flax*', 'tf*'])
        shards = [file for file in os.listdir(weights_location) if file.startswith("pytorch_model") and file.endswith(".bin")]
        for shard in shards:
            print(f'caching {shard}')
            shard_path = os.path.join(weights_location, shard)
            shard_state_dict = torch.load(shard_path)
            for key in shard_state_dict.keys():
                path = f'{self.weight_path}{key.replace("model.", "")}'
                memmap = np.memmap(path, dtype='float16', mode='w+', shape=(shard_state_dict[key].shape))
                memmap[:] = shard_state_dict[key].cpu().numpy()
        
        # Store weights in shape for efficient indexing
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f'decoder.layers.{i}.'
            # FC2 in transpose
            fc2t = torch.from_numpy(np.array(self.load_memmap_weight(f'{layer_prefix}fc2.weight')[:])).t().contiguous().clone()
            np.memmap(f'{self.weight_path}decoder.layers.{i}.fc2.weight', dtype='float16', mode='w+', shape=fc2t.shape)[:] = fc2t.numpy()

            # Attention weights by head
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            qw = mmap_to_tensor(self.load_memmap_weight(f'{layer_prefix}self_attn.q_proj.weight')[:])
            kw = mmap_to_tensor(self.load_memmap_weight(f'{layer_prefix}self_attn.k_proj.weight')[:])
            vw = mmap_to_tensor(self.load_memmap_weight(f'{layer_prefix}self_attn.v_proj.weight')[:])
            ow = mmap_to_tensor(self.load_memmap_weight(f'{layer_prefix}self_attn.out_proj.weight')[:])
            pre_cat_shape = (self.config.num_attention_heads, head_dim, self.config.hidden_size)
            # [head, head_dim * 4, hidden_size]
            catted_head_weights = torch.cat(
                [qw.view(pre_cat_shape).clone(), kw.view(pre_cat_shape).clone(), vw.view(pre_cat_shape).clone(), ow.T.view(pre_cat_shape).clone(),],
                dim=1,
            ).contiguous().clone()
            np.memmap(f'{self.weight_path}{layer_prefix}self_attn.catted_head_weights', dtype='float16', mode='w+', shape=catted_head_weights.shape)[:] =\
                catted_head_weights.numpy()

            # Attention biases by head
            qb = mmap_to_tensor(self.load_memmap_weight(f'{layer_prefix}self_attn.q_proj.bias')[:])
            kb = mmap_to_tensor(self.load_memmap_weight(f'{layer_prefix}self_attn.k_proj.bias')[:])
            vb = mmap_to_tensor(self.load_memmap_weight(f'{layer_prefix}self_attn.v_proj.bias')[:])
            pre_cat_shape = (self.config.num_attention_heads, 1, head_dim)
            # [head, 3, head_dim]
            catted_head_biases = torch.cat(
                # Don't index out bias since we need all dims after projecting back up to hidden size
                [qb.view(pre_cat_shape).clone(), kb.view(pre_cat_shape).clone(), vb.view(pre_cat_shape).clone()],
                dim=1,
            ).contiguous().clone()
            np.memmap(f'{self.weight_path}{layer_prefix}self_attn.catted_head_biases', dtype='float16', mode='w+', shape=catted_head_biases.shape)[:] =\
                catted_head_biases.numpy()

            self.delete_weights([
                f'{layer_prefix}self_attn.q_proj.weight',
                f'{layer_prefix}self_attn.k_proj.weight',
                f'{layer_prefix}self_attn.v_proj.weight',
                f'{layer_prefix}self_attn.out_proj.weight',
                f'{layer_prefix}self_attn.q_proj.bias',
                f'{layer_prefix}self_attn.k_proj.bias',
                f'{layer_prefix}self_attn.v_proj.bias'
            ])
            self.add_weights([
                (f'{layer_prefix}self_attn.catted_head_weights', catted_head_weights.shape),
                (f'{layer_prefix}self_attn.catted_head_biases', catted_head_biases.shape),
            ])

class TricksyContext:
    def __init__(self, tricksy_config: TricksyConfig, opt_config: OPTConfig):
        self.indices = SparseIndices(tricksy_config, opt_config)
        self.load_weight_stream = torch.cuda.Stream()
        self.layer = 0
        self.is_prompt_phase = True
        self.forward_times = []

class TricksyLayer:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def load_weights(self, tricksy_context: TricksyContext):
        pass

class TricksyLayerInputs:
    def __init__(
        self,
        disk_weights: OPTDiskWeights,
        layer_key_prefix: str = None,
        next_layer: TricksyLayer = None,
        sparsity_predictors: List[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.disk_weights = disk_weights
        # self.get_weight = lambda key: self.disk_weights.load_memmap_weight(f'{layer_key_prefix}{key}')
        self.get_weight = lambda key: self.disk_weights.memmap_weights[(f'{layer_key_prefix}{key}')]
        self.layer_key_prefix = layer_key_prefix
        self.next_layer = next_layer
        self.sparsity_predictors = sparsity_predictors

class TricksyOPTLearnedPositionalEmbedding(TricksyLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, tricksy_context):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        self.tricksy_context = tricksy_context
        self.weight = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()
        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        out = F.embedding(positions + self.offset, self.weight)
        return out

class TricksyOPTAttention(OPTAttention, TricksyLayer):
    def __init__(self, tricksy_config: TricksyConfig, inputs: TricksyLayerInputs, tricksy_context: TricksyContext, is_decoder: bool = False, **kwargs):
        nn.Module.__init__(self)
        self.tricksy_config = tricksy_config
        self.config = tricksy_config.opt_config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                print(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument("hidden_size", self.config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", self.config, "num_heads", kwargs)
        self.dropout = _handle_deprecated_argument("attention_dropout", self.config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", self.config, "bias", kwargs)

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # [Tricksy]
        self.tricksy_context = tricksy_context
        self.inputs = inputs
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.qw = self.kw = self.vw = self.ow = self.qb = self.kb = self.vb = self.out_proj_bias = self.layer_norm_weight = self.layer_norm_bias = None
        self.q_proj = lambda x: F.linear(x, self.qw, self.qb)
        self.k_proj = lambda x: F.linear(x, self.kw, self.kb)
        self.v_proj = lambda x: F.linear(x, self.vw, self.vb)
        self.out_proj = lambda x: F.linear(x, self.ow, self.out_proj_bias)
        self.layer_norm = lambda x: F.layer_norm(x, (self.config.hidden_size,), self.layer_norm_weight, self.layer_norm_bias)

    def load_weights(self, tricksy_context: TricksyContext):
        if self.tricksy_context.is_prompt_phase:
            # Full weights for prompt phase
            self.catted_weights, self.catted_biases, self.out_proj_bias, self.layer_norm_weight, self.layer_norm_bias = batch_copy(
                [
                    mmap_to_tensor(self.inputs.get_weight('self_attn.catted_head_weights')[:], pin_memory=True),
                    mmap_to_tensor(self.inputs.get_weight('self_attn.catted_head_biases')[:], pin_memory=True),
                    mmap_to_tensor(self.inputs.get_weight('self_attn.out_proj.bias')[:], pin_memory=True),
                    mmap_to_tensor(self.inputs.get_weight('self_attn_layer_norm.weight')[:], pin_memory=True),
                    mmap_to_tensor(self.inputs.get_weight('self_attn_layer_norm.bias')[:], pin_memory=True),
                ],
                tricksy_context.load_weight_stream,
            )
            torch.cuda.synchronize()
            # Weights stored in shape for efficient indexing to support offloading attention heads (not currently being done)
            self.qw = self.catted_weights[:, :self.head_dim, :].reshape(self.config.hidden_size, self.config.hidden_size).contiguous()
            self.kw = self.catted_weights[:, self.head_dim:self.head_dim * 2, :].reshape(self.config.hidden_size, self.config.hidden_size).contiguous()
            self.vw = self.catted_weights[:, self.head_dim * 2:self.head_dim * 3, :].reshape(self.config.hidden_size, self.config.hidden_size).contiguous()
            self.ow = self.catted_weights[:, self.head_dim * 3:, :].reshape(self.config.hidden_size, self.config.hidden_size).t().contiguous()
            self.catted_weights = None

            self.qb = self.catted_biases[:, 0, :].reshape(self.config.hidden_size).contiguous()
            self.kb = self.catted_biases[:, 1, :].reshape(self.config.hidden_size).contiguous()
            self.vb = self.catted_biases[:, 2, :].reshape(self.config.hidden_size).contiguous()
            self.catted_biases = None

    def forward(self, hidden_states, **kwargs):
        # Wait for attention weights to get to GPU
        torch.cuda.synchronize()

        # Predict MLP sparsity based on attention input
        self.tricksy_context.indices.mlp_indices_buffer_gpu = topk_and_threshold(
            self.inputs.sparsity_predictors[0](hidden_states)[0, -1, :],
            int(self.config.ffn_dim * self.tricksy_config.min_mlp_sparsity_gpu),
        )
        self.tricksy_context.indices.copy_mlp_indices_to_cpu()
        torch.cuda.synchronize()

        # Load MLP weights while computing attention
        self.inputs.next_layer.load_weights(self.tricksy_context)

        out = super().forward(self.layer_norm(hidden_states), **kwargs)

        # Wait for MLP weights to get to GPU
        torch.cuda.synchronize()

        return out

class TricksyOPTDecoderLayer(OPTDecoderLayer):
    def __init__(self, tricksy_config: TricksyConfig, inputs: TricksyLayerInputs, tricksy_context: TricksyContext):
        nn.Module.__init__(self)
        self.tricksy_config = tricksy_config
        self.config = tricksy_config.opt_config
        self.embed_dim = self.config.hidden_size

        self.tricksy_context = tricksy_context
        self.self_attn_layer_inputs = TricksyLayerInputs(
            disk_weights=inputs.disk_weights,
            layer_key_prefix=inputs.layer_key_prefix,
            # While computing attention, load MLP
            next_layer=self,
            sparsity_predictors=inputs.sparsity_predictors,
        )
        self.self_attn = TricksyOPTAttention(tricksy_config, self.self_attn_layer_inputs, tricksy_context, is_decoder=True)

        self.do_layer_norm_before = self.config.do_layer_norm_before
        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]

        self.inputs = inputs
        random_mlp_indices_gpu =\
            torch.randperm(self.config.ffn_dim, device='cpu', dtype=torch.int32)[:int(self.config.ffn_dim * self.tricksy_config.min_mlp_sparsity_gpu)]
        self.index_cache = SparseMLPCache(gpu_cached_mlp_indices=random_mlp_indices_gpu)

        # identity since we move this to attention layer
        # extreme tricksy
        self.self_attn_layer_norm = lambda x: x

        self.fc1_weight = self.fc2_weight = self.final_layer_norm_weight = self.fc1_bias = self.fc2_bias = self.final_layer_norm_bias = None
        self.ring_idx = 0
        self.fc1_weight_diff = self.fc2_weight_diff = self.fc1_bias_diff = None
        self.fc1 = lambda x: F.linear(x, torch.cat([self.fc1_weight, self.fc1_weight_diff]), torch.cat([self.fc1_bias, self.fc1_bias_diff]))
        self.fc2 = lambda x: F.linear(x, torch.cat([self.fc2_weight, self.fc2_weight_diff]).T, self.fc2_bias)
        self.final_layer_norm = lambda x: F.layer_norm(x, (self.embed_dim,), self.final_layer_norm_weight, self.final_layer_norm_bias)
    
    def load_weights(self, tricksy_context: TricksyContext):
        if self.fc1_weight is None:
            # Full weights for prompt phase
            fc1w = mmap_to_tensor(self.inputs.get_weight('fc1.weight')[:], pin_memory=True)
            fc1b = mmap_to_tensor(self.inputs.get_weight('fc1.bias')[:], pin_memory=True)
            fc2w = mmap_to_tensor(self.inputs.get_weight('fc2.weight')[:], pin_memory=True)
            fc2b = mmap_to_tensor(self.inputs.get_weight('fc2.bias')[:], pin_memory=True)
            lnw = mmap_to_tensor(self.inputs.get_weight('final_layer_norm.weight')[:], pin_memory=True)
            lnb = mmap_to_tensor(self.inputs.get_weight('final_layer_norm.bias')[:], pin_memory=True)

            self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias, self.final_layer_norm_weight, self.final_layer_norm_bias =\
                batch_copy([fc1w, fc1b, fc2w, fc2b, lnw, lnb], tricksy_context.load_weight_stream)
            self.fc1_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc1_bias_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc2_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')

            index_diffs = compute_index_diffs(tricksy_context.indices.mlp_indices_buffer_cpu, [self.index_cache.gpu_cached_mlp_indices])
            if len(index_diffs) > 0:
                gpu_index_diff = index_diffs[0]
                self.index_cache.gpu_cached_mlp_indices[gpu_index_diff.off_positions] = gpu_index_diff.off_elements

            self.index_cache.indexed_fc1_weight = fc1w.contiguous().pin_memory()
            self.index_cache.indexed_fc1_bias = fc1b.contiguous().pin_memory()
            self.index_cache.indexed_fc2_weight = fc2w.contiguous().pin_memory()
            return

        off_elements = torch.tensor(
            list(set(tricksy_context.indices.mlp_indices_buffer_cpu.tolist()).difference(set(self.index_cache.gpu_cached_mlp_indices.tolist()))),
            device='cpu',
            dtype=torch.int32,
            pin_memory=True
        )
        if off_elements.size(0) == 0:
            self.fc1_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc1_bias_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc2_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            return

        new_ring_idx = (self.ring_idx + off_elements.size(0)) % self.index_cache.gpu_cached_mlp_indices.size(0)
        if new_ring_idx > self.ring_idx:
            # single contiguous update
            self.index_cache.gpu_cached_mlp_indices[self.ring_idx:new_ring_idx] = off_elements
        elif off_elements.size(0) > 0:
            split = self.index_cache.gpu_cached_mlp_indices.size(0) - self.ring_idx
            # end of ring
            self.index_cache.gpu_cached_mlp_indices[self.ring_idx:] = off_elements[:split]
            # beginning of ring
            self.index_cache.gpu_cached_mlp_indices[:new_ring_idx] = off_elements[split:]

        # Allocate
        self.fc1_weight_diff = torch.empty((off_elements.size(0), self.config.hidden_size), dtype=self.tricksy_config.dtype, device='cuda')
        self.fc1_bias_diff = torch.empty((off_elements.size(0)), dtype=self.tricksy_config.dtype, device='cuda')
        self.fc2_weight_diff = torch.empty((off_elements.size(0), self.config.hidden_size), dtype=self.tricksy_config.dtype, device='cuda')
        # Index
        fc1wd = self.index_cache.indexed_fc1_weight[off_elements].pin_memory()
        fc1bd = self.index_cache.indexed_fc1_bias[off_elements].pin_memory()
        fc2wd = self.index_cache.indexed_fc2_weight[off_elements].pin_memory()
        # Copy
        self.fc1_weight_diff, self.fc1_bias_diff, self.fc2_weight_diff = batch_copy([fc1wd, fc1bd, fc2wd], tricksy_context.load_weight_stream)

    def forward(self, *args, **kwargs):
        # Wait for attention weights to get to GPU
        torch.cuda.synchronize()

        # Load next layer's attention weights
        self.inputs.next_layer.load_weights(self.tricksy_context)

        out = super().forward(*args, **kwargs)

        if self.tricksy_config.full_offload:
            self.fc1_weight = self.fc2_weight = self.final_layer_norm_weight = self.fc1_bias = self.fc2_bias = self.final_layer_norm_bias = None
        elif self.tricksy_context.is_prompt_phase:
            # Only keep sparse MLP weights on GPU after prompt phase
            self.fc1_weight = self.fc1_weight[self.index_cache.gpu_cached_mlp_indices.to('cuda')]
            self.fc1_bias = self.fc1_bias[self.index_cache.gpu_cached_mlp_indices.to('cuda')]
            self.fc2_weight = self.fc2_weight[self.index_cache.gpu_cached_mlp_indices.to('cuda')]

        # Update ring buffers
        if not self.tricksy_config.full_offload:
            prev_ring_idx = self.ring_idx
            self.ring_idx = (self.ring_idx + self.fc1_weight_diff.size(0)) % self.fc1_weight.size(0)
            if self.ring_idx > prev_ring_idx:
                # does not wrap around ring
                self.fc1_weight[prev_ring_idx:self.ring_idx] = self.fc1_weight_diff
                self.fc1_bias[prev_ring_idx:self.ring_idx] = self.fc1_bias_diff
                self.fc2_weight[prev_ring_idx:self.ring_idx] = self.fc2_weight_diff
            elif self.fc1_weight_diff.size(0) > 0:
                # wraps around ring
                split = self.fc1_weight_diff.size(0) - self.ring_idx
                self.fc1_weight[prev_ring_idx:] = self.fc1_weight_diff[:split]
                self.fc1_weight[:self.ring_idx] = self.fc1_weight_diff[split:]
                self.fc1_bias[prev_ring_idx:] = self.fc1_bias_diff[:split]
                self.fc1_bias[:self.ring_idx] = self.fc1_bias_diff[split:]
                self.fc2_weight[prev_ring_idx:] = self.fc2_weight_diff[:split]
                self.fc2_weight[:self.ring_idx] = self.fc2_weight_diff[split:]
        self.fc1_weight_diff = self.fc2_weight_diff = self.fc1_bias_diff = None

        self.tricksy_context.layer += 1
        return out

class TricksyOPTDecoder(OPTDecoder, TricksyLayer):
    def __init__(self, tricksy_config: TricksyConfig, disk_weights: OPTDiskWeights, tricksy_opt_for_causal_lm, tricksy_context: TricksyContext):
        nn.Module.__init__(self)
        self.config = tricksy_config.opt_config
        self.dropout = self.config.dropout
        self.layerdrop = self.config.layerdrop
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self._use_flash_attention_2 = False
        self.gradient_checkpointing = False
        self.project_out = None
        self.project_in = None

        self.embed_tokens_weight = None
        self.embed_positions = TricksyOPTLearnedPositionalEmbedding(tricksy_context)

        self.tricksy_context = tricksy_context
        self.layers: List[TricksyOPTDecoderLayer] = []
        for i in range(self.config.num_hidden_layers):
            pretrained_layer_num = self.config.num_hidden_layers - i - 1
            sparsity_predictors = [load_mlp_sparsity_predictor(disk_weights.weight_path, pretrained_layer_num, tricksy_config.dtype)]
            if sparsity_predictors[0] is None:
                sparsity_predictors[0] = lambda x: F.linear(x, torch.rand((self.config.ffn_dim, self.config.hidden_size), device='cuda', dtype=tricksy_config.dtype))
            self.layers.append(TricksyOPTDecoderLayer(
                tricksy_config,
                TricksyLayerInputs(
                    disk_weights=disk_weights,
                    layer_key_prefix=f'decoder.layers.{pretrained_layer_num}.',
                    # While computing MLP, load next attention
                    # While computing last MLP, load output embeddings (stored in TricksyOPTForCausalLM)
                    next_layer=self.layers[i - 1].self_attn if i > 0 else tricksy_opt_for_causal_lm,
                    sparsity_predictors=sparsity_predictors,
                ),
                tricksy_context,
            ))
        self.layers.reverse()

        self.final_layer_norm = lambda x: x
        self.inputs = TricksyLayerInputs(disk_weights=disk_weights, layer_key_prefix='decoder.')

    def embed_tokens(self, x):
        return F.embedding(x, self.embed_tokens_weight, self.padding_idx)
    
    def load_weights(self, tricksy_context: TricksyContext):
        if self.embed_tokens_weight is None:
            self.embed_tokens_weight, self.embed_positions.weight = batch_copy(
                [
                    mmap_to_tensor(self.inputs.get_weight('embed_tokens.weight')[:], pin_memory=True),
                    mmap_to_tensor(self.inputs.get_weight('embed_positions.weight')[:], pin_memory=True),
                ],
                tricksy_context.load_weight_stream,
            )

    def forward(self, *args, **kwargs):
        # Wait for input embedding weights to get to GPU
        torch.cuda.synchronize()

        # While computing input embeddings, load first attention
        self.layers[0].self_attn.load_weights(self.tricksy_context)

        out = super().forward(*args, **kwargs)

        # Wait for output embedding weights to get to GPU
        torch.cuda.synchronize()

        # No longer prompt phase after first full pass
        self.tricksy_context.is_prompt_phase = False
        # Load input embeddings while computing output
        self.load_weights(self.tricksy_context)

        return out

class TricksyOPTModel(OPTModel):
    def __init__(self, tricksy_config: TricksyConfig, disk_weights: OPTDiskWeights, tricksy_opt_for_causal_lm, tricksy_context: TricksyContext):
        nn.Module.__init__(self)
        self.config = tricksy_config.opt_config
        self.tricksy_context = tricksy_context
        self.decoder = TricksyOPTDecoder(tricksy_config, disk_weights, tricksy_opt_for_causal_lm, tricksy_context)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return out

# who's got the weights?
# [InputEmbedding,    Attention.0,           MLP.0,                    Attention.1,           MLP.1,                 ..., OutputEmbedding]
# [TricksyOPTDecoder, TricksyOPTAttention.0, TricksyOPTDecoderLayer.0, TricksyOPTAttention.1, TricksyDecoderLayer.1, ..., TricksyOPTForCausalLM]
#
# 1. Prompt pass: Before computing layer, send full dense weights to GPU. After computing layer, only keep sparse weights on GPU.
# 2. Generation passes: Before computing layer, compute and send sparse weight diff to GPU.
class TricksyOPTForCausalLM(OPTForCausalLM, TricksyLayer):
    def __init__(self, tricksy_config: TricksyConfig, disk_weights: OPTDiskWeights):
        nn.Module.__init__(self)
        self.config = disk_weights.config
        self.generation_config = GenerationConfig.from_model_config(self.config) if self.can_generate() else None

        self.tricksy_context = TricksyContext(tricksy_config, self.config)
        self.model = TricksyOPTModel(tricksy_config, disk_weights, self, self.tricksy_context)

        self.final_layer_norm_weight = self.lm_head_weight = self.final_layer_norm_bias = None
        # double stacking tricksy!
        self.final_layer_norm = lambda x: F.layer_norm(x, (self.config.hidden_size,), self.final_layer_norm_weight, self.final_layer_norm_bias)
        self.lm_head = lambda x: F.linear(self.final_layer_norm(x), self.lm_head_weight)

        self.inputs = TricksyLayerInputs(disk_weights=disk_weights, layer_key_prefix='decoder.', next_layer=self.model.decoder)
    
    def load_weights(self, tricksy_context: TricksyContext):
        if self.final_layer_norm_weight is None:
            self.final_layer_norm_weight, self.lm_head_weight, self.final_layer_norm_bias = batch_copy(
                [
                    mmap_to_tensor(self.inputs.get_weight('final_layer_norm.weight')[:], pin_memory=True),
                    mmap_to_tensor(self.inputs.get_weight('embed_tokens.weight')[:], pin_memory=True),
                    mmap_to_tensor(self.inputs.get_weight('final_layer_norm.bias')[:], pin_memory=True),
                ],
                tricksy_context.load_weight_stream,
            )
    
    def forward(self, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        out = super().forward(*args, **kwargs)
        torch.cuda.synchronize()
        self.tricksy_context.forward_times.append(time.time() - start)
        self.tricksy_context.layer = 0
        return out

    def generate(self, *args, **kwargs):
        # Load input embeddings for first token
        self.model.decoder.load_weights(self.tricksy_context)
        torch.cuda.synchronize()
        out = super().generate(*args, **kwargs)
        print(f'\n===\nDecoding tok/s: {1 / (sum(self.tricksy_context.forward_times[1:]) / (len(self.tricksy_context.forward_times) - 1))}\n===\n')
        return out