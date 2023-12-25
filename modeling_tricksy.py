from typing import Any, Optional, Callable, Dict, List, Tuple
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
from util import IndexDiff, batch_copy, compute_index_diffs, getenv, load_predictor, mmap_to_tensor, topk_and_threshold

import psutil

TRICKSY_WEIGHTS_PATH = 'tricksy-weights/'
# TRICKSY_WEIGHTS_PATH = '/content/drive/MyDrive/RTC/tricksy-weights/'
DEBUG = getenv("DEBUG")

def load_attn_head_sparsity_predictor(weight_path_prefix: str, layer_num: int, dtype: torch.dtype, device: str = 'cuda') -> Callable:
    path_prefix = f'{weight_path_prefix}decoder.layers.{layer_num}.attn-head-sparsity-predictor.'
    return load_predictor(path_prefix, dtype, device=device)

def load_mlp_sparsity_predictor(weight_path_prefix: str, layer_num: int, dtype: torch.dtype, device: str = 'cuda') -> Callable:
    path_prefix = f'{weight_path_prefix}decoder.layers.{layer_num}.attn.mlp-sparsity-predictor.'
    return load_predictor(path_prefix, dtype, device=device)

class SparseInputEmbeddingCache:
    def __init__(
        self,
        indexed_input_embedding: Optional[torch.Tensor] = None,
        cached_vocab_indices: Optional[torch.Tensor] = None,
    ):
        # [vocab_size * min_embedding_sparsity, hidden_size]
        self.indexed_input_embedding = indexed_input_embedding

        # [vocab_size * min_embedding_sparsity]
        self.cached_vocab_indices = cached_vocab_indices

class SparseAttentionCache:
    def __init__(
        self,
        indexed_weights: Optional[torch.Tensor] = None,
        indexed_biases: Optional[torch.Tensor] = None,
        cpu_cached_head_indices: Optional[torch.Tensor] = None,
        gpu_cached_head_indices: Optional[torch.Tensor] = None,
    ):
        # [n_head * min_head_sparsity, head_dim * 4, hidden_size]
        self.indexed_weights = indexed_weights
        # [n_head * min_head_sparsity, 3, head_dim]
        self.indexed_biases = indexed_biases

        # [n_head * min_head_sparsity]
        # Indices that are already on CPU (this tensor is stored on the CPU)
        self.cpu_cached_head_indices = cpu_cached_head_indices
        # Indices that are already on GPU (this tensor is stored on the CPU)
        self.gpu_cached_head_indices = gpu_cached_head_indices

class SparseMLPCache:
    def __init__(
        self,
        indexed_fc1_weight: Optional[torch.Tensor] = None,
        indexed_fc1_bias: Optional[torch.Tensor] = None,
        indexed_fc2_weight: Optional[torch.Tensor] = None,
        cpu_cached_mlp_indices: Optional[torch.Tensor] = None,
        gpu_cached_mlp_indices: Optional[torch.Tensor] = None,
    ):
        # [ffn_embed_dim * min_mlp_sparsity, hidden_size]
        self.indexed_fc1_weight = indexed_fc1_weight
        # [ffn_embed_dim * min_mlp_sparsity]
        self.indexed_fc1_bias = indexed_fc1_bias
        # [ffn_embed_dim * min_mlp_sparsity, hidden_size] (stored in transpose for efficient indexing)
        self.indexed_fc2_weight = indexed_fc2_weight

        # [ffn_embed_dim * min_mlp_sparsity]
        # Indices that are already on CPU (this tensor is stored on the CPU)
        self.cpu_cached_mlp_indices = cpu_cached_mlp_indices
        # Indices that are already on GPU (this tensor is stored on the CPU)
        # [ffn_embed_dim * min_mlp_sparsity]
        self.gpu_cached_mlp_indices = gpu_cached_mlp_indices

class SparseOutputEmbeddingCache:
    def __init__(
        self,
        indexed_output_embedding: Optional[torch.Tensor] = None,
        cached_vocab_indices: Optional[torch.Tensor] = None,
    ):
        # [vocab_size * min_embedding_sparsity, hidden_size]
        self.indexed_output_embedding = indexed_output_embedding

        # [vocab_size * min_embedding_sparsity]
        self.cached_vocab_indices = cached_vocab_indices

class SparseIndices:
    def __init__(self, tricksy_config: TricksyConfig, opt_config: OPTConfig):
        self.input_embedding_indices_buffer_gpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cuda'
        )
        self.head_indices_buffer_gpu = torch.randperm(
            opt_config.num_attention_heads,
            dtype=torch.int8,
            device='cuda'
        )[:int(opt_config.num_attention_heads * tricksy_config.min_head_sparsity_gpu)]
        self.mlp_indices_buffer_gpu = torch.empty(
            (int(opt_config.ffn_dim * tricksy_config.min_mlp_sparsity_gpu),),
            dtype=torch.int32,
            device='cuda'
        )
        self.output_embedding_indices_buffer_gpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cuda'
        )

        self.input_embedding_indices_buffer_cpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cpu',
            pin_memory=True,
        )
        self.head_indices_buffer_cpu = torch.randperm(
            opt_config.num_attention_heads,
            dtype=torch.int8,
            device='cpu',
            pin_memory=True,
        )[:int(opt_config.num_attention_heads * tricksy_config.min_head_sparsity_gpu)]
        self.mlp_indices_buffer_cpu = torch.empty(
            (int(opt_config.ffn_dim * tricksy_config.min_mlp_sparsity_gpu),),
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.output_embedding_indices_buffer_cpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cpu',
            pin_memory=True,
        )

        # Default stream blocks until indices are copied to CPU
        self.index_copy_stream = torch.cuda.default_stream()
    
    def copy_input_embedding_indices_to_cpu(self):
        self.input_embedding_indices_buffer_cpu = batch_copy([self.input_embedding_indices_buffer_gpu], self.index_copy_stream, device='cpu')[0]

    def copy_head_indices_to_cpu(self):
        self.head_indices_buffer_cpu = batch_copy([self.head_indices_buffer_gpu], self.index_copy_stream, device='cpu')[0]

    def copy_mlp_indices_to_cpu(self):
        self.mlp_indices_buffer_cpu = batch_copy([self.mlp_indices_buffer_gpu], self.index_copy_stream, device='cpu')[0]

    def copy_output_embedding_indices_to_cpu(self):
        self.output_embedding_indices_buffer_cpu = batch_copy([self.output_embedding_indices_buffer_gpu], self.index_copy_stream, device='cpu')[0]

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
        print(f'state_dict keys: {self.state_dict.keys()}')

        print(f'checking path: {self.weight_path}decoder.embed_tokens.weight')
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

        print(f'loading memmap weights for {self.model_name}')
        self.memmap_weights = { key: self.load_memmap_weight(key) for key in self.state_dict.keys() }

    def load_memmap_weight(self, key: str):
        return torch.from_numpy(np.memmap(f'{self.weight_path}{key}', dtype='float16', mode='r', shape=(self.state_dict[key].shape)))

    def add_weights(self, weights: List[Tuple[str, torch.Size]]):
        for key, shape in weights:
            self.state_dict[key] = torch.empty(shape, dtype=torch.float16, device='meta')
            print(f'adding weight key: {key}')

    def delete_weights(self, keys: List[str]):
        for key in keys:
            if key in self.state_dict:
                del self.state_dict[key]
            path = f'{self.weight_path}{key}'
            if os.path.exists(path):
                print(f'removing: {path}')
                os.remove(path)

    def cache_weights(self):
        os.makedirs(self.weight_path, exist_ok=True)
        weights_location = snapshot_download(repo_id=self.model_name, ignore_patterns=['flax*', 'tf*'])
        shards = [file for file in os.listdir(weights_location) if file.startswith("pytorch_model") and file.endswith(".bin")]
        for shard in shards:
            print(f'caching {shard}')
            shard_path = os.path.join(weights_location, shard)
            print(f'shard_path: {shard_path}')
            shard_state_dict = torch.load(shard_path)
            for key in shard_state_dict.keys():
                path = f'{self.weight_path}{key.replace("model.", "")}'
                print(f'caching {path}')
                memmap = np.memmap(path, dtype='float16', mode='w+', shape=(shard_state_dict[key].shape))
                memmap[:] = shard_state_dict[key].cpu().numpy()
        
        # Store weights in shape for efficient indexing
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f'decoder.layers.{i}.'
            print(f'reshaping weights for layer {i}')
            # FC2 in transpose
            fc2t = torch.from_numpy(np.array(self.load_memmap_weight(f'{layer_prefix}fc2.weight')[:])).t().contiguous().clone()
            print(f'fc2t shape: {fc2t.shape}')
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

        cpu_random_head_indices =\
            torch.randperm(self.config.num_attention_heads, device='cpu', dtype=torch.int32)[:int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity_cpu)]
        gpu_random_head_indices =\
            torch.randperm(self.config.num_attention_heads, device='cpu', dtype=torch.int32)[:int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity_gpu)]
        self.index_cache = SparseAttentionCache(cpu_cached_head_indices=cpu_random_head_indices, gpu_cached_head_indices=gpu_random_head_indices)

        self.catted_weights = self.catted_biases = self.out_proj_bias = self.layer_norm_weight = self.layer_norm_bias = None
        self.qw = self.kw = self.vw = self.ow = self.qb = self.kb = self.vb = None
        self.q_proj = lambda x: F.linear(
            x,
            # self.catted_weights[:, :self.head_dim, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size),
            # self.catted_biases[:, 0, :].reshape(self.head_dim * self.catted_biases.size(0))
            self.qw,
            self.qb
        )
        self.k_proj = lambda x: F.linear(
            x,
            # self.catted_weights[:, self.head_dim:self.head_dim * 2, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size),
            # self.catted_biases[:, 1, :].reshape(self.head_dim * self.catted_biases.size(0))
            self.kw,
            self.kb
        )
        self.v_proj = lambda x: F.linear(
            x,
            # self.catted_weights[:, self.head_dim * 2:self.head_dim * 3, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size),
            # self.catted_biases[:, 2, :].reshape(self.head_dim * self.catted_biases.size(0))
            self.vw,
            self.vb
        )
        self.out_proj = lambda x: F.linear(
            x,
            # self.catted_weights[:, self.head_dim * 3:, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size).T,
            self.ow,
            self.out_proj_bias,
        )
        self.layer_norm = lambda x: F.layer_norm(x, (self.config.hidden_size,), self.layer_norm_weight, self.layer_norm_bias)

        self.last_pk = None

    def load_weights(self, tricksy_context: TricksyContext):
        if self.tricksy_context.is_prompt_phase:
            torch.cuda.synchronize()
            start = time.time()
            # Full weights for prompt phase
            catted_head_weights = mmap_to_tensor(self.inputs.get_weight('self_attn.catted_head_weights')[:], pin_memory=True)
            catted_head_biases = mmap_to_tensor(self.inputs.get_weight('self_attn.catted_head_biases')[:], pin_memory=True)
            out_proj_bias = mmap_to_tensor(self.inputs.get_weight('self_attn.out_proj.bias')[:], pin_memory=True)
            layer_norm_weight = mmap_to_tensor(self.inputs.get_weight('self_attn_layer_norm.weight')[:], pin_memory=True)
            layer_norm_bias = mmap_to_tensor(self.inputs.get_weight('self_attn_layer_norm.bias')[:], pin_memory=True)
            self.catted_weights, self.catted_biases, self.out_proj_bias, self.layer_norm_weight, self.layer_norm_bias = batch_copy(
                [
                    catted_head_weights,
                    catted_head_biases,
                    out_proj_bias,
                    layer_norm_weight,
                    layer_norm_bias,
                ],
                tricksy_context.load_weight_stream,
            )
            self.qw = self.catted_weights[:, :self.head_dim, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size).contiguous()
            self.kw = self.catted_weights[:, self.head_dim:self.head_dim * 2, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size).contiguous()
            self.vw = self.catted_weights[:, self.head_dim * 2:self.head_dim * 3, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size).contiguous()
            self.ow = self.catted_weights[:, self.head_dim * 3:, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size).t().contiguous()
            self.catted_weights = None

            self.qb = self.catted_biases[:, 0, :].reshape(self.head_dim * self.catted_biases.size(0)).contiguous()
            self.kb = self.catted_biases[:, 1, :].reshape(self.head_dim * self.catted_biases.size(0)).contiguous()
            self.vb = self.catted_biases[:, 2, :].reshape(self.head_dim * self.catted_biases.size(0)).contiguous()
            self.catted_biases = None

            if self.tricksy_config.min_head_sparsity_gpu == 1:
                return

            index_diffs = compute_index_diffs(
                tricksy_context.indices.head_indices_buffer_cpu,
                [self.index_cache.gpu_cached_head_indices, self.index_cache.cpu_cached_head_indices]
            )
            if len(index_diffs) > 0:
                gpu_index_diff = index_diffs[0]
                self.index_cache.gpu_cached_head_indices[gpu_index_diff.off_positions] = gpu_index_diff.off_elements
            if len(index_diffs) > 1:
                # CPU may not have all new indices
                cpu_index_diff = index_diffs[1]
                self.index_cache.cpu_cached_head_indices[cpu_index_diff.off_positions] = cpu_index_diff.off_elements

            self.index_cache.indexed_weights = catted_head_weights[self.index_cache.cpu_cached_head_indices].contiguous().clone().pin_memory()
            self.index_cache.indexed_biases = catted_head_biases[self.index_cache.cpu_cached_head_indices].contiguous().clone().pin_memory()
            return

        if self.tricksy_config.min_head_sparsity_gpu == 1:
            return
        self.num_heads = self.index_cache.gpu_cached_head_indices.size(0)
        # Forward pass uses this to reshape attention output before output projection
        self.embed_dim = self.num_heads * self.head_dim

        torch.cuda.synchronize()
        beginning = time.time()
        start = time.time()

        index_diffs = compute_index_diffs(
            tricksy_context.indices.head_indices_buffer_cpu,
            [self.index_cache.gpu_cached_head_indices, self.index_cache.cpu_cached_head_indices]
        )
        gpu_index_diff: IndexDiff = index_diffs[0]

        torch.cuda.synchronize()
        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'Computed off head positions and elements in {time.time() - start} seconds')
        torch.cuda.synchronize()
        start = time.time()
        off_positions_gpu = gpu_index_diff.off_positions.to('cuda')
        torch.cuda.synchronize()
        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'sent off positions to gpu in {time.time() - start} seconds')

        if gpu_index_diff.off_elements.size(0) == 0:
            if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
                print(f'No new head indices, skipping load_weights')
            return

        cpu_index_diff: IndexDiff = index_diffs[1]

        torch.cuda.synchronize()
        start = time.time()
        self.index_cache.gpu_cached_head_indices[gpu_index_diff.off_positions] = gpu_index_diff.off_elements
        self.index_cache.cpu_cached_head_indices[cpu_index_diff.off_positions] = cpu_index_diff.off_elements
        torch.cuda.synchronize()
        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'Updated cached head indices in {time.time() - start} seconds')

        # Index and overwrite the diff (much faster than full reindexing)
        #   e.g. adjacent tokens in 1.3b layer 5 have ~90% overlap of sparse indices
        #        adjacent tokens in 1.3b layer 19 have ~60% overlap of sparse indices
        torch.cuda.synchronize()
        start = time.time()

        if cpu_index_diff.off_elements.size(0) > 0:
            if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
                print(f'cpu_index_diff.off_elements.size(0): {cpu_index_diff.off_elements.size(0)}')
            # CPU grabs missing indices from disk
            self.index_cache.indexed_weights[cpu_index_diff.off_positions].copy_(self.inputs.get_weight('self_attn.catted_head_weights')[cpu_index_diff.off_elements])
            self.index_cache.indexed_biases[cpu_index_diff.off_positions].copy_(self.inputs.get_weight('self_attn.catted_head_biases')[cpu_index_diff.off_elements])

        torch.cuda.synchronize()
        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'Indexed heads in {time.time() - start} seconds')

            print(f'indexed_weights is pinned: {self.index_cache.indexed_weights.is_pinned()}')
            print(f'indexed_biases is pinned: {self.index_cache.indexed_biases.is_pinned()}')
        torch.cuda.synchronize()
        start = time.time()
        # Postions where CPU already had indices GPU needs, and positions where CPU just got indices GPU needs from disk
        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'cpu on positions shape: {cpu_index_diff.on_positions.shape}')
            print(f'cpu off positions shape: {cpu_index_diff.off_positions.shape}')
        cpu_diff_positions = torch.cat([cpu_index_diff.on_positions, cpu_index_diff.off_positions], dim=0)

        catted_weights_diff = torch.empty((cpu_diff_positions.size(0), self.catted_weights.size(1), self.catted_weights.size(2)), dtype=torch.float16, device='cpu', pin_memory=True)
        catted_biases_diff = torch.empty((cpu_diff_positions.size(0), self.catted_biases.size(1), self.catted_biases.size(2)), dtype=torch.float16, device='cpu', pin_memory=True)
        catted_weights_diff.copy_(self.index_cache.indexed_weights[cpu_diff_positions])
        catted_biases_diff.copy_(self.index_cache.indexed_biases[cpu_diff_positions])
        torch.cuda.synchronize()
        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'Indexed CPU attention weights in {time.time() - start} seconds')

        torch.cuda.synchronize()
        start = time.time()
        self.catted_weights[off_positions_gpu].copy_(catted_weights_diff, non_blocking=True)
        self.catted_biases[off_positions_gpu].copy_(catted_biases_diff, non_blocking=True)
        torch.cuda.synchronize()
        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'Copied sparse attention weights in {time.time() - start} seconds')

        if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            print(f'Finished loading weights in {time.time() - beginning} seconds')
        
    def forward(self, hidden_states, **kwargs):
        if len(self.inputs.sparsity_predictors) > 2:
        #     torch.cuda.synchronize()
        #     start = time.time()
        #     # Predict head sparsity based on input embedding output
        #     self.tricksy_context.indices.head_indices_buffer_gpu = topk_and_threshold(
        #         self.inputs.sparsity_predictors[2](hidden_states)[0, -1, :],
        #         int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity_gpu),
        #     )
        #     # print(f'head indices gpu: {self.tricksy_context.indices.head_indices_buffer_gpu}')
        #     self.tricksy_context.indices.copy_head_indices_to_cpu()
            torch.cuda.synchronize()
        #     # print(f'Computed head indices based on input embeddings in {time.time() - start} seconds')
            self.load_weights(self.tricksy_context)

        # # Wait for weights to get to GPU
        torch.cuda.synchronize()

        # start = time.time()
        # Predict MLP sparsity based on attention input
        mlp_sparsity = int(self.config.ffn_dim * self.tricksy_config.min_mlp_sparsity_gpu)
        self.tricksy_context.indices.mlp_indices_buffer_gpu = topk_and_threshold(
            self.inputs.sparsity_predictors[0](hidden_states)[0, -1, :],
            mlp_sparsity,
        )
        self.tricksy_context.indices.copy_mlp_indices_to_cpu()
        torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Computed MLP indices in {time.time() - start} seconds')

        self.inputs.next_layer.load_weights(self.tricksy_context)

        # torch.cuda.synchronize()
        # start = time.time()
        out = super().forward(self.layer_norm(hidden_states), **kwargs)
        # Wait for MLP weights to get to GPU
        torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Computed attention forward in {time.time() - start} seconds')

        # if self.tricksy_context.is_prompt_phase:
            # Only keep sparse weights on GPU after prompt phase
            # self.catted_weights = self.catted_weights[self.index_cache.gpu_cached_head_indices.to('cuda')]
            # self.catted_biases = self.catted_biases[self.index_cache.gpu_cached_head_indices.to('cuda')]

        # torch.cuda.synchronize()
        # start = time.time()
        # if self.inputs.sparsity_predictors[1] is not None:
        #     # Predict head sparsity based on MLP input
        #     # print(f'out shape: {out[0].shape}')
        #     self.tricksy_context.indices.head_indices_buffer_gpu = topk_and_threshold(
        #         self.inputs.sparsity_predictors[1](out[0])[0, -1, :],
        #         int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity_gpu),
        #     )
        #     self.tricksy_context.indices.copy_head_indices_to_cpu()
        #     torch.cuda.synchronize()
        #     # print(f'Computed head indices in {time.time() - start} seconds')

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
        random_mlp_indices_cpu =\
            torch.randperm(self.config.ffn_dim, device='cpu', dtype=torch.int32)[:int(self.config.ffn_dim * self.tricksy_config.min_mlp_sparsity_cpu)]
        random_mlp_indices_gpu =\
            torch.randperm(self.config.ffn_dim, device='cpu', dtype=torch.int32)[:int(self.config.ffn_dim * self.tricksy_config.min_mlp_sparsity_gpu)]
        self.index_cache = SparseMLPCache(cpu_cached_mlp_indices=random_mlp_indices_cpu, gpu_cached_mlp_indices=random_mlp_indices_gpu)

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
            # torch.cuda.synchronize()
            # start = time.time()
            # Full weights for prompt phase
            fc1w = mmap_to_tensor(self.inputs.get_weight('fc1.weight')[:], pin_memory=True)
            fc1b = mmap_to_tensor(self.inputs.get_weight('fc1.bias')[:], pin_memory=True)
            fc2w = mmap_to_tensor(self.inputs.get_weight('fc2.weight')[:], pin_memory=True)
            fc2b = mmap_to_tensor(self.inputs.get_weight('fc2.bias')[:], pin_memory=True)
            lnw = mmap_to_tensor(self.inputs.get_weight('final_layer_norm.weight')[:], pin_memory=True)
            lnb = mmap_to_tensor(self.inputs.get_weight('final_layer_norm.bias')[:], pin_memory=True)
            # torch.cuda.synchronize()
            # torch.cuda.synchronize()
            # start = time.time()
            self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias, self.final_layer_norm_weight, self.final_layer_norm_bias =\
                batch_copy([fc1w, fc1b, fc2w, fc2b, lnw, lnb], tricksy_context.load_weight_stream)
            self.fc1_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc1_bias_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc2_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            # torch.cuda.synchronize()

            index_diffs = compute_index_diffs(
                tricksy_context.indices.mlp_indices_buffer_cpu,
                [self.index_cache.gpu_cached_mlp_indices, self.index_cache.cpu_cached_mlp_indices]
            )
            if len(index_diffs) > 0:
                gpu_index_diff = index_diffs[0]
                self.index_cache.gpu_cached_mlp_indices[gpu_index_diff.off_positions] = gpu_index_diff.off_elements
            if len(index_diffs) > 1:
                # CPU may not have all new indices
                # cpu_index_diff = index_diffs[1]
                # self.index_cache.cpu_cached_mlp_indices[cpu_index_diff.off_positions] = cpu_index_diff.off_elements
                self.index_cache.cpu_cached_mlp_indices = torch.arange(self.config.ffn_dim, device='cpu', dtype=torch.int32)

            # self.index_cache.indexed_fc1_weight = fc1w[self.index_cache.cpu_cached_mlp_indices].contiguous().clone().pin_memory()
            # self.index_cache.indexed_fc1_bias = fc1b[self.index_cache.cpu_cached_mlp_indices].contiguous().clone().pin_memory()
            # self.index_cache.indexed_fc2_weight = fc2w[self.index_cache.cpu_cached_mlp_indices].contiguous().clone().pin_memory()

            self.index_cache.indexed_fc1_weight = fc1w.contiguous().pin_memory()
            self.index_cache.indexed_fc1_bias = fc1b.contiguous().pin_memory()
            self.index_cache.indexed_fc2_weight = fc2w.contiguous().pin_memory()
            return

        # torch.cuda.synchronize()
        # beginning = time.time()
        # start = time.time()
        # index_diffs = compute_index_diffs(
        #     tricksy_context.indices.mlp_indices_buffer_cpu,
        #     [self.index_cache.gpu_cached_mlp_indices, self.index_cache.cpu_cached_mlp_indices]
        # )
        gpu_index_diff: IndexDiff = IndexDiff(
            off_elements=torch.tensor(
                list(set(tricksy_context.indices.mlp_indices_buffer_cpu.tolist()).difference(set(self.index_cache.gpu_cached_mlp_indices.tolist()))),
                device='cpu',
                dtype=torch.int32,
                pin_memory=True
            )
        )
        # print(f'gpu_index_diff.off_elements: {gpu_index_diff.off_elements}')
        # print(f'GPU indices before overwriting diff: {self.index_cache.gpu_cached_mlp_indices}')
        # print(f'new mlp indices: {tricksy_context.indices.mlp_indices_buffer_cpu}')
        if gpu_index_diff.off_elements.size(0) == 0:
        # if len(index_diffs) == 0 or index_diffs[0].off_elements.size(0) == 0:
            self.fc1_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc1_bias_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            self.fc2_weight_diff = torch.tensor([], dtype=self.tricksy_config.dtype, device='cuda')
            # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            #     print(f'No new MLP indices, skipping load_weights')
            return

        # gpu_index_diff: IndexDiff = index_diffs[0]
        # cpu_index_diff: IndexDiff = index_diffs[1]
        # cpu_diff_positions = torch.cat([cpu_index_diff.on_positions, cpu_index_diff.off_positions], dim=0)
        cpu_diff_positions = gpu_index_diff.off_elements

        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Computed MLP off positions and elements in {time.time() - start} seconds')
            # print(f'GPU indices before overwriting diff: {self.index_cache.gpu_cached_mlp_indices}')
            # print(f'new mlp indices: {tricksy_context.indices.mlp_indices_buffer_cpu}')
            # print(f'gpu index diff: {gpu_index_diff}')
        # print(f'new mlp indices shape: {tricksy_context.indices.mlp_indices_buffer_cpu.shape}')
        # print(f'off_elements.shape: {gpu_index_diff.off_elements.shape}')
        # print(f'off_positions.shape: {gpu_index_diff.off_positions.shape}')
        # off_elements = torch.randperm(self.config.ffn_dim, pin_memory=False)[:int(self.index_cache.cached_ffn_indices.size(0) * self.tricksy_config.adjacent_mlp_sparsity)]
        # off_positions = torch.randperm(self.index_cache.cached_ffn_indices.size(0), pin_memory=False)[:int(self.index_cache.cached_ffn_indices.size(0) * self.tricksy_config.adjacent_mlp_sparsity)]
        # torch.cuda.synchronize()
        # start = time.time()
        # off_positions_gpu = gpu_index_diff.off_positions.to('cuda')
        # torch.cuda.synchronize()
        # print(f'sent MLP off positions to gpu in {time.time() - start} seconds')

        # torch.cuda.synchronize()
        # start = time.time()
        # self.index_cache.cpu_cached_mlp_indices[cpu_index_diff.off_positions] = cpu_index_diff.off_elements
        # self.index_cache.gpu_cached_mlp_indices[gpu_index_diff.off_positions] = self.index_cache.cpu_cached_mlp_indices[cpu_diff_positions]
        new_ring_idx = (self.ring_idx + cpu_diff_positions.size(0)) % self.index_cache.gpu_cached_mlp_indices.size(0)
        if new_ring_idx > self.ring_idx:
            # single contiguous update
            self.index_cache.gpu_cached_mlp_indices[self.ring_idx:new_ring_idx] = self.index_cache.cpu_cached_mlp_indices[cpu_diff_positions]
        elif cpu_diff_positions.size(0) > 0:
            split = self.index_cache.gpu_cached_mlp_indices.size(0) - self.ring_idx
            # end of ring
            self.index_cache.gpu_cached_mlp_indices[self.ring_idx:] = self.index_cache.cpu_cached_mlp_indices[cpu_diff_positions][:split]
            # beginning of ring
            self.index_cache.gpu_cached_mlp_indices[:new_ring_idx] = self.index_cache.cpu_cached_mlp_indices[cpu_diff_positions][split:]
        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Updated MLP cached indices in {time.time() - start} seconds')
            # print(f'new gpu mlp indices: {self.index_cache.gpu_cached_mlp_indices}')
            # print(f'cpu index diff: {cpu_index_diff}')

        # Index and overwrite the diff (much faster than full reindexing)
        #   e.g. adjacent tokens in 1.3b layer 5 have ~90% overlap of sparse indices
        #        adjacent tokens in 1.3b layer 19 have ~60% overlap of sparse indices
        # so tricksy!
        # torch.cuda.synchronize()
        # start = time.time()

        # if cpu_index_diff.off_elements.size(0) > 0:
            # CPU grabs missing indices from disk
            # TODO - broken copy
            # self.index_cache.indexed_fc1_weight[cpu_index_diff.off_positions].copy_(self.inputs.get_weight('fc1.weight')[cpu_index_diff.off_elements])
            # self.index_cache.indexed_fc1_bias[cpu_index_diff.off_positions].copy_(self.inputs.get_weight('fc1.bias')[cpu_index_diff.off_elements])
            # self.index_cache.indexed_fc2_weight[cpu_index_diff.off_positions].copy_(self.inputs.get_weight('fc2.weight')[cpu_index_diff.off_elements])

        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Indexed disk MLP weights in {time.time() - start} seconds')
            # torch.set_printoptions(profile='full', sci_mode=False)
            # print(f'Expecting indices: {self.tricksy_context.indices.mlp_indices_buffer_cpu}')
            # print(f'Expecting fc1 weights: {self.inputs.get_weight("fc1.weight")[self.tricksy_context.indices.mlp_indices_buffer_cpu][:, 0]}')
            # torch.set_printoptions(profile='default')
        # print(f'indexed_fc1_weight is pinned: {self.index_cache.indexed_fc1_weight.is_pinned()}')
        # print(f'indexed_fc1_bias is pinned: {self.index_cache.indexed_fc1_bias.is_pinned()}')
        # print(f'indexed_fc2_weight is pinned: {self.index_cache.indexed_fc2_weight.is_pinned()}')

        # print(f'MLP CPU diff shape: {cpu_diff_positions.shape}')
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'MLP CPU diff: {cpu_diff_positions.sort()[0]}')
        # torch.cuda.synchronize()
        # start = time.time()
        self.fc1_weight_diff = torch.empty((cpu_diff_positions.size(0), self.config.hidden_size), dtype=self.tricksy_config.dtype, device='cuda')
        self.fc1_bias_diff = torch.empty((cpu_diff_positions.size(0)), dtype=self.tricksy_config.dtype, device='cuda')
        self.fc2_weight_diff = torch.empty((cpu_diff_positions.size(0), self.config.hidden_size), dtype=self.tricksy_config.dtype, device='cuda')
        fc1wd = self.index_cache.indexed_fc1_weight[cpu_diff_positions]
        fc1bd = self.index_cache.indexed_fc1_bias[cpu_diff_positions]
        fc2wd = self.index_cache.indexed_fc2_weight[cpu_diff_positions]
        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Indexed cpu MLP weights in {time.time() - start} seconds')
        # torch.cuda.synchronize()
        # start = time.time()
        fc1wd = fc1wd.pin_memory()
        fc1bd = fc1bd.pin_memory()
        fc2wd = fc2wd.pin_memory()
        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Pinned memory in {time.time() - start} seconds')
        # torch.cuda.synchronize()
        # start = time.time()
        self.fc1_weight_diff.copy_(fc1wd, non_blocking=True)
        self.fc1_bias_diff.copy_(fc1bd, non_blocking=True)
        self.fc2_weight_diff.copy_(fc2wd, non_blocking=True)
        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'copied to GPU in {time.time() - start} seconds')
            # torch.set_printoptions(profile='full', sci_mode=False)
            # print(f'GPU fc1 weights before copying diff: {self.fc1_weight[:, 0]}')
            # torch.set_printoptions(profile='default')

        # torch.cuda.synchronize()
        # start = time.time()
        # self.fc1_weight[gpu_index_diff.off_positions].copy_(self.index_cache.indexed_fc1_weight[cpu_diff_positions], non_blocking=True)
        # self.fc1_weight.index_copy_(0, gpu_index_diff.off_positions.to('cuda'), fc1w_diff)
        # self.fc1_bias[gpu_index_diff.off_positions].copy_(self.index_cache.indexed_fc1_bias[cpu_diff_positions], non_blocking=True)
        # self.fc1_bias.index_copy_(0, gpu_index_diff.off_positions.to('cuda'), fc1b_diff)
        # self.fc2_weight[gpu_index_diff.off_positions].copy_(self.index_cache.indexed_fc2_weight[cpu_diff_positions], non_blocking=True)
        # self.fc2_weight.index_copy_(0, gpu_index_diff.off_positions.to('cuda'), fc2w_diff)
        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            # torch.set_printoptions(profile='full', sci_mode=False)
            # print(f'GPU indices after overwriting diff: {self.index_cache.gpu_cached_mlp_indices}')
            # print(f'GPU fc1 weights after copying diff: {torch.cat([self.fc1_weight, self.fc1_weight_diff])[:, 0]}')
            # torch.set_printoptions(profile='default')
            # print(f'Finished loading MLP weights in {time.time() - beginning} seconds')
            # actual = (torch.cat([self.fc1_weight, self.fc1_weight_diff])[:, 0]).cpu()
            # expected = self.inputs.get_weight("fc1.weight")[self.tricksy_context.indices.mlp_indices_buffer_cpu][:, 0]
            # weight_diff = torch.count_nonzero(~torch.isin(expected, actual))
            # if weight_diff > 1:
            #     raise Exception(f'fc1 weight diff is non-zero: {weight_diff}')
            # print(f'Applied sparse mlp weight diff on GPU in {time.time() - start} seconds')

    def forward(self, *args, **kwargs):
        torch.cuda.synchronize()
        # start = time.time()
        
        # print(f'=== Layer {self.tricksy_context.layer} ===')
        self.inputs.next_layer.load_weights(self.tricksy_context)

        # torch.cuda.synchronize()
        # start = time.time()
        out = super().forward(*args, **kwargs)
        # torch.cuda.synchronize()
        # print(f'Computed Attention + MLP forward in {time.time() - start} seconds')
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            # print(f'Decoder layer out: {out}')

        if self.tricksy_config.full_offload:
            self.fc1_weight = self.fc2_weight = self.final_layer_norm_weight = self.fc1_bias = self.fc2_bias = self.final_layer_norm_bias = None
        elif self.tricksy_context.is_prompt_phase:
            # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            #     print(f'gpu indices keeping sparse weights in prompt phase: {self.index_cache.gpu_cached_mlp_indices}')
            # Only keep sparse weights on GPU after prompt phase
            self.fc1_weight = self.fc1_weight[self.index_cache.gpu_cached_mlp_indices.to('cuda')]
            self.fc1_bias = self.fc1_bias[self.index_cache.gpu_cached_mlp_indices.to('cuda')]
            self.fc2_weight = self.fc2_weight[self.index_cache.gpu_cached_mlp_indices.to('cuda')]

        # Update ring buffers
        # torch.cuda.synchronize()
        # start = time.time()
        if not self.tricksy_config.full_offload:
            prev_ring_idx = self.ring_idx
            self.ring_idx = (self.ring_idx + self.fc1_weight_diff.size(0)) % self.fc1_weight.size(0)
            if self.ring_idx > prev_ring_idx:
                self.fc1_weight[prev_ring_idx:self.ring_idx] = self.fc1_weight_diff
                self.fc1_bias[prev_ring_idx:self.ring_idx] = self.fc1_bias_diff
                self.fc2_weight[prev_ring_idx:self.ring_idx] = self.fc2_weight_diff
            elif self.fc1_weight_diff.size(0) > 0:
                split = self.fc1_weight_diff.size(0) - self.ring_idx
                self.fc1_weight[prev_ring_idx:] = self.fc1_weight_diff[:split]
                self.fc1_weight[:self.ring_idx] = self.fc1_weight_diff[split:]
                self.fc1_bias[prev_ring_idx:] = self.fc1_bias_diff[:split]
                self.fc1_bias[:self.ring_idx] = self.fc1_bias_diff[split:]
                self.fc2_weight[prev_ring_idx:] = self.fc2_weight_diff[:split]
                self.fc2_weight[:self.ring_idx] = self.fc2_weight_diff[split:]
        self.fc1_weight_diff = self.fc2_weight_diff = self.fc1_bias_diff = None
        # torch.cuda.synchronize()
        # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
        #     print(f'Updated ring buffers in {time.time() - start} seconds')

            # if self.tricksy_context.layer == 9 or self.tricksy_context.layer == 40:
            #     torch.set_printoptions(profile='full', sci_mode=False)
            #     print(f'fc1 after sparse weight index in prompt phase: {self.fc1_weight[:, 0]}')
            #     torch.set_printoptions(profile='default')

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
            sparsity_predictors = [
                load_mlp_sparsity_predictor(disk_weights.weight_path, pretrained_layer_num, tricksy_config.dtype),
                load_attn_head_sparsity_predictor(disk_weights.weight_path, pretrained_layer_num, tricksy_config.dtype),
            ]
            if i == self.config.num_hidden_layers - 1:
                # Add predictor to compute attn head sparsity based on input embedding output
                sparsity_predictors.append(load_attn_head_sparsity_predictor(disk_weights.weight_path, -1, tricksy_config.dtype))
            if sparsity_predictors[0] is None:
                sparsity_predictors[0] = lambda x: F.linear(x, torch.rand((self.config.ffn_dim, self.config.hidden_size), device='cuda', dtype=tricksy_config.dtype))
            if sparsity_predictors[1] is None:
                sparsity_predictors[1] = lambda x: F.linear(x, torch.rand((self.config.num_attention_heads, self.config.hidden_size), device='cuda', dtype=tricksy_config.dtype))
            if len(sparsity_predictors) >= 3 and sparsity_predictors[2] is None:
                sparsity_predictors[2] = lambda x: F.linear(x, torch.rand((self.config.num_attention_heads, self.config.hidden_size), device='cuda', dtype=tricksy_config.dtype))
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
        # self.layers[0].self_attn.load_weights(self.tricksy_context)

        out = super().forward(*args, **kwargs)

        # Wait for output embedding weights to get to GPU
        torch.cuda.synchronize()

        # print(f'=== Pre-Output Embeddings: Layer {self.tricksy_context.layer} ===')
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
        # Disable KV cache since attention head sparsity results in missing keys/values (cpu -> gpu bandwidth is the bottleneck anyway)
        # disk_weights.config.use_cache = False
        self.config = disk_weights.config
        print(f'config: {self.config}')
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
        print(f'Full forward in {time.time() - start} seconds')
        self.tricksy_context.layer = 0
        return out

    def generate(self, *args, **kwargs):
        # Load input embeddings for first token
        self.model.decoder.load_weights(self.tricksy_context)
        torch.cuda.synchronize()
        out = super().generate(*args, **kwargs)
        print(f'\n===\nDecoding tok/s: {1 / (sum(self.tricksy_context.forward_times[1:]) / (len(self.tricksy_context.forward_times) - 1))}')
        return out