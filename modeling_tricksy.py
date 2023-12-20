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
from util import batch_copy, getenv, load_predictor, mmap_to_tensor, topk_and_threshold

TRICKSY_WEIGHTS_PATH = '~/tricksy-weights/'
DEBUG = getenv("DEBUG")

def load_attn_head_sparsity_predictor(model_name: str, layer_num: int, dtype: torch.dtype, device: str = 'cuda') -> Callable:
    path_prefix = f'{TRICKSY_WEIGHTS_PATH}{model_name}/decoder.layers.{layer_num}.attn-head-sparsity-predictor.'
    return load_predictor(path_prefix, dtype, device=device)

def load_mlp_sparsity_predictor(model_name: str, layer_num: int, dtype: torch.dtype, device: str = 'cuda') -> Callable:
    path_prefix = f'{TRICKSY_WEIGHTS_PATH}{model_name}/decoder.layers.{layer_num}.attn.mlp-sparsity-predictor.'
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
        cached_head_indices: Optional[torch.Tensor] = None,
    ):
        # [n_head * min_head_sparsity, head_dim * 4, hidden_size]
        self.indexed_weights = indexed_weights
        # [n_head * min_head_sparsity, 3, head_dim]
        self.indexed_biases = indexed_biases

        # [n_head * min_head_sparsity]
        self.cached_head_indices = cached_head_indices

class SparseMLPCache:
    def __init__(
        self,
        indexed_fc1_weight: Optional[torch.Tensor] = None,
        indexed_fc1_bias: Optional[torch.Tensor] = None,
        indexed_fc2_weight: Optional[torch.Tensor] = None,
        cached_ffn_indices: Optional[torch.Tensor] = None,
    ):
        # [ffn_embed_dim * min_mlp_sparsity, hidden_size]
        self.indexed_fc1_weight = indexed_fc1_weight
        # [ffn_embed_dim * min_mlp_sparsity]
        self.indexed_fc1_bias = indexed_fc1_bias
        # [ffn_embed_dim * min_mlp_sparsity, hidden_size] (stored in transpose for efficient indexing)
        self.indexed_fc2_weight = indexed_fc2_weight

        # [ffn_embed_dim * min_mlp_sparsity]
        self.cached_ffn_indices = cached_ffn_indices

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
        self.input_embedding_indices_gpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cuda'
        )
        self.head_indices_gpu = torch.empty(
            (int(opt_config.num_attention_heads * tricksy_config.min_head_sparsity),),
            dtype=torch.int8,
            device='cuda'
        )
        self.mlp_indices_gpu = torch.empty(
            (int(opt_config.ffn_dim * tricksy_config.min_mlp_sparsity),),
            dtype=torch.int16,
            device='cuda'
        )
        self.output_embedding_indices_gpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cuda'
        )

        self.input_embedding_indices_cpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cpu',
            pin_memory=True,
        )
        self.head_indices_cpu = torch.empty(
            (int(opt_config.num_attention_heads * tricksy_config.min_head_sparsity),),
            dtype=torch.int8,
            device='cpu',
            pin_memory=True,
        )
        self.mlp_indices_cpu = torch.empty(
            (int(opt_config.ffn_dim * tricksy_config.min_mlp_sparsity),),
            dtype=torch.int16,
            device='cpu',
            pin_memory=True,
        )
        self.output_embedding_indices_cpu = torch.empty(
            (int(opt_config.vocab_size * tricksy_config.min_embedding_sparsity),),
            dtype=torch.int16,
            device='cpu',
            pin_memory=True,
        )

        # Default stream blocks until indices are copied to CPU
        self.index_copy_stream = torch.cuda.default_stream()
    
    def copy_input_embedding_indices_to_cpu(self):
        self.input_embedding_indices_cpu = batch_copy([self.input_embedding_indices_gpu], self.index_copy_stream, device='cpu')[0]

    def copy_head_indices_to_cpu(self):
        self.head_indices_cpu = batch_copy([self.head_indices_gpu], self.index_copy_stream, device='cpu')[0]

    def copy_mlp_indices_to_cpu(self):
        self.mlp_indices_cpu = batch_copy([self.mlp_indices_gpu], self.index_copy_stream, device='cpu')[0]

    def copy_output_embedding_indices_to_cpu(self):
        self.output_embedding_indices_cpu = batch_copy([self.output_embedding_indices_gpu], self.index_copy_stream, device='cpu')[0]

class OPTDiskWeights:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_suffix = model_name.split('/')[-1]
        self.config = OPTConfig.from_pretrained(model_name)

        self.weight_path = f'{TRICKSY_WEIGHTS_PATH}{self.model_suffix}/'

        with init_empty_weights():
            model = OPTModel(self.config)
        self.state_dict = model.state_dict()
        print(f'state_dict keys: {self.state_dict.keys()}')

        print(f'checking path: {self.weight_path}decoder.embed_tokens.weight')
        if not os.path.exists(f'{self.weight_path}decoder.embed_tokens.weight'):
            # Download weights and write memmap files
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
                (f'{layer_prefix}self_attn.catted_head_weights', (self.config.num_attention_heads, head_dim * 4, self.config.hidden_size)),
                (f'{layer_prefix}self_attn.catted_head_biases', (self.config.num_attention_heads, 3, head_dim)),
            ])

        print(f'loading memmap weights for {self.model_name}')
        self.weights = { key: self.load_weight(key) for key in self.state_dict.keys() }

    def load_weight(self, key: str):
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
            fc2t = torch.from_numpy(np.array(self.load_weight(f'{layer_prefix}fc2.weight')[:])).t().contiguous().clone()
            np.memmap(f'{self.weight_path}decoder.layers.{i}.fc2.weight', dtype='float16', mode='w+', shape=fc2t.shape)[:] = fc2t

            # Attention weights by head
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            qw = mmap_to_tensor(self.load_weight(f'{layer_prefix}self_attn.q_proj.weight')[:])
            kw = mmap_to_tensor(self.load_weight(f'{layer_prefix}self_attn.k_proj.weight')[:])
            vw = mmap_to_tensor(self.load_weight(f'{layer_prefix}self_attn.v_proj.weight')[:])
            ow = mmap_to_tensor(self.load_weight(f'{layer_prefix}self_attn.out_proj.weight')[:])
            pre_cat_shape = (self.config.num_attention_heads, head_dim, self.config.hidden_size)
            # [head, head_dim * 4, hidden_size]
            catted_head_weights = torch.cat(
                [qw.view(pre_cat_shape).clone(), kw.view(pre_cat_shape).clone(), vw.view(pre_cat_shape).clone(), ow.T.view(pre_cat_shape).clone(),],
                dim=1,
            ).contiguous().clone()
            np.memmap(f'{self.weight_path}{layer_prefix}self_attn.catted_head_weights', dtype='float16', mode='w+', shape=catted_head_weights.shape)[:] =\
                catted_head_weights.numpy()

            # Attention biases by head
            qb = mmap_to_tensor(self.load_weight(f'{layer_prefix}self_attn.q_proj.bias')[:])
            kb = mmap_to_tensor(self.load_weight(f'{layer_prefix}self_attn.k_proj.bias')[:])
            vb = mmap_to_tensor(self.load_weight(f'{layer_prefix}self_attn.v_proj.bias')[:])
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

class TricksyLayer:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def load_weights(self, tricksy_context: TricksyContext):
        pass

class TricksyLayerInputs:
    def __init__(
        self,
        disk_weight_dict: Dict[str, np.memmap] = None,
        layer_key_prefix: str = None,
        next_layer: TricksyLayer = None,
        sparsity_predictors: List[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.get_weight = lambda key: disk_weight_dict[f'{layer_key_prefix}{key}']
        self.disk_weight_dict = disk_weight_dict
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
        if self.tricksy_context.is_prompt_phase:
            self.weight = None
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

        random_head_indices = torch.randperm(self.config.num_attention_heads, device='cpu', dtype=torch.int32)[:int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity)]
        self.index_cache = SparseAttentionCache(cached_head_indices=random_head_indices)

        self.catted_weights = self.catted_biases = self.out_proj_bias = self.layer_norm_weight = self.layer_norm_bias = None
        self.q_proj = lambda x: F.linear(
            x,
            self.catted_weights[:, :self.head_dim, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size),
            self.catted_biases[:, 0, :].reshape(self.head_dim * self.catted_biases.size(0))
        )
        self.k_proj = lambda x: F.linear(
            x,
            self.catted_weights[:, self.head_dim:self.head_dim * 2, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size),
            self.catted_biases[:, 1, :].reshape(self.head_dim * self.catted_biases.size(0))
        )
        self.v_proj = lambda x: F.linear(
            x,
            self.catted_weights[:, self.head_dim * 2:self.head_dim * 3, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size),
            self.catted_biases[:, 2, :].reshape(self.head_dim * self.catted_biases.size(0))
        )
        self.out_proj = lambda x: F.linear(
            x,
            self.catted_weights[:, self.head_dim * 3:, :].reshape(self.head_dim * self.catted_weights.size(0), self.config.hidden_size).T,
            self.out_proj_bias,
        )
        self.layer_norm = lambda x: F.layer_norm(x, (self.config.hidden_size,), self.layer_norm_weight, self.layer_norm_bias)

    def load_weights(self, tricksy_context: TricksyContext):
        if self.tricksy_context.is_prompt_phase:
            # Full weights for prompt phase
            self.catted_weights, self.catted_biases, self.out_proj_bias, self.layer_norm_weight, self.layer_norm_bias = batch_copy(
                [
                    self.inputs.get_weight('self_attn.catted_head_weights'),
                    self.inputs.get_weight('self_attn.catted_head_biases'),
                    self.inputs.get_weight('self_attn.out_proj.bias'),
                    self.inputs.get_weight('self_attn_layer_norm.weight'),
                    self.inputs.get_weight('self_attn_layer_norm.bias'),
                ],
                tricksy_context.load_weight_stream,
            )
            self.index_cache.cached_head_indices[:tricksy_context.indices.head_indices_cpu.size(0)] = tricksy_context.indices.head_indices_cpu
            self.index_cache.indexed_weights = self.inputs.get_weight('self_attn.catted_head_weights')[self.index_cache.cached_head_indices].contiguous().clone().pin_memory()
            self.index_cache.indexed_biases = self.inputs.get_weight('self_attn.catted_head_biases')[self.index_cache.cached_head_indices].contiguous().clone().pin_memory()
            return

        self.num_heads = self.index_cache.cached_head_indices.size(0)
        # Forward pass uses this to reshape attention output before output projection
        self.embed_dim = self.num_heads * self.head_dim

        print(f'new indices shape: {tricksy_context.indices.head_indices_cpu.shape}')
        torch.cuda.synchronize()
        beginning = time.time()
        start = time.time()
        # Compute different elements between this token's indices and last token's
        off_elements = torch.tensor(
            list(set(tricksy_context.indices.head_indices_cpu.tolist()).difference(set(self.index_cache.cached_head_indices.tolist()))),
            device='cpu',
            dtype=torch.int32,
            pin_memory=True
        )
        # Compute positions where this token's indices are different from last token's
        off_positions = torch.nonzero(~torch.isin(self.index_cache.cached_head_indices, tricksy_context.indices.head_indices_cpu, assume_unique=True)).flatten()[:off_elements.size(0)]
        torch.cuda.synchronize()
        print(f'Computed off head positions and elements in {time.time() - start} seconds')
        print(f'new head indices shape: {tricksy_context.indices.head_indices_cpu.shape}')
        print(f'head off_elements.shape: {off_elements.shape}')
        print(f'head off_positions.shape: {off_positions.shape}')
        # off_elements = torch.randperm(self.config.num_attention_heads, pin_memory=True)[:int(self.index_cache.cached_head_indices.size(0) * self.tricksy_config.adjacent_head_sparsity)]
        # off_positions = torch.randperm(self.index_cache.cached_head_indices.size(0), pin_memory=True)[:int(self.index_cache.cached_head_indices.size(0) * self.tricksy_config.adjacent_head_sparsity)]
        off_positions_gpu = off_positions.clone().to('cuda')

        if off_elements.size(0) == 0:
            print(f'No new head indices, skipping load_weights')
            return

        torch.cuda.synchronize()
        start = time.time()
        self.index_cache.cached_head_indices[off_positions] = off_elements
        torch.cuda.synchronize()
        print(f'Updated cached head indices in {time.time() - start} seconds')

        # Index and overwrite the diff (much faster than full reindexing)
        #   e.g. adjacent tokens in 1.3b layer 5 have ~90% overlap of sparse indices
        #        adjacent tokens in 1.3b layer 19 have ~60% overlap of sparse indices
        torch.cuda.synchronize()
        start = time.time()
        catted_weights_diff = mmap_to_tensor(self.inputs.get_weight('self_attn.catted_head_weights')[off_elements], pin_memory=True)
        catted_biases_diff = mmap_to_tensor(self.inputs.get_weight('self_attn.catted_head_biases')[off_elements], pin_memory=True)
        self.index_cache.indexed_weights[off_positions] = catted_weights_diff
        self.index_cache.indexed_biases[off_positions] = catted_biases_diff
        torch.cuda.synchronize()
        print(f'Indexed heads in {time.time() - start} seconds')

        print(f'catted_weights_diff shape: {catted_weights_diff.shape}')
        print(f'catted_biases_diff shape: {catted_biases_diff.shape}')

        print(f'indexed_weights is pinned: {self.index_cache.indexed_weights.is_pinned()}')
        print(f'indexed_biases is pinned: {self.index_cache.indexed_biases.is_pinned()}')
        torch.cuda.synchronize()
        start = time.time()
        self.catted_weights[off_positions_gpu].copy_(catted_weights_diff, non_blocking=True)
        self.catted_biases[off_positions_gpu].copy_(catted_biases_diff, non_blocking=True)
        torch.cuda.synchronize()
        print(f'Copied sparse attention weights in {time.time() - start} seconds')

        print(f'Finished loading weights in {time.time() - beginning} seconds')
        
    def forward(self, hidden_states, **kwargs):
        if len(self.inputs.sparsity_predictors) > 2:
            torch.cuda.synchronize()
            start = time.time()
            # Predict head sparsity based on input embedding output
            self.tricksy_context.indices.head_indices_gpu = topk_and_threshold(
                self.inputs.sparsity_predictors[2](hidden_states)[0, -1, :],
            int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity),
                self.tricksy_config.min_head_probability,
            )
            self.tricksy_context.indices.copy_head_indices_to_cpu()
            torch.cuda.synchronize()
            print(f'Computed head indices based on input embeddings in {time.time() - start} seconds')
            self.load_weights(self.tricksy_context)

        # Wait for weights to get to GPU
        torch.cuda.synchronize()

        start = time.time()
        # Predict MLP sparsity based on attention input
        self.tricksy_context.indices.mlp_indices_gpu = topk_and_threshold(
            self.inputs.sparsity_predictors[0](hidden_states)[0, -1, :],
            int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity),
            self.tricksy_config.min_mlp_probability,
        )
        self.tricksy_context.indices.copy_mlp_indices_to_cpu()
        torch.cuda.synchronize()
        print(f'Computed MLP indices in {time.time() - start} seconds')

        self.inputs.next_layer.load_weights(self.tricksy_context)

        torch.cuda.synchronize()
        start = time.time()
        out = super().forward(self.layer_norm(hidden_states), **kwargs)
        torch.cuda.synchronize()
        print(f'Computed attention forward in {time.time() - start} seconds')

        if self.tricksy_context.is_prompt_phase:
            # Only keep sparse weights on GPU after prompt phase
            self.catted_weights = self.catted_weights[self.index_cache.cached_head_indices.to('cuda')]
            self.catted_biases = self.catted_biases[self.index_cache.cached_head_indices.to('cuda')]

        torch.cuda.synchronize()
        start = time.time()
        if self.inputs.sparsity_predictors[1] is not None:
            # Predict head sparsity based on MLP input
            self.tricksy_context.indices.head_indices_gpu = topk_and_threshold(
                self.inputs.sparsity_predictors[1](out)[0, -1, :],
                int(self.config.num_attention_heads * self.tricksy_config.min_head_sparsity),
                self.tricksy_config.min_head_probability,
            )
            self.tricksy_context.indices.copy_head_indices_to_cpu()
            torch.cuda.synchronize()
            print(f'Computed head indices in {time.time() - start} seconds')

        return out

class TricksyOPTDecoderLayer(OPTDecoderLayer):
    def __init__(self, tricksy_config: TricksyConfig, inputs: TricksyLayerInputs, tricksy_context: TricksyContext):
        nn.Module.__init__(self)
        self.tricksy_config = tricksy_config
        self.config = tricksy_config.opt_config
        self.embed_dim = self.config.hidden_size

        self.tricksy_context = tricksy_context
        self.self_attn_layer_inputs = TricksyLayerInputs(
            disk_weight_dict=inputs.disk_weight_dict,
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
        random_mlp_indices = torch.randperm(self.config.ffn_dim, device='cpu', dtype=torch.int32)[:int(self.config.ffn_dim * self.tricksy_config.min_mlp_sparsity)]
        self.index_cache = SparseMLPCache(cached_ffn_indices=random_mlp_indices)

        # identity since we move this to attention layer
        # extreme tricksy
        self.self_attn_layer_norm = lambda x: x

        self.fc1_weight = self.fc2_weight = self.final_layer_norm_weight = self.fc1_bias = self.fc2_bias = self.final_layer_norm_bias = None
        self.fc1 = lambda x: F.linear(x, self.fc1_weight, self.fc1_bias)
        self.fc2 = lambda x: F.linear(x, self.fc2_weight.T, self.fc2_bias)
        self.final_layer_norm = lambda x: F.layer_norm(x, (self.embed_dim,), self.final_layer_norm_weight, self.final_layer_norm_bias)

    def load_weights(self, tricksy_context: TricksyContext):
        if self.tricksy_context.is_prompt_phase:
            # Full weights for prompt phase
            self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias, self.final_layer_norm_weight, self.final_layer_norm_bias = batch_copy(
                [
                    mmap_to_tensor(self.inputs.get_weight('fc1.weight')[:]),
                    mmap_to_tensor(self.inputs.get_weight('fc1.bias')[:]),
                    mmap_to_tensor(self.inputs.get_weight('fc2.weight')[:]),
                    mmap_to_tensor(self.inputs.get_weight('fc2.bias')[:]),
                    mmap_to_tensor(self.inputs.get_weight('final_layer_norm.weight')[:]),
                    mmap_to_tensor(self.inputs.get_weight('final_layer_norm.bias')[:]),
                ],
                tricksy_context.load_weight_stream,
            )
            self.index_cache.cached_ffn_indices[:tricksy_context.indices.mlp_indices_cpu.size(0)] = tricksy_context.indices.mlp_indices_cpu
            self.index_cache.indexed_fc1_weight =\
                mmap_to_tensor(self.inputs.get_weight('fc1.weight')[self.index_cache.cached_ffn_indices]).contiguous().clone().pin_memory()
            self.index_cache.indexed_fc1_bias =\
                mmap_to_tensor(self.inputs.get_weight('fc1.bias')[self.index_cache.cached_ffn_indices]).contiguous().clone().pin_memory()
            self.index_cache.indexed_fc2_weight =\
                mmap_to_tensor(self.inputs.get_weight('fc2.weight')[self.index_cache.cached_ffn_indices]).contiguous().clone().pin_memory()
            return

        torch.cuda.synchronize()
        beginning = time.time()
        start = time.time()
        # Compute different elements between this token's indices and last token's
        off_elements = torch.tensor(
            list(set(tricksy_context.indices.mlp_indices_cpu.tolist()).difference(set(self.index_cache.cached_ffn_indices.tolist()))),
            device='cpu',
            dtype=torch.int32,
        )
        if off_elements.size(0) == 0:
            return

        # Compute positions where this token's indices are different from last token's
        off_positions = torch.nonzero(~torch.isin(self.index_cache.cached_ffn_indices, tricksy_context.indices.mlp_indices_cpu, assume_unique=True)).flatten()[:off_elements.size(0)]

        torch.cuda.synchronize()
        print(f'Computed off positions and elements in {time.time() - start} seconds')
        print(f'new mlp indices shape: {tricksy_context.indices.mlp_indices_cpu.shape}')
        print(f'off_elements.shape: {off_elements.shape}')
        print(f'off_positions.shape: {off_positions.shape}')
        # off_elements = torch.randperm(self.config.ffn_dim, pin_memory=False)[:int(self.index_cache.cached_ffn_indices.size(0) * self.tricksy_config.adjacent_mlp_sparsity)]
        # off_positions = torch.randperm(self.index_cache.cached_ffn_indices.size(0), pin_memory=False)[:int(self.index_cache.cached_ffn_indices.size(0) * self.tricksy_config.adjacent_mlp_sparsity)]
        off_positions_gpu = off_positions.clone().to('cuda')

        torch.cuda.synchronize()
        start = time.time()
        self.index_cache.cached_ffn_indices[off_positions] = off_elements
        torch.cuda.synchronize()
        print(f'Updated cached indices in {time.time() - start} seconds')

        # Index and overwrite the diff (much faster than full reindexing)
        #   e.g. adjacent tokens in 1.3b layer 5 have ~90% overlap of sparse indices
        #        adjacent tokens in 1.3b layer 19 have ~60% overlap of sparse indices
        # so tricksy!
        torch.cuda.synchronize()
        start = time.time()
        fc1_weight_diff = mmap_to_tensor(self.inputs.get_weight('fc1.weight')[off_elements], pin_memory=True)
        fc1_bias_diff = mmap_to_tensor(self.inputs.get_weight('fc1.bias')[off_elements], pin_memory=True)
        fc2_weight_diff = mmap_to_tensor(self.inputs.get_weight('fc2.weight')[off_elements], pin_memory=True)
        self.index_cache.indexed_fc1_weight[off_positions] = fc1_weight_diff
        self.index_cache.indexed_fc1_bias[off_positions] = fc1_bias_diff
        self.index_cache.indexed_fc2_weight[off_positions] = fc2_weight_diff
        torch.cuda.synchronize()
        print(f'Indexed weights in {time.time() - start} seconds')

        print(f'indexed_fc1_weight is pinned: {self.index_cache.indexed_fc1_weight.is_pinned()}')
        print(f'indexed_fc1_bias is pinned: {self.index_cache.indexed_fc1_bias.is_pinned()}')
        print(f'indexed_fc2_weight is pinned: {self.index_cache.indexed_fc2_weight.is_pinned()}')

        print(f'fc1_weight_diff shape: {fc1_weight_diff.shape}')
        torch.cuda.synchronize()
        start = time.time()
        self.fc1_weight[off_positions_gpu].copy_(fc1_weight_diff, non_blocking=True)
        self.fc1_bias[off_positions_gpu].copy_(fc1_bias_diff, non_blocking=True)
        self.fc2_weight[off_positions_gpu].copy_(fc2_weight_diff, non_blocking=True)
        torch.cuda.synchronize()
        print(f'Copied sparse mlp weight diff in {time.time() - start} seconds')

        print(f'Finished loading MLP weights in {time.time() - beginning} seconds')

    def forward(self, *args, **kwargs):
        torch.cuda.synchronize()
        
        print(f'=== Layer {self.tricksy_context.layer} ===')
        self.inputs.next_layer.load_weights(self.tricksy_context)

        torch.cuda.synchronize()
        start = time.time()
        out = super().forward(*args, **kwargs)
        torch.cuda.synchronize()
        print(f'Computed Attention + MLP forward in {time.time() - start} seconds')

        if self.tricksy_context.is_prompt_phase:
            # Only keep sparse weights on GPU after prompt phase
            self.fc1_weight = self.fc1_weight[self.index_cache.cached_ffn_indices.to('cuda')]
            self.fc1_bias = self.fc1_bias[self.index_cache.cached_ffn_indices.to('cuda')]
            self.fc2_weight = self.fc2_weight[self.index_cache.cached_ffn_indices.to('cuda')]

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
                load_mlp_sparsity_predictor(disk_weights.model_suffix, pretrained_layer_num, tricksy_config.dtype),
                load_attn_head_sparsity_predictor(disk_weights.model_suffix, pretrained_layer_num, tricksy_config.dtype),
            ]
            if i == self.config.num_hidden_layers - 1:
                # Add predictor to compute attn head sparsity based on input embedding output
                sparsity_predictors.append(load_attn_head_sparsity_predictor(disk_weights.model_suffix, -1, tricksy_config.dtype))
            if sparsity_predictors[0] is None:
                sparsity_predictors[0] = lambda x: F.linear(x, torch.rand((self.config.ffn_dim, self.config.hidden_size), device='cuda', dtype=tricksy_config.dtype))
            if sparsity_predictors[1] is None:
                sparsity_predictors[1] = lambda x: F.linear(x, torch.rand((self.config.num_attention_heads, self.config.hidden_size), device='cuda', dtype=tricksy_config.dtype))
            if len(sparsity_predictors) >= 3 and sparsity_predictors[2] is None:
                sparsity_predictors[2] = lambda x: F.linear(x, torch.rand((self.config.num_attention_heads, self.config.hidden_size), device='cuda', dtype=tricksy_config.dtype))
            self.layers.append(TricksyOPTDecoderLayer(
                tricksy_config,
                TricksyLayerInputs(
                    disk_weight_dict=disk_weights.weights,
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
        self.inputs = TricksyLayerInputs(disk_weight_dict=disk_weights.weights, layer_key_prefix='decoder.')

    def embed_tokens(self, x):
        return F.embedding(x, self.embed_tokens_weight, self.padding_idx)
    
    def load_weights(self, tricksy_context: TricksyContext):
        if self.embed_tokens_weight is None:
            self.embed_tokens_weight, self.embed_positions.weight = batch_copy(
                [
                    self.inputs.get_weight('embed_tokens.weight')[:],
                    self.inputs.get_weight('embed_positions.weight')[:],
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

        print(f'=== Pre-Output Embeddings: Layer {self.tricksy_context.layer} ===')
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
        disk_weights.config.use_cache = False
        self.config = disk_weights.config
        print(f'config: {self.config}')
        self.generation_config = GenerationConfig.from_model_config(self.config) if self.can_generate() else None

        self.tricksy_context = TricksyContext(tricksy_config, self.config)
        self.model = TricksyOPTModel(tricksy_config, disk_weights, self, self.tricksy_context)

        self.final_layer_norm_weight = self.lm_head_weight = self.final_layer_norm_bias = None
        # double stacking tricksy!
        self.final_layer_norm = lambda x: F.layer_norm(x, (self.config.hidden_size,), self.final_layer_norm_weight, self.final_layer_norm_bias)
        self.lm_head = lambda x: F.linear(self.final_layer_norm(x), self.lm_head_weight.T)

        self.inputs = TricksyLayerInputs(disk_weight_dict=disk_weights.weights, layer_key_prefix='decoder.', next_layer=self.model.decoder)
    
    def load_weights(self, tricksy_context: TricksyContext):
        if self.final_layer_norm_weight is None:
            self.final_layer_norm_weight, self.lm_head_weight, self.final_layer_norm_bias = batch_copy(
                [
                    self.inputs.get_weight('final_layer_norm.weight')[:],
                    self.inputs.get_weight('embed_tokens.weight')[:],
                    self.inputs.get_weight('final_layer_norm.bias')[:],
                ],
                tricksy_context.load_weight_stream,
            )
    
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        self.tricksy_context.layer = 0
        return out

    def generate(self, *args, **kwargs):
        # Load input embeddings for first token
        self.model.decoder.load_weights(self.tricksy_context)
        torch.cuda.synchronize()
        return super().generate(*args, **kwargs)