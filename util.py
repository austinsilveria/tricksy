from typing import List, Callable
import torch
from torch.nn import functional as F

import numpy as np

import os

np_dtype_to_torch_dtype = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int32: torch.int32,
    np.int64: torch.int64,
    bool: torch.bool,
}

def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

def batch_copy(sources: List[torch.Tensor], copy_stream, indices=None, device='cuda'):
    with torch.cuda.stream(copy_stream):
        out = ()
        for src in sources:
            indexed = src[indices] if indices is not None else src
            dst = torch.empty(indexed.shape, device=device, dtype=src.dtype)
            dst.copy_(indexed, non_blocking=True)
            out += (dst,)

        return out

def mmap_to_tensor(torch_wrapped_mmap, pin_memory=False) -> torch.Tensor:
    out = torch.empty(torch_wrapped_mmap.shape, dtype=torch_wrapped_mmap.dtype, device='cpu', pin_memory=pin_memory)
    out.copy_(torch_wrapped_mmap)
    return out

def compute_index_diff(new_indices, current_indices):
    return (
        # Compute positions of current indices where new indices does not contain the element
        torch.isin(current_indices, new_indices, assume_unique=True, invert=True),
        # Compute elements of new indices not contained current indices
        torch.tensor(
            list(set(new_indices.tolist()).difference(set(current_indices.tolist()))),
            device='cpu',
            dtype=torch.int32,
            pin_memory=True
        )
    )

def topk_and_threshold(x, k, threshold):
    print(f'sparsity predictor output: {x}')
    vals, indices = torch.topk(x, k, sorted=False)
    # return indices[vals > threshold].short()
    return indices.short()

def load_predictor(path_prefix: str, dtype: torch.dtype, device: str='cuda') -> Callable:
    path = lambda i: os.path.expanduser(f'{path_prefix}{i}.weight')
    if os.path.exists(path(1)):
        l1 = torch.load(path(1)).to(device).to(dtype)
        l2 = torch.load(path(2)).to(device).to(dtype)
        return lambda x: F.linear(F.linear(x, l1), l2)
    else:
        print(f'could not find predictor at {path(1)}')
        return None