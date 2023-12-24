import time
from typing import List, Callable
import torch
from torch.nn import functional as F

import numpy as np
import threading

import os
import psutil

np_dtype_to_torch_dtype = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int32: torch.int32,
    np.int64: torch.int64,
    bool: torch.bool,
}

class IndexDiff:
    def __init__(self, off_elements: torch.Tensor=None, off_positions: torch.Tensor=None, on_positions: torch.Tensor=None):
        self.off_elements = off_elements
        self.off_positions = off_positions
        self.on_positions = on_positions

    def __repr__(self):
        return f'IndexDiff(off_elements={self.off_elements}, off_positions={self.off_positions}, on_positions={self.on_positions})'

    def __str__(self):
        return self.__repr__()

def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

def batch_copy(sources: List[torch.Tensor], copy_stream, indices=None, device='cuda'):
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.stream(copy_stream):
        out = ()
        for src in sources:
            indexed = src[indices] if indices is not None else src
            dst = torch.empty(indexed.shape, device=device, dtype=src.dtype)
            dst.copy_(indexed, non_blocking=True)
            out += (dst,)

    torch.cuda.synchronize()
    # print(f'batch copy time: {time.time() - start}')
    return out

def mmap_to_tensor(torch_wrapped_mmap, pin_memory=False) -> torch.Tensor:
    out = torch.empty(torch_wrapped_mmap.shape, dtype=torch_wrapped_mmap.dtype, device='cpu', pin_memory=pin_memory)
    torch.cuda.synchronize()
    start = time.time()
    out.copy_(torch_wrapped_mmap)
    torch.cuda.synchronize()
    # print(f'mmap to tensor time: {time.time() - start}')
    return out

def parallel_memmap_read(memmap, num_threads=8):
    def read_chunk(memmap, start, end, results, index):
        results[index] = memmap[start:end]

    # Calculate chunk size
    chunk_size = len(memmap) // num_threads

    # Store results and threads
    results = [None] * num_threads
    threads = []

    # Create and start threads
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(memmap)
        thread = threading.Thread(target=read_chunk, args=(memmap, start, end, results, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Combine results
    return torch.from_numpy(np.concatenate(results))

# Assuming that each entry of cached_indices is step down the memory hierarchy,
# compute the diff at each level of the hierarchy.
#   e.g. the first loop computes the indices that the GPU does not have,
#        and the second loop computes the indices *of that diff* that the CPU does not have.
def compute_index_diffs(new_indices: torch.Tensor, cached_indices_list: List[torch.Tensor]):
    diffs = []
    current_diff = new_indices
    for cached_indices in cached_indices_list:
        # print(f'current diff: {current_diff}')
        if current_diff.size(0) == 0:
            # No need to go further down the hierarchy
            break

        # Compute elements of new indices not contained current indices
        off_elements = torch.tensor(
            list(set(current_diff.tolist()).difference(set(cached_indices.tolist()))),
            device='cpu',
            dtype=torch.int32,
            pin_memory=True
        )
        # Compute mask of current indices where new indices does not contain the element
        on_position_mask = torch.isin(cached_indices, current_diff, assume_unique=True)
        on_positions = torch.nonzero(on_position_mask).flatten()
        off_positions = torch.nonzero(~on_position_mask).flatten()[:off_elements.size(0)]

        diffs.append(IndexDiff(off_elements, off_positions, on_positions))
        current_diff = off_elements
    return diffs

def topk_and_threshold(x, k, threshold=1):
    # print(f'sparsity predictor output: {x}')
    vals, indices = torch.topk(x, k, sorted=True)
    return indices[vals > threshold].int()
    # return indices.int()

def load_predictor(path_prefix: str, dtype: torch.dtype, device: str='cuda') -> Callable:
    path = lambda i: os.path.expanduser(f'{path_prefix}{i}.weight')
    if os.path.exists(path(1)):
        proc = psutil.Process()
        # print(f'open files: {proc.open_files()}')
        # print(f'num open files: {len(proc.open_files())}')
        torch.cuda.synchronize()
        start = time.time()
        l1 = torch.load(path(1))
        torch.cuda.synchronize()
        # print(f'load predictor 1 time: {time.time() - start}')
        torch.cuda.synchronize()
        start = time.time()
        l2 = torch.load(path(2))
        torch.cuda.synchronize()
        # print(f'load predictor 2 time: {time.time() - start}')
        torch.cuda.synchronize()
        start = time.time()
        l1 = l1.to(device).to(dtype)
        l2 = l2.to(device).to(dtype)
        torch.cuda.synchronize()
        # print(f'predictor to cuda in time: {time.time() - start}')
        return lambda x: F.linear(F.linear(x, l1), l2)
    else:
        print(f'could not find predictor at {path(1)}')
        return None