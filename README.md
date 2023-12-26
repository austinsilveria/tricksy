# Tricksy
Fast approximate inference on a single GPU with sparsity aware offloading
* ~15x faster than naive offloading
* ~7x faster than partial dense offloading with same GPU memory usage

```python
import torch
from transformers import AutoTokenizer, set_seed

from tricksy.modeling_tricksy import TricksyOPTForCausalLM, OPTDiskWeights
from tricksy.configuration_tricksy import TricksyConfig

set_seed(42)

# 60 GB (16 bit)
model_name = 'facebook/opt-30b'
disk_weights = OPTDiskWeights(model_name)
tricksy_model = TricksyOPTForCausalLM(TricksyConfig(disk_weights.config), disk_weights)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = 'Making pesto from scratch can be done with these ingredients in 4 simple steps:\nStep 1'
inputs = tokenizer(prompt, return_tensors='pt')

generate_ids = tricksy_model.generate(inputs.input_ids.to('cuda'), max_length=100, do_sample=True, top_k=50, top_p=0.9)
print(f'\n{(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))[0]}')

print(f'\n===\nCurrent GPU mem usage: {torch.cuda.memory_allocated("cuda") / 1024 ** 3} GB')
print(f'Max GPU mem usage: {torch.cuda.max_memory_allocated("cuda") / 1024 ** 3} GB')
```
~~~
===
Decoding tok/s: 3.7130587408915883
===


Making pesto from scratch can be done with these ingredients in 4 simple steps:
Step 1: Put the raw pine nuts in the food processor.
Step 2: Add the basil and 1/2 cup of the parmesan.
Step 3: Add a little water to blend.
Step 4: Blend for 30 seconds.

It’s that simple! The pine nuts bring a great texture to the pesto, making it super tasty and creamy. If you like

===
Current GPU mem usage: 33.99628019332886 GB
Max GPU mem usage: 34.945725440979004 GB
~~~

### Usage
```bash
git clone https://github.com/austinsilveria/tricksy.git
cd tricksy
python3 generate.py
```

### Description
MLP layers of large language models are naturally sparse--e.g. > 99% of layer 3's and > 90% of layer 20's neurons in OPT-1.3b have no effect (due to relu) for most inputs. Adjacent tokens also share a significant number of active neurons--e.g. for layers 1-7 of OPT-1.3b, > 90% of neurons active for token k are also active for token k + 1 (and 60-65% for layers 20-23).

We exploit this natural sparisity to minimize CPU-GPU data transfer.

At **initialization**, we:
1. Store a subset of each MLP layer (e.g. 30%) and full attention layers on the GPU
2. Store full MLP layers in CPU RAM
3. Store a cache of which neuron indices we currently have on the GPU

**Before each** decoder layer's **foward pass**, we:
1. Predict active MLP neurons based on the attention layer input (following [Deja Vu](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf))

**During** each decoder layer's **attention** computation, we, asynchronously on the CPU:
1. Compute the difference between the set of predicted active neuron indices and the set of neuron indices we currently have on the GPU
2. Index those neurons from CPU RAM
3. Copy them to the GPU
4. Update the layer's neuron indices cache

And finally, **during** each decoder layer's **MLP** computation, we:
1. Concatenate the newly received neuron diff with our existing neurons
2. Compute the MLP

   **Note**: As long as fully-connected layer 1 and fully-connected layer 2 share the same neuron ordering, the full two layer computation is invariant with respect to neuron ordering.
4. Overwrite a subset of our neuron buffer with the diff (FIFO order)
5. Delete the diff

### Limitations
1. This is approximate inference. The active neuron predictors do not have perfect recall, leading to slight accuracy degradation. See the [Deja Vu paper](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf) for an in depth evaluation.

### Potential Improvements
1. Evaluations--to push the sparsity levels, we need evaluations to measure accuracy degradation.
2. Indexing the non-contiguous neuron diff from CPU RAM comes nowhere near saturating CPU-RAM memory bandwidth. We may be able to improve this with a custom C++ indexer.
3. Early layers are extremely sparse while later layers are less sparse--perhaps we can allocate smaller GPU neuron buffers to early layers to free up space for larger buffers for later layers.
4. Applying an advanced index to a pinned tensor in PyTorch will return an unpinned copy of the indexed data, which means it needs to be recopied to pinned memory before it can be sent to the GPU. If we can override this default PyTorch behavior to allow direct CPU-GPU copying from a specified advanced index without intermediate copies, we should get a nice speedup.
