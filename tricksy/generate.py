import torch
from transformers import AutoTokenizer, TextStreamer, set_seed

from tricksy.modeling_tricksy import TricksyOPTForCausalLM, OPTDiskWeights
from tricksy.configuration_tricksy import TricksyConfig

set_seed(42)

# 13.4 GB (16 bit)
model_name = 'facebook/opt-6.7b'
disk_weights = OPTDiskWeights(model_name)
tricksy_model = TricksyOPTForCausalLM(TricksyConfig(disk_weights.config), disk_weights)
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

prompt = 'Making pesto from scratch can be done with these ingredients in 4 simple steps:\nStep 1'
inputs = tokenizer(prompt, return_tensors='pt')

print()
tricksy_model.generate(inputs.input_ids.to('cuda'), max_new_tokens=100, do_sample=True, top_k=50, top_p=0.9, streamer=streamer)

print(f'\n===\nDecoding tok/s: {1 / (sum(tricksy_model.tricksy_context.forward_times[1:]) / (len(tricksy_model.tricksy_context.forward_times) - 1))}')
print(f'Current GPU mem usage: {torch.cuda.memory_allocated("cuda") / 1024 ** 3} GB')
print(f'Max GPU mem usage: {torch.cuda.max_memory_allocated("cuda") / 1024 ** 3} GB\n===')