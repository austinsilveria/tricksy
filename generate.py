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