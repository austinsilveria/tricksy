import torch
from transformers import AutoTokenizer, set_seed

from modeling_tricksy import TricksyOPTForCausalLM, OPTDiskWeights
from configuration_tricksy import TricksyConfig

set_seed(3)
torch.manual_seed(2)

model_name = 'facebook/opt-30b'
disk_weights = OPTDiskWeights(model_name)
tricksy_model = TricksyOPTForCausalLM(TricksyConfig(disk_weights.config), disk_weights)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "what are we holding onto sam?"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = tricksy_model.generate(inputs.input_ids.to('cuda'), max_length=256, do_sample=True, top_k=50, top_p=0.95)
print(f'\n{(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))[0]}')

print(f'\n===\nCurrent GPU mem usage: {torch.cuda.memory_allocated("cuda") / 1024 ** 3} GB')
print(f'Max GPU mem usage: {torch.cuda.max_memory_allocated("cuda") / 1024 ** 3} GB')