import torch

from transformers import H3Config, H3ForCausalLM


model = H3ForCausalLM(H3Config(use_cache=False))

# for name, param in model.named_parameters():
#     print(name, param.shape)

input_ids = torch.tensor([[101]])

outputs = model(input_ids)
