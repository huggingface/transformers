from transformers import GITConfig, GITForCausalLM


# Initializing a GIT microsoft/git-base style configuration
configuration = GITConfig()

# Initializing a model (with random weights) from the microsoft/git-base style configuration
model = GITForCausalLM(configuration)

import torch


pixel_values = torch.randn(1, 3, 224, 224)
input_ids = torch.tensor([[101]])

dict_outputs = model(input_ids, pixel_values=pixel_values, output_hidden_states=True)

for k, v in dict_outputs.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k, len(v))

tuple_outputs = model(input_ids, pixel_values=pixel_values, output_hidden_states=True, return_dict=False)

print(len(tuple_outputs))
