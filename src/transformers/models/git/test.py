from transformers import GITConfig, GITModel


# Initializing a GIT microsoft/git-base style configuration
configuration = GITConfig()

# Initializing a model (with random weights) from the microsoft/git-base style configuration
model = GITModel(configuration)

import torch

pixel_values = torch.randn(1, 3, 224, 224)
input_ids = torch.tensor([[101]])

outputs = model(input_ids, pixel_values=pixel_values)

for k,v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k,len(v))
