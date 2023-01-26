from functools import partial

import torch
from torch import nn

from transformers import H3Config, H3ForCausalLM


# test = partial(nn.functional.gelu, approximate="tanh")

model = H3ForCausalLM(H3Config())

for name, param in model.named_parameters():
    print(name, param.shape)

input_ids = torch.tensor([[101]])

outputs = model(input_ids)
