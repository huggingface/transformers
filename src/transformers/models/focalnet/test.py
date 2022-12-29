import torch

from transformers import FocalNetConfig, FocalNetModel


# Initializing a FocalNet microsoft/focalnet-tiny style configuration
configuration = FocalNetConfig()

# Initializing a model (with random weights) from the microsoft/focalnet-tiny style configuration
model = FocalNetModel(configuration)

outputs = model(torch.randn(1, 3, 224, 224))

print(outputs.keys())
