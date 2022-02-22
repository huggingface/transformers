import torch

from transformers import DPTConfig, DPTModel


model = DPTModel(DPTConfig())

pixel_values = torch.randn((1, 3, 224, 224))

outputs = model(pixel_values)
