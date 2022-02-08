from transformers import ViTConfig, ViTForMaskedImageModeling
import torch

model = ViTForMaskedImageModeling(ViTConfig())

pixel_values = torch.randn(1,3,224,224)
mask = torch.ones(1,196).bool()

outputs = model(pixel_values, bool_masked_pos=mask)