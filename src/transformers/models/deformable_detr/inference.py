import MultiScaleDeformableAttention as MSDA

from transformers import DeformableDetrConfig, DeformableDetrModel
import torch

config = DeformableDetrConfig()

model = DeformableDetrModel(config)

pixel_values = torch.randn((1,3,224,224))

outputs = model(pixel_values)