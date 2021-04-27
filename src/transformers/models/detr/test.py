from transformers import DetrConfig, DetrForPanopticSegmentation
import torch

config = DetrConfig(masks=True)
model = DetrForPanopticSegmentation(config)

pixel_values = torch.randn((1,3,224,224))
model(pixel_values)