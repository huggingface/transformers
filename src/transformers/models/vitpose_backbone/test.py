import torch

from transformers import ViTPoseBackbone, ViTPoseBackboneConfig


model = ViTPoseBackbone(ViTPoseBackboneConfig())

pixel_values = torch.randn(1, 3, 256, 192)

feature_maps = model(pixel_values)
