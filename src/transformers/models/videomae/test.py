import torch

from transformers import VideoMAEConfig, VideoMAEModel


model = VideoMAEModel(VideoMAEConfig())

pixel_values = torch.randn(1, 3, 16, 224, 224)

outputs = model(pixel_values)

print(outputs.keys())
