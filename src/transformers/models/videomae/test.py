import torch

from transformers import VideoMAEConfig, VideoMAEForPreTraining


model = VideoMAEForPreTraining(VideoMAEConfig())

pixel_values = torch.randn(1, 3, 16, 224, 224)
bool_masked_pos = torch.randint(0, 1, (1, 3 * 16 * 224 * 224)).bool()

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

print(outputs.keys())
