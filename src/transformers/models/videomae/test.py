# import torch
import numpy as np

# from transformers import VideoMAEConfig, VideoMAEForPreTraining
from transformers import VideoMAEFeatureExtractor


# model = VideoMAEForPreTraining(VideoMAEConfig())

# pixel_values = torch.randn(1, 3, 16, 224, 224)
# bool_masked_pos = torch.randint(0, 1, (1, 3 * 16 * 224 * 224)).bool()

# outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

# print(outputs.keys())

feature_extractor = VideoMAEFeatureExtractor()

video = [np.random.rand(512, 640, 3), np.random.rand(312, 200, 3)]

video = np.random.rand(16, 360, 640, 3)
video = [video[i] for i in range(video.shape[0])]

encoding = feature_extractor(video, return_tensors="pt")

print(encoding.pixel_values.shape)
