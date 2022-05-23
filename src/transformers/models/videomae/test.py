# import torch
import numpy as np

from transformers import VideoMAEFeatureExtractor


# test feature extractor

feature_extractor = VideoMAEFeatureExtractor()

video = [np.random.rand(512, 640, 3), np.random.rand(312, 200, 3)]

video = np.random.rand(16, 360, 640, 3)
video = [video[i] for i in range(video.shape[0])]

encoding = feature_extractor(video, return_tensors="pt")

print(encoding.pixel_values.shape)
