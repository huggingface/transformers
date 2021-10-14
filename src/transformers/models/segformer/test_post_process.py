import torch

from transformers import SegformerConfig, SegformerFeatureExtractor, SegformerForImageSegmentation


config = SegformerConfig()
model = SegformerForImageSegmentation(config)
feature_extractor = SegformerFeatureExtractor()

pixel_values = torch.randn((1, 3, 512, 512))

# forward pass
outputs = model(pixel_values)

# post process
segmentation_maps = feature_extractor.post_process_semantic(outputs, target_sizes=torch.tensor([[512, 512]]))

print("Segmentation maps:", segmentation_maps[0].shape)
