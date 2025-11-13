import torch

# DINOv3
dinov3_vit7b16_lc = torch.hub.load("./dinov3", 'dinov3_vit7b16_lc', source="local", weights="/tmp/lc.pth", backbone_weights="/tmp/backbone.pth")
print(dinov3_vit7b16_lc)