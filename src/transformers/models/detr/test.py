from transformers import DetrConfig, DetrForPanopticSegmentation
import torch

config = DetrConfig(masks=True)
model = DetrForPanopticSegmentation(config)

for name, param in model.named_parameters():
    print(name, param.shape)

# pixel_values = torch.randn([2, 3, 873, 1201])
# pixel_mask = torch.randint(0,1, (2, 873, 1201))
# outputs = model(pixel_values, pixel_mask)