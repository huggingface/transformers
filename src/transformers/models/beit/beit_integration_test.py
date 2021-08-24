import torch
from PIL import Image
from torchvision import transforms

import requests
from transformers import BeitForMaskedImageModeling


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# prepare for model (simply resize + normalize)
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
pixel_values = transform(image).unsqueeze(0)

# prepare bool_masked_pos
bool_masked_pos = torch.ones((1, 196), dtype=torch.bool)
bool_masked_pos[0, 2] = 0
bool_masked_pos[0, 44] = 0

model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

# forward pass
outputs = model(pixel_values, bool_masked_pos)
logits = outputs.logits

print("Shape of logits:", logits.shape)
print("First few logits:", logits[bool_masked_pos][:3, :3])

print("Sum of logits:", logits[bool_masked_pos].sum())
