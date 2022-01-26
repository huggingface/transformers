import torch
from PIL import Image
from torchvision import transforms as T

import requests
from transformers import DeformableDetrConfig, DeformableDetrModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = DeformableDetrConfig()
model = DeformableDetrModel(config).eval()
model.to(device)

transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = transform(image).unsqueeze(0).to(device)

outputs = model(pixel_values)
