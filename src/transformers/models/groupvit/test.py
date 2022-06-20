from PIL import Image

import requests
from transformers import AutoProcessor, GroupViTConfig, GroupViTModel


processor = AutoProcessor.from_pretrained("nvidia/groupvit-gccyfcc")
model = GroupViTModel(GroupViTConfig()).eval()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

for k, v in inputs.items():
    print(k, v.shape)

outputs = model(**inputs)

for name, param in model.named_parameters():
    print(name, param.shape)
