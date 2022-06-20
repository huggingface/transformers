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
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
