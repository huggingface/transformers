from PIL import Image

import requests
from transformers import ViltConfig, ViltModel, ViltProcessor


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

model = ViltModel(ViltConfig(image_size=384, patch_size=32))

# prepare image + text
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "hello world"

# encode
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)

print(outputs.last_hidden_state.shape)
