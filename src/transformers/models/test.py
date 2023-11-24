import requests
from PIL import Image

from transformers import CLIPImageProcessor


size = {"height": 224, "width": 224}
processor = CLIPImageProcessor(size=size)

image = Image.open(
    requests.get("https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true", stream=True).raw
).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values

print(pixel_values.shape)
