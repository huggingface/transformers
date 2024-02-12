import requests
from PIL import Image

from transformers import LlavaImageProcessor


image = Image.open(
    requests.get(
        "https://github.com/haotian-liu/LLaVA/blob/main/images/llava_v1_5_radar.jpg?raw=true", stream=True
    ).raw
)

print(image.size)

processor = LlavaImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

pixel_values = processor(image, return_tensors="pt")["pixel_values"]

print(pixel_values.shape)
print(pixel_values.mean())
