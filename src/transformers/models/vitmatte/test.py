import requests
from PIL import Image

from transformers import VitMatteImageProcessor


url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_rgb.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_trimap.png?raw=true"
trimap = Image.open(requests.get(url, stream=True).raw)

image = image.resize((33, 58))
trimap = trimap.resize((33, 58)).convert("L")

processor = VitMatteImageProcessor()

pixel_values = processor(images=image, trimaps=trimap, return_tensors="pt").pixel_values

print(pixel_values.shape)
