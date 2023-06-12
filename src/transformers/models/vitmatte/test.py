import requests
from PIL import Image

from transformers import VitMatteImageProcessor


url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_rgb.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

url = "https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_trimap.png?raw=true"
trimap = Image.open(requests.get(url, stream=True).raw).convert("L")

processor = VitMatteImageProcessor()

inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

for k, v in inputs.items():
    print(k, v.shape)
