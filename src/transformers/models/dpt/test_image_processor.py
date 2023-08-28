import requests
from PIL import Image

from transformers import DPTImageProcessor


size = {"height": 512, "width": 512}

processor = DPTImageProcessor(size=size, keep_aspect_ratio=True, ensure_multiple_of=32)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(image, return_tensors="pt").pixel_values

print(pixel_values.shape)
