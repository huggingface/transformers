import io

import requests
from PIL import Image

from transformers import pipeline


pipe = pipeline(task="image-to-text", model="adept/fuyu-8b")

url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
image = Image.open(io.BytesIO(requests.get(url).content))
prompt = "Generate a coco-style caption.\n"

results = pipe(image, prompt=prompt)
