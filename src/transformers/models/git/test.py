from PIL import Image
import requests
from transformers import GITProcessor

processor = GITProcessor.from_pretrained("nielsr/git-base")

print(processor.model_input_names)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

for key, value in inputs.items():
   print(key, value.shape)
