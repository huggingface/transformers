import requests
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

# image only
inputs = processor(images=image, return_tensors="pt")

model.config.is_encoder_decoder = True
predictions = model.generate(**inputs)
print(processor.decode(predictions[0], skip_special_tokens=True))
