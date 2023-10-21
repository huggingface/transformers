from transformers import Beit3Processor, Beit3Model
from PIL import Image
import requests

processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")
model = Beit3Model.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "This is photo of a cat"

inputs = processor(text=text, images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)