from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")

# prepare inputs for the model
text_prompt = "Generate a coco-style caption.\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

# autoregressively generate text
generation_output = model.generate(**inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)

print(generation_text)
