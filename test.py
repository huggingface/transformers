# A simple test. To be removed when done.
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import time
import torch
# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16, use_flash_attention_2=True)

# prepare inputs for the model
text_prompt = "Generate a coco-style caption.\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

# autoregressively generate text
# Test time
start = time.time()
for i in range(10):
    generation_output = model.generate(**inputs, max_new_tokens=7)
end = time.time()
print(end - start)

generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
assert generation_text == ['A blue bus parked on the side of a road.']