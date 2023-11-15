from transformers import TomatoProcessor, TomatoForCausalLM, TomatoImageProcessor, TomatoConfig
from PIL import Image
import requests
import json


with open('/p/scratch/ccstdl/transformers_cache/tomato-8b-1111/config.json', 'r') as config_file:
    config_dict = json.load(config_file)

config = TomatoConfig.from_dict(config_dict)

# model_id = "/p/scratch/ccstdl/transformers_cache/tomato-8b"
model_id = "OneJz/tomato"
processor = TomatoProcessor.from_pretrained(model_id)
# model = TomatoForCausalLM.from_pretrained(model_id).to("cpu")
model = TomatoForCausalLM(config=config).to("cpu")

print(processor.tokenizer.__class__.__name__)
print(model)


# prepare inputs for the model
text_prompt_1 = "Generate a coco-style caption.\n"
text_prompt_2 = "Generate a coco-style caption.jiaojfkdakjdaslk\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=[text_prompt_1, text_prompt_2], images=[image, image], return_tensors="pt").to("cpu")
print(inputs)
# autoregressively generate text
generation_output = model.generate(**inputs)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
# assert generation_text == ['A blue bus parked on the side of a road.']
print(generation_text)
