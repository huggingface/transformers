import torch
from huggingface_hub import hf_hub_download
import requests
from PIL import Image


from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.models.llava_next.modeling_better_llava_next import BetterLlavaNextForConditionalGeneration


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor.tokenizer.padding_side = "left"

device = "cuda:0"

model = BetterLlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda",
)
# pad_token_id should be 0, because 1 is bos_token_id
model.config.pad_token_id = 0

# ! Chart and cat
cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)


prompts = [
    "[INST] <image>\nWhat is shown in this image? [/INST]",
    "[INST] <image>\nWhat is shown in this image? [/INST]"
]
inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")

output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)

for o in output:
    print(processor.decode(o, skip_special_tokens=True))


# ! Error, generation still different between batch-2 and batch-1 generation

# ! only chart

prompts = [
    "[INST] <image>\nWhat is shown in this image? [/INST]"
]
inputs = processor(prompts, [chart_img], return_tensors='pt', padding=True).to("cuda")

output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)

for o in output:
    print(processor.decode(o, skip_special_tokens=True))




# ! scene
image_file = "https://llava-vl.github.io/static/images/view.jpg"
prompt = f"[INST] <image> Describe what you see. [/INST]"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, [raw_image], return_tensors='pt').to("cuda", torch.bfloat16)

output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
first_response = processor.decode(output[0], skip_special_tokens=True)
print(first_response)

# ! same scene but different question
# 2 batch
prompts = [
    f"[INST] <image> What are the things I should be cautious about when I visit here? [/INST]",
    f"[INST] <image> Describe what you see. [/INST]",
]
inputs = processor(prompts, [raw_image] * 2, return_tensors='pt', padding=True).to("cuda")

output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)

for o in output:
    print(processor.decode(o, skip_special_tokens=True))




