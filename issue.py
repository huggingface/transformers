from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from datasets import load_dataset
import torch


processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", torch_dtype=torch.float16, device_map="cuda:0") 
dataset = load_dataset("lmms-lab/docvqa", 'DocVQA', split="test")

d = dataset[2482]
question = d['question']
image = d['image']
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0", torch.float16)
with torch.no_grad():
    outputs = model(**inputs)
