from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import requests
import torch
from PIL import Image

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw).convert("RGB")
prompt = "What is unusual about this image?"
inputs = processor(images=image, text=prompt,return_tensors="pt").to(device)
print(inputs)

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
model.to(device)
inputs = processor(images=[image,image], text=[prompt,'another prompt'],padding=True,return_tensors="pt").to(device)
outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)

print(outputs)

generated_text = processor.batch_decode(outputs, skip_special_tokens=False)
print(generated_text)
