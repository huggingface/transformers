import numpy as np
import requests
import torch
from PIL import Image

from transformers import Beit3ForCaptioning, Beit3Processor


url = "https://datasets-server.huggingface.co/assets/HuggingFaceM4/VQAv2/--/default/train/8/image/image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = Beit3ForCaptioning.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning")

processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning")
inputs = processor(text=["This is photo of a dog"], images=image, return_tensors="pt")

language_masked_pos = torch.zeros_like(inputs.input_ids)
language_masked_pos[:, 6] = 1
inputs.input_ids[:, 6] = 64001

output = model(
    input_ids=inputs.input_ids,
    pixel_values=inputs.pixel_values,
    attention_mask=torch.zeros_like(language_masked_pos),
    language_masked_pos=language_masked_pos,
)

print(processor.tokenizer.decode([np.argmax(output.logits.cpu().detach().numpy())]))
