from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.bfloat16
)
model.to(device)
url = "https://user-images.githubusercontent.com/50018861/252267123-a49ec5be-d964-4760-9ef5-3f006a353720.png"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: what is the structure and geometry of this chair?"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.bfloat16)

generated_ids = model.generate(**inputs, 
                        num_beams=5,
                        max_length=30,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        temperature=1,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)