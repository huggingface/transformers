from transformers import pipeline
from PIL import Image    
import requests

model_id = "nielsr/cogvlm-chat-hf"
pipe = pipeline("image-to-text", model=model_id)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"

image = Image.open(requests.get(url, stream=True).raw)
query = "How many cats are there?"
prompt = f"Question: {query} Answer:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)