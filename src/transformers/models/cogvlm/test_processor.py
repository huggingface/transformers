from transformers import CogVLMProcessor, LlamaTokenizer, CLIPImageProcessor
from PIL import Image
import requests

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

processor = CogVLMProcessor(image_processor, tokenizer, image_size=224, patch_size=14)

image = Image.open(requests.get("https://i.imgur.com/7JXOe8X.png", stream=True).raw)

text = "how are you?"

batch = processor(text=text, images=image, return_tensors="pt")

for k,v in batch.items():
    print(k,v.shape)
