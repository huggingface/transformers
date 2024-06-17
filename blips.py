# from PIL import Image
# import requests
# from transformers import AutoProcessor, BlipForQuestionAnswering
# 
# model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
# processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
# 
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# 
# # training
# text = "How many cats are in the picture?"
# label = "2"
# inputs = processor(images=image, text=text, return_tensors="pt")
# labels = processor(text=label, return_tensors="pt").input_ids
# 
# inputs["labels"] = labels
# outputs = model(**inputs)
# loss = outputs.loss
# loss.backward()
# 
# # inference
# text = "How many cats are in the picture?"
# inputs = processor(images=image, text=text, return_tensors="pt")
# outputs = model.generate(**inputs)
# print(processor.decode(outputs[0], skip_special_tokens=True))



from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map="cuda", torch_dtype=torch.float16
)

processor.num_query_tokens = model.config.num_query_tokens
model.resize_token_embeddings(processor.tokenizer.vocab_size, pad_to_multiple_of=64) # pad for efficient computation
model.config.image_token_index = processor.tokenizer.vocab_size

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device=model.device, dtype=torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=15)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

