from transformers import TomatoProcessor, TomatoForCausalLM, TomatoConfig
from PIL import Image
import requests
import json
import torch


with open('/data/lychen/transformers_cache/tomato-1113/config.json', 'r') as config_file:
    config_dict = json.load(config_file)

config = TomatoConfig.from_dict(config_dict)

model_id = "/data/lychen/transformers_cache/tomato-1113"
# model_id = "OneJz/tomato"
processor = TomatoProcessor.from_pretrained(model_id)
model = TomatoForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cpu")
# model = TomatoForCausalLM(config=config).to("cuda:0")

print(processor.tokenizer.__class__.__name__)
print(model)


def convert(list_of_dicts):# Convert to a dictionary of lists
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists

text_prompt1 = "<|Image|> Generate a coco-style caption. <|Image|> Be reminded that the caption should be longer than 2000 words but shorter than 1 million words. \n"
url1 = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image1 = Image.open(requests.get(url1, stream=True).raw)

text_prompt2 = "What doesn this chart describe?\n"
url2 = "https://huggingface.co/adept/fuyu-8b/resolve/main/chart.png"
image2 = Image.open(requests.get(url2, stream=True).raw)

test_examples = [
    # {"text": "<|Image|> Generate a coco-style caption. <|Image|> Be reminded that the caption should be longer than 2000 words but shorter than 1 million words. \n", "images": image1}, # should assert error
    {"text": text_prompt1, "images": [image1, image2]}, # normal
    # {"text": text_prompt2, "images": [image2 for i in range(5)]}, # should add indicator
    {"text": "<|Image|><|Image|> Generate a coco-style caption. Be reminded that the caption should be longer than 2000 words but shorter than 1 million words. \n", "images": [image1, image2]}, # normal
    {"text": " Generate a coco-style caption. Be reminded that the caption should be longer than 2000 words but shorter than 1 million words. \n<|Image|><|Image|>", "images": [image1, image2]}, # normal
    {"text": " Generate a coco-style caption. Be reminded that the caption should be longer than 2000 words but shorter than 1 million words." * 1000, "images": image1}, # no image, we had error with this case
    {"text": None, "images": [image1]}, # no text
    
]
inputs_to_model = processor(**convert(test_examples), return_tensors="pt", truncation=True)
generation_output = model.generate(**inputs_to_model)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
# assert generation_text == ['A blue bus parked on the side of a road.']
print(generation_text)
