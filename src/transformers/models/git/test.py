from PIL import Image

import requests
from transformers import BertTokenizer, CLIPFeatureExtractor, GITModel, GITProcessor


processor = GITProcessor(
    feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
)
model = GITModel.from_pretrained("nielsr/git-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = "this is an image of two cats"

print(processor.tokenizer.model_input_names)

inputs = processor(text, images=image, return_tensors="pt")

for k, v in inputs.items():
    print(k, v.shape)

outputs = model(**inputs)
