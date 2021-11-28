from transformers import ViltFeatureExtractor, BertTokenizer, ViltProcessor
from PIL import Image
import requests

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
feature_extractor = ViltFeatureExtractor(size=384)

processor = ViltProcessor(feature_extractor, tokenizer)

#test processor with image and text
text = "hello world"
image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

encoding = processor(image, text, return_tensors="pt")

for k,v in encoding.items():
    print(k, v.shape)
