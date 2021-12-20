import requests
from PIL import Image
from transformers import ViltFeatureExtractor

feature_extractor = ViltFeatureExtractor()

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

encoding = feature_extractor(image, return_tensors='pt')

for k,v in encoding.items():
    print(k, v.shape, v.dtype)

print((encoding['pixel_mask'] == 0).sum())