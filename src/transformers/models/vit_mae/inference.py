from transformers import ViTFeatureExtractor, ViTMAEForPreTraining
import requests
from PIL import Image

feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

# read image from URL
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

encoding = feature_extractor(image, return_tensors="pt")

outputs = model(**encoding)

for k,v in outputs.items():
    print(k, v.shape)

print("Mask:", outputs.mask)
print("Ids_restore:", outputs.ids_restore)