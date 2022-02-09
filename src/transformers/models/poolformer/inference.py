from transformers import PoolFormerFeatureExtractor, PoolFormerForImageClassification
import requests 
from PIL import Image

feature_extractor = PoolFormerFeatureExtractor()

model = PoolFormerForImageClassification.from_pretrained("sail/poolformer_s12")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)

predicted_class = outputs.logits.argmax(-1).item()
print(model.config.id2label[predicted_class])

print(outputs.logits[0,:3])