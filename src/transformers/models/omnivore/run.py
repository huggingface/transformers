from transformers import OmnivoreFeatureExtractor, OmnivoreConfig, OmnivoreForVisionClassification, LevitForImageClassificationWithTeacher
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
config = OmnivoreConfig()
feature_extractor = OmnivoreFeatureExtractor()
model = OmnivoreForVisionClassification(config)
dmodel = LevitForImageClassificationWithTeacher.from_pretrained('facebook/levit-128')

weights = torch.load("om/swint.bin", map_location="cpu")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", dmodel.config.id2label[predicted_class_idx])
