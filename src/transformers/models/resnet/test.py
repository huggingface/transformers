from transformers import AutoImageProcessor, AutoBackbone
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoBackbone.from_pretrained("microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"])

inputs = processor(image, return_tensors="pt")

outputs = model(**inputs)

print(outputs.keys())

for k,v in zip(outputs.stage_names, outputs.hidden_states):
    print(k, v.shape)