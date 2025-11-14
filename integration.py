from transformers import AutoImageProcessor, AutoModelForImageClassification
model_name = "dimidagd/dinov3-vit7b16-pretrain-lvd1689m-imagenet1k-lc"
processor_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
image_processor = AutoImageProcessor.from_pretrained(processor_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
from PIL import Image
image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")

inputs = image_processor(image, return_tensors="pt")
import torch
with torch.no_grad():
    outputs = model(**inputs)
breakpoint()