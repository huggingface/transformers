import io

import requests
from PIL import Image

from transformers import AutoImageProcessor, RFDetrBackbone, RFDetrConfig


images = ["https://media.roboflow.com/notebooks/examples/dog-2.jpeg"]

images = [Image.open(io.BytesIO(requests.get(url).content)) for url in images]

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
inputs = processor(images, return_tensors="pt")

config = RFDetrConfig()
backbone = RFDetrBackbone(config=config.backbone_config)
# model = RFDetrForObjectDetection.from_config()
