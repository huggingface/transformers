# uv pip install kornia einops hydra-core opencv-python-headless pillow requests matplotlib
import pathlib

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from torch import Tensor

from transformers import AutoImageProcessor
from transformers.models.efficientloftr.configuration_efficientloftr import EfficientLoFTRConfig
from transformers.models.efficientloftr.modeling_efficientloftr import EfficientLoFTRForKeypointMatching


torch.manual_seed(42)


device = "cuda"
url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

image_processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
pixel_values = image_processor(images, return_tensors="pt").to(device)

print(pixel_values)
print(pixel_values["pixel_values"].shape)

with torch.no_grad():
    eloftr_config = EfficientLoFTRConfig()
    model = EfficientLoFTRForKeypointMatching(eloftr_config)
    model.to(device)
    model.eval()
    outputs = model(**pixel_values)
    print(outputs)
