# uv pip install kornia einops hydra-core opencv-python-headless pillow requests matplotlib
import pathlib

import cv2
import hydra.utils
import numpy as np
import requests
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor

from transformers import AutoImageProcessor, AutoModel
from transformers.models.efficientloftr.original_eloftr import EfficientLoFTR

torch.manual_seed(42)

def read_image(image_path: pathlib.Path) -> np.ndarray:
    return cv2.imread(str(image_path))


def preprocess_image(image: np.ndarray, w: int, h: int, device) -> Tensor:
    image = cv2.resize(image, (w, h)).astype(np.float32)
    image /= 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = torch.from_numpy(image)
    return image.reshape((1, 1, h, w)).to(device)

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
    eloftr_config = OmegaConf.load("original_config.yaml")
    eloftr_weights = "eloftr.pth"
    original_model = hydra.utils.instantiate(eloftr_config)
    original_model.to(device)
    original_model.eval()
    original_outputs = original_model(**pixel_values)
    print(original_outputs)
