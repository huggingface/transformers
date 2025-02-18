# uv pip install kornia einops hydra-core opencv-python-headless pillow requests matplotlib
import pathlib

import cv2
import hydra.utils
import numpy as np
import requests
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor

from transformers import AutoImageProcessor, AutoModel
from transformers.models.efficientloftr.compare_versions import plot_pair
from transformers.models.efficientloftr.original_eloftr import EfficientLoFTR

torch.manual_seed(42)
torch.set_printoptions(precision=15)

def prepare_imgs():
    dataset = load_dataset("hf-internal-testing/image-matching-test-dataset", split="train")
    image0 = dataset[0]["image"]
    image1 = dataset[1]["image"]
    image2 = dataset[2]["image"]
    # [image1, image1] on purpose to test the model early stopping
    return [[image2, image0]]

device = "cuda"
url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

# images = prepare_imgs()
images = [image1, image2]

image_processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
pixel_values = image_processor(images, return_tensors="pt").to(device)

print(pixel_values)
print(pixel_values["pixel_values"].shape)

with torch.no_grad():
    eloftr_config = OmegaConf.load("modified_config.yaml")
    eloftr_weights = "eloftr.pth"
    modified_model = hydra.utils.instantiate(eloftr_config)
    modified_model.to(device)
    modified_model.eval()
    modified_outputs = modified_model(**pixel_values)
    print(modified_outputs)

    image_sizes = [[(image.height, image.width) for image in images]]
    outputs = image_processor.post_process_keypoint_matching(modified_outputs, image_sizes)
    print(outputs)
    plot_pair(outputs, image1, image2, "modified.png")