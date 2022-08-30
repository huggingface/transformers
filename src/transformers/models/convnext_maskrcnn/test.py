import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

import requests
from transformers import ConvNextMaskRCNNForObjectDetection


model = ConvNextMaskRCNNForObjectDetection.from_pretrained("nielsr/convnext-tiny-maskrcnn")

url = "https://miro.medium.com/max/1000/0*w1s81z-Q72obhE_z"
image = Image.open(requests.get(url, stream=True).raw)

# standard PyTorch mean-std input image normalization
transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

pixel_values = transform(image)
pixel_values = torch.stack([pixel_values, pixel_values], dim=0)

width, height = image.size
pixel_values_height, pixel_values_width = pixel_values.shape[-2:]
width_scale = pixel_values_width / width
height_scale = pixel_values_height / height

img_metas = [
    dict(
        img_shape=tuple(pixel_values.shape[2:]) + (3,),
        scale_factor=np.array([width_scale, height_scale, width_scale, height_scale], dtype=np.float32),
        ori_shape=(height, width, 3),
    ),
    dict(
        img_shape=tuple(pixel_values.shape[2:]) + (3,),
        scale_factor=np.array([width_scale, height_scale, width_scale, height_scale], dtype=np.float32),
        ori_shape=(height, width, 3),
    ),
]

# forward pass
with torch.no_grad():
    outputs = model(pixel_values, img_metas=img_metas)
    bbox_results = outputs.results[0][0]

detections = []
for label in range(len(bbox_results)):
    if len(bbox_results[label]) > 0:
        for detection in bbox_results[label]:
            detections.append((label, detection))

print("Number of detections:", len(detections))
