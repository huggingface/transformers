

import requests
from PIL import Image
import torch
from transformers import ConFiDeNetForDepthEstimation, ConFiDeNetImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open("/data2/onkar/Depth-Anything-V2/ml-depth-pro/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151610412404.jpg").convert("RGB")
print(image.size)
# image.save("image.jpg")

image_processor = ConFiDeNetImageProcessor.from_pretrained("/data2/onkar/Depth-Anything-V2/temp/DepthPro-hf")
model = ConFiDeNetForDepthEstimation.from_pretrained("/data2/onkar/Depth-Anything-V2/temp/DepthPro-hf").to(device)

inputs = image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)],
)

depth = post_processed_output[0]["predicted_depth_uint16"].detach().cpu().numpy()
depth = Image.fromarray(depth, mode="I;16")
depth.save("depth.png")
