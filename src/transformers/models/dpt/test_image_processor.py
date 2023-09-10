from transformers import DPTImageProcessor
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image
import requests

from torchvision import transforms


processor = DPTImageProcessor(
        do_resize=False,
        do_rescale=False,
        # do_pad=True,
        # pad_multiple_of=14,
        do_normalize=True,
        image_mean=(123.675, 116.28, 103.53),
        image_std=(58.395, 57.12, 57.375),
    )

url = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(image, return_tensors="pt").pixel_values

print(pixel_values.shape)
print(pixel_values.mean())

def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

transform = make_depth_transform()
original_pixel_values = transform(image).unsqueeze(0)

print(original_pixel_values.shape)
print(original_pixel_values.mean())