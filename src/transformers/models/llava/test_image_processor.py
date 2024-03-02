import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import LlavaImageProcessor


image_processor = LlavaImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


image = load_image()

inputs = image_processor(images=[image, image], return_tensors="pt")


filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_pixel_values.pt", repo_type="dataset")
original_pixel_values = torch.load(filepath, map_location="cpu")

assert torch.allclose(inputs.pixel_values.half(), original_pixel_values)

image_sizes = torch.tensor([[1024, 899]])
assert image_sizes[0].tolist() == inputs.image_sizes[0].tolist()

print(inputs.pixel_values.shape)

concat_images = torch.cat(list(inputs.pixel_values), dim=0)

print(concat_images.shape)
