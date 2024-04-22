from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ZoeDepthImageProcessor


filepath = hf_hub_download(repo_id="shariqfarooq/ZoeDepth", filename="examples/person_1.jpeg", repo_type="space")
image = Image.open(filepath).convert("RGB")

image_processor = ZoeDepthImageProcessor()

pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

print(pixel_values.shape)
print(pixel_values.mean())
