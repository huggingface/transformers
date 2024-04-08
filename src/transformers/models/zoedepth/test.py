from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ZoeDepthImageProcessor


processor = ZoeDepthImageProcessor(keep_aspect_ratio=True, size={"height": 768, "width": 512})

filepath = hf_hub_download(repo_id="shariqfarooq/ZoeDepth", filename="examples/person_1.jpeg", repo_type="space")
image = Image.open(filepath).convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values

print(pixel_values.shape)
print(pixel_values.mean())
