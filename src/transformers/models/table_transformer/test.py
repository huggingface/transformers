from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import TableTransformerImageProcessor


processor = TableTransformerImageProcessor(max_size=800)

# let's load an example image
file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="image.png")
image = Image.open(file_path).convert("RGB")

# Apply image transformations
inputs = processor(images=image, return_tensors="pt")

print(processor.max_size)

print(inputs.pixel_values.shape)
