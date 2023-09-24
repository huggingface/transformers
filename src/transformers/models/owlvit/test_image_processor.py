from transformers import Owlv2ImageProcessor
from PIL import Image
from huggingface_hub import hf_hub_download

processor = Owlv2ImageProcessor()

filepath = hf_hub_download(repo_id="adirik/OWL-ViT", repo_type="space", filename="assets/astronaut.png")
image = Image.open(filepath)
pixel_values = processor(image, return_tensors="pt").pixel_values

print(pixel_values.shape)
print(pixel_values.mean())