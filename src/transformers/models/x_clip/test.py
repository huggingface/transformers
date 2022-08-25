import torch

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, XClipConfig, XClipModel


config = XClipConfig()
model = XClipModel(config)

file_path = hf_hub_download(
    repo_id="hf-internal-testing/spaghetti-video-8-frames", filename="pixel_values.pt", repo_type="dataset"
)
pixel_values = torch.load(file_path)

pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
print("Shape of pixel values:", pixel_values.shape)

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
input_ids = tokenizer(
    ["playing sports", "eating spaghetti", "go shopping"], padding="max_length", return_tensors="pt"
).input_ids


with torch.no_grad():
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    logits_per_image = outputs.logits_per_image

print("Shape of logits per image:", logits_per_image.shape)
