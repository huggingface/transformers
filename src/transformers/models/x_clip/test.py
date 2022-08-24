from transformers import AutoTokenizer, XClipConfig, XClipModel
from huggingface_hub import hf_hub_download
import torch

config = XClipConfig()
model = XClipModel(config)

file_path = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video-8-frames", filename="pixel_values.pt", repo_type="dataset"
)
pixel_values = torch.load(file_path)

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
input_ids = tokenizer(
    ["playing sports", "eating spaghetti", "go shopping"], padding="max_length", return_tensors="pt"
).input_ids


with torch.no_grad():
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)