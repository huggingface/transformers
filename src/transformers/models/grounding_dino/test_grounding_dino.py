import requests
import torch
from PIL import Image

from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor


image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")


processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


text_prompt = "a face"
inputs = processor(images=image, text=text_prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}


outputs = model(**inputs)


logits = outputs.logits
boxes = outputs.pred_boxes

print("Logits shape:", logits.shape)
print("Boxes shape:", boxes.shape)
print("Boxes:", boxes)
