import torch

from transformers import LayoutLMv2Processor, LayoutLMv3Model


# from datasets import load_dataset
# from PIL import Image


processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

# ds = load_dataset("hf-internal-testing/fixtures_docvqa", split="test")
# image = Image.open(ds[0]["file"]).convert("RGB")

# encoding = processor(image, return_tensors="pt")

input_ids = torch.tensor([[1, 2]])
bbox = torch.tensor([[0, 0, 1, 2], [1, 5, 2, 5]]).unsqueeze(0)
pixel_values = torch.randn(1, 3, 224, 224)

print("Shape of input_ids:", input_ids.shape)
print("Shape of bbox:", bbox.shape)

outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values)
last_hidden_states = outputs.last_hidden_state
