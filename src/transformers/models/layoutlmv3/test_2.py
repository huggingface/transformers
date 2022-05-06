from transformers import LayoutLMv3Model
import torch

model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

# text + image
input_ids = torch.tensor([[1,2,3,4]])
pixel_values = torch.randn(1,3,224,224)

outputs = model(input_ids=input_ids, pixel_values=pixel_values)

print(outputs.keys())

# text only
outputs = model(input_ids)

print(outputs.keys())

# image only
outputs = model(pixel_values=pixel_values)

print(outputs.keys())
