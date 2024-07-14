# Will be removed before merging

import requests
from PIL import Image

from transformers import AutoImageProcessor, Dinov2Model, FlaxDinov2Model


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = FlaxDinov2Model.from_pretrained("facebook/dinov2-base", from_pt=True)
original_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
inputs = image_processor(images=image, return_tensors="np")
org_inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
org_op = original_model(**org_inputs)
last_hidden_states = outputs.last_hidden_state[0, :3, :3]
org_last_hidden_states = org_op.last_hidden_state[0, :3, :3]
print(last_hidden_states, last_hidden_states.shape, last_hidden_states.mean())
print(org_last_hidden_states, org_last_hidden_states.shape, org_last_hidden_states.mean())
