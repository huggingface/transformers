import torch

from transformers import OmnivoreConfig, OmnivoreForVisionClassification


name = "omnivore_swinT"
inputs = torch.randn(2, 3, 6, 224, 224)
model = torch.hub.load("facebookresearch/omnivore:main", model=name)
logits2 = model(inputs, "video")

print("*****************************")
config = OmnivoreConfig()
model = OmnivoreForVisionClassification(config)
weights = torch.load("om/swint.bin", map_location="cpu")
model.load_state_dict(weights)


outputs = model(inputs, "video")
logits1 = outputs.logits
