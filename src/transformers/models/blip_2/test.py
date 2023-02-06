import torch

from transformers import Blip2Config, Blip2ForConditionalGeneration, Blip2VisionConfig


vision_config = Blip2VisionConfig(num_hidden_layers=2)
config = Blip2Config(vision_config=vision_config.to_dict())

model = Blip2ForConditionalGeneration(config).eval()

pixel_values = torch.randn(1, 3, 224, 224)
input_ids = torch.tensor([[101]])

dict_outputs = model(pixel_values, input_ids)

print("Dict output keys:", dict_outputs.keys())

tuple_outputs = model(pixel_values, input_ids, return_dict=False)

print("Number of tuple outputs:", len(tuple_outputs))

print(dict_outputs["logits"].shape)
print(tuple_outputs[0].shape)

# logits are OK
assert torch.allclose(dict_outputs["logits"], tuple_outputs[0])

# vision outputs seem OK
for i, value in enumerate(dict_outputs["vision_outputs"].values()):
    assert torch.allclose(value, tuple_outputs[1][i])

print(len(tuple_outputs[1]))
