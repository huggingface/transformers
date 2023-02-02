import torch

from transformers import Blip2Config, Blip2VisionConfig, Blip2ForConditionalGeneration


vision_config = Blip2VisionConfig(num_hidden_layers=2)
config = Blip2Config(vision_config=vision_config.to_dict())

model = Blip2ForConditionalGeneration(config)

# for name, param in model.named_parameters():
#     print(name, param.shape)

pixel_values = torch.randn(1, 3, 224, 224)
input_ids = torch.tensor([[101]])

outputs = model.generate(pixel_values, input_ids)
