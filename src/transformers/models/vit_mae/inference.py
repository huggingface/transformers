from transformers import ViTMAEConfig, ViTMAEForPreTraining
import torch

model = ViTMAEForPreTraining(ViTMAEConfig())

for name, param in model.named_parameters():
    print(name, param.shape)

pixel_values = torch.randn((1,3,224,224))

outputs = model(pixel_values)
