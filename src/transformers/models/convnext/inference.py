import torch

from transformers import ConvNextConfig, ConvNextModel


config = ConvNextConfig()
model = ConvNextModel(config)

pixel_values = torch.randn((1, 3, 224, 224))

outputs = model(pixel_values=pixel_values)

for name, param in model.named_parameters():
    print(name, param.shape)

# Shape of hidden states: torch.Size([1, 96, 56, 56]) / 4
# Shape of hidden states: torch.Size([1, 192, 28, 28]) / 8
# Shape of hidden states: torch.Size([1, 384, 14, 14]) / 16
# Shape of hidden states: torch.Size([1, 768, 7, 7]) / 32

print("Shape of outputs:", outputs.last_hidden_state.shape)
