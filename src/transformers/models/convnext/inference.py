import torch

from transformers import ConvNextConfig, ConvNextModel


config = ConvNextConfig()
model = ConvNextModel(config)

pixel_values = torch.randn((1, 3, 224, 224))

outputs = model(pixel_values=pixel_values)

# Shape of hidden states: torch.Size([1, 96, 56, 56])
# Shape of hidden states: torch.Size([1, 192, 28, 28])
# Shape of hidden states: torch.Size([1, 384, 14, 14])
# Shape of hidden states: torch.Size([1, 768, 7, 7])

print("Shape of pooled output:", outputs.pooler_output.shape)
