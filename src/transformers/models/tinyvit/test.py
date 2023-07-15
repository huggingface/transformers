import torch

from transformers import TinyVitConfig, TinyVitModel


config = TinyVitConfig()

model = TinyVitModel(config)

outputs = model(torch.randn(1, 3, 224, 224), output_hidden_states=True)

for i in outputs.hidden_states:
    print(i.shape)
