import torch

from transformers import TinyVitConfig, TinyVitModel


config = TinyVitConfig()

model = TinyVitModel(config)

outputs = model(torch.randn(1, 3, 224, 224), output_attentions=True)

for i in outputs.attentions:
    print(i.shape)
