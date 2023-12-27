import torch

from transformers import TinyVitBackbone, TinyVitConfig, TinyVitModel


print("Testing backbone...")

config = TinyVitConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

model = TinyVitBackbone(config)

outputs = model(torch.ones(1, 3, 224, 224))

for i in outputs.feature_maps:
    print(i.shape)


print("Testing regular model...")

config = TinyVitConfig()

model = TinyVitModel(config)

outputs = model(torch.ones(1, 3, 224, 224), output_hidden_states=True)

for i in outputs.hidden_states:
    print(i.shape)
