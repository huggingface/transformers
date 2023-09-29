from transformers import SiglipConfig, SiglipModel

config = SiglipConfig()

model = SiglipModel(config)

for name, param in model.named_parameters():
    print(name, param.shape)