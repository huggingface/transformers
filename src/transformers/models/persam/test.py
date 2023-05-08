from transformers import PerSamConfig, PerSamModel

model = PerSamModel(PerSamConfig())

for name, param in model.named_parameters():
    print(name, param.shape)