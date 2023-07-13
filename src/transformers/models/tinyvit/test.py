from transformers import TinyVitConfig, TinyVitModel

config = TinyVitConfig()

model = TinyVitModel(config)

for name, param in model.named_parameters():
    print(name, param.shape)