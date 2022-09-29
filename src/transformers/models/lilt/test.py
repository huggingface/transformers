from transformers import LiltConfig, LiltModel


config = LiltConfig()

model = LiltModel(config)

for name, param in model.named_parameters():
    print(name, param.shape)
