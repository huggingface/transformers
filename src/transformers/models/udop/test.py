from transformers import UdopConfig, UdopForConditionalGeneration


config = UdopConfig()
model = UdopForConditionalGeneration(config)

for name, param in model.named_parameters():
    print(name, param.shape)
