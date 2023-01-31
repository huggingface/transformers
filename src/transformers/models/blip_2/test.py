from transformers import Blip2Config, Blip2ForConditionalGeneration


config = Blip2Config()

model = Blip2ForConditionalGeneration(config)

for name, param in model.named_parameters():
    print(name, param.shape)
