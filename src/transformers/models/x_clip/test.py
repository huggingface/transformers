from transformers import XClipConfig, XClipModel

config = XClipConfig()
model = XClipModel(config)

for name, param in model.named_parameters():
    print(name, param.shape)