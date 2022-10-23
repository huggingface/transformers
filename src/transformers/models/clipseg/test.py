from transformers import CLIPSegConfig, CLIPSegForImageSegmentation


model = CLIPSegForImageSegmentation(CLIPSegConfig())

for name, param in model.named_parameters():
    print(name, param.shape)
