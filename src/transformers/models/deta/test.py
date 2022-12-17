from transformers import DetaConfig, DetaForObjectDetection

model = DetaForObjectDetection(DetaConfig())

for name, param in model.named_parameters():
    print(name, param.shape)