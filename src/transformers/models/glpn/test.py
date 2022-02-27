from transformers import GLPNConfig, GLPNForDepthEstimation

model = GLPNForDepthEstimation(GLPNConfig())

for name, param in model.named_parameters():
    print(name, param.shape)