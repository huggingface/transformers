import torch
from torch.utils.cpp_extension import CUDA_HOME

from transformers import DeformableDetrConfig, DeformableDetrModel


print(torch.cuda.is_available(), CUDA_HOME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = DeformableDetrConfig()
model = DeformableDetrModel(config)
model.to(device)

pixel_values = torch.randn((1, 3, 224, 224)).to(device)

outputs = model(pixel_values)
