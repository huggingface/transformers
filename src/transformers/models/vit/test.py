from transformers import ViTModel
import torch

model = ViTModel.from_pretrained("nielsr/dino_vitb16", add_pooling_layer=False)

pixel_values = torch.randn((1,3,480,480))

outputs = model(pixel_values=pixel_values, output_attentions=True)

