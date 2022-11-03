import torch

from transformers import SwinConfig, SwinForMaskedImageModeling


state_dict = torch.load(
    "/Users/nielsrogge/Documents/SwinSimMIM/simmim_pretrain__swin_large__img192_window12__800ep.pth",
    map_location="cpu",
)["model"]

for name, param in state_dict.items():
    print(name, param.shape)

config = SwinConfig(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), image_size=192, window_size=6)

model = SwinForMaskedImageModeling(config)

# for name, param in model.named_parameters():
#     print(name, param.shape)
