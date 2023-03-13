import torch

from transformers import UdopConfig, UdopForConditionalGeneration


config = UdopConfig()
model = UdopForConditionalGeneration(config)

# for name, param in model.named_parameters():
#     print(name, param.shape)

# let's test a forward pass
input_ids = torch.tensor([[101, 102]])
seg_data = torch.tensor([[[0, 0, 0, 0], [1, 2, 3, 4]]]).float()
image = torch.randn(1, 3, 224, 224)
decoder_input_ids = torch.tensor([[101]])


print("Shape of seg_data: ", seg_data.shape)

outputs = model(
    input_ids=input_ids,
    seg_data=seg_data,
    image=image,
    decoder_input_ids=decoder_input_ids,
)

print("Outputs:", outputs.keys())
