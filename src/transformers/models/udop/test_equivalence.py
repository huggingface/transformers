import torch

from transformers import UdopConfig, UdopModel


config = UdopConfig()
model = UdopModel(config)

input_ids = torch.tensor([[101]])
seg_data = torch.tensor([[[1, 2, 3, 4]]]).float()
pixel_values = torch.randn(1, 3, 224, 224)

print("------DICT OUTPUTS-------")
dict_outputs = model(input_ids=input_ids, seg_data=seg_data, images=pixel_values, decoder_input_ids=input_ids)
for k, v in dict_outputs.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k, len(v))

print("-------TUPLE OUTPUTS-------")
tuple_outputs = model(
    input_ids=input_ids, seg_data=seg_data, images=pixel_values, decoder_input_ids=input_ids, return_dict=False
)
print("Length of tuple outputs:", len(tuple_outputs))
for i in tuple_outputs:
    if isinstance(i, torch.Tensor):
        print(i.shape)
    else:
        print(len(i))
