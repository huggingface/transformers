import torch

from transformers import Beit3Config, Beit3Model


config = Beit3Config()
model = Beit3Model(config)

pixel_values = torch.randn(1, 3, 224, 224)
input_ids = torch.randint(0, 8192, (1, 224))

dict_outputs = model(pixel_values=pixel_values, input_ids=input_ids, output_attentions=True, output_hidden_states=True)

print(dict_outputs.keys())

tuple_outputs = model(
    pixel_values=pixel_values,
    input_ids=input_ids,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=False,
)

print(len(tuple_outputs))

for i in tuple_outputs:
    if isinstance(i, torch.Tensor):
        print(i.shape)
    elif i is not None:
        print(len(i))
    else:
        print(i)
