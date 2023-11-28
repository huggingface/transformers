import torch

from transformers import Swinv2Backbone, Swinv2Config


config = Swinv2Config(out_features=["stage2", "stage3", "stage4"])

model = Swinv2Backbone(config)

pixel_values = torch.rand(1, 3, 224, 224)

print("DICT OUTPUTS:")

dict_outputs = model(pixel_values, output_hidden_states=True, output_attentions=True)

print(dict_outputs.keys())

print(len(dict_outputs[0]))

for i in dict_outputs[0]:
    print(i.shape)

print("TUPLE OUTPUTS:")

tuple_outputs = model(pixel_values, output_hidden_states=True, output_attentions=True, return_dict=False)

print(len(tuple_outputs))

print(len(tuple_outputs[0]))

for i in tuple_outputs[0]:
    print(i.shape)
