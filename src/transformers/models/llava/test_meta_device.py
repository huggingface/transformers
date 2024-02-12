import torch

from transformers import BertConfig, BertModel


config = BertConfig()

with torch.device("meta"):
    model = BertModel(config)

pretrained_model = BertModel.from_pretrained("bert-base-uncased")
original_state_dict =  {k:v.to("cuda") for k,v in pretrained_model.state_dict().items()}
model.load_state_dict(original_state_dict, assign=True)

device = "cuda:0"
model.to(device)
