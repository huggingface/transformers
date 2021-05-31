# from transformers import CanineConfig, CanineModel
# import torch

# config = CanineConfig()
# model = CanineModel(config)

# for name, param in model.named_parameters():
#     print(name, param.shape)

# input_ids = torch.randint(0, 1, (2,2048))
# outputs = model(input_ids)

# #print(outputs.last_hidden_state.shape)

from transformers import CanineTokenizer, CanineModel

tokenizer = CanineTokenizer()

print(tokenizer.model_max_length)

model = CanineModel.from_pretrained("nielsr/canine-s")

text = "hello world"

encoding = tokenizer(text, padding="max_length", return_tensors="pt")
outputs = model(**encoding)
pooled_output = outputs.pooler_output
sequence_output = outputs.last_hidden_state

print("Shape of sequence_output:", sequence_output.shape)