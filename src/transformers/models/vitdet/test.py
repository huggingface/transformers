import torch

from transformers import VitDetConfig, VitDetModel


config = VitDetConfig()

model = VitDetModel(config)

outputs = model(torch.randn(1, 3, 224, 224), output_hidden_states=True, output_attentions=True)

output = outputs[0]

hidden_states = outputs.hidden_states[0]
attentions = outputs.attentions[0]
hidden_states.retain_grad()
attentions.retain_grad()

print(attentions.shape)

output.flatten()[0].backward(retain_graph=True)

assert hidden_states.grad is not None
# only works when commenting out the code of reshaping attention_probs
assert attentions.grad is not None
