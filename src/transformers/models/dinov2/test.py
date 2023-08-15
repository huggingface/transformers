import torch

from transformers import Dinov2Backbone


model = Dinov2Backbone.from_pretrained(
    "facebook/dinov2-small", out_features=["stage12"], reshape=False, apply_layernorm=False
)

outputs = model(torch.randn(1, 3, 224, 224), output_hidden_states=True)

for i in outputs.feature_maps:
    print(i[0, :3, :3])

print(outputs.feature_maps[-1][0, :3, :3])
print(outputs.hidden_states[-1][0, :3, :3])

assert torch.allclose(outputs.feature_maps[-1], outputs.hidden_states[-1])
