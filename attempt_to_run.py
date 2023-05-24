from src.transformers.models.performer.t5.modeling_t5_performer import T5Attention
from src.transformers.models.performer.t5.configuration_t5_performer import T5PerformerConfig

config = T5PerformerConfig()
import torch
attention = T5Attention(config)
tensor = torch.randn(size=(4, 512, 512))
output = attention(tensor)
print(output.size())