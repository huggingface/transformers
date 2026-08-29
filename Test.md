PYTHONPATH=src 

<<python>>
import torch
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=100,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    attn_implementation="eager",
)

model = LlamaForCausalLM(config).eval()
input_ids = torch.randint(0, config.vocab_size, (1, 16))

with torch.no_grad():
    out = model(input_ids=input_ids)

print(out.logits.shape)
print("ok")
<< python >>