#!/usr/bin/env python3
"""Reproduce issue #44242: Load balancing loss not added when output_router_logits=False"""

from transformers import MixtralConfig, MixtralForCausalLM
import torch

print("Reproducing issue #44242...")

# 1. Configure the model to NOT output router logits
config = MixtralConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=2,  # small for demonstration
    num_local_experts=8,
    output_router_logits=False,  # This is the issue - no router logits output
    router_aux_loss_coef=0.001   # But we want aux loss computed!
)

# 2. Initialize the model
model = MixtralForCausalLM(config)

# 3. Create dummy inputs and labels for a training step
input_ids = torch.tensor([[1, 254, 99, 32]])
labels = torch.tensor([[1, 254, 99, 32]]) # Next-token prediction labels

# 4. Perform the forward pass
outputs = model(input_ids=input_ids, labels=labels)

# 5. Read the losses
total_loss = outputs.loss
aux_loss = outputs.aux_loss 
router_logits = outputs.router_logits # Tuple of router logits for each layer

print(f"Config router_aux_loss_coef: {config.router_aux_loss_coef}")
print(f"Config output_router_logits: {config.output_router_logits}")
print(f"Total Loss: {total_loss.item() if total_loss is not None else None:.4f}")
print(f"Auxiliary Load Balancing Loss: {aux_loss.item() if aux_loss is not None else None}")
print(f"Router logits: {router_logits}")

print()
print("According to the issue, aux_loss should be computed when router_aux_loss_coef != 0")
print("regardless of output_router_logits setting, but currently it's only computed")
print("when output_router_logits=True")