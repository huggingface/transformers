#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/eric_bezzam/FORKS/transformers-vibevoice/src')

import torch
from transformers.models.vibevoice import VibeVoiceConfig, VibeVoiceForConditionalGeneration

# Create a simple config matching test config
config = VibeVoiceConfig(
    text_config={
        "model_type": "qwen2",
        "intermediate_size": 36,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 99,
        "pad_token_id": 0,
        "eos_token_id": 0,
    },
    acoustic_tokenizer_config={
        "model_type": "vibevoice_acoustic_tokenizer", 
        "hidden_size": 16,  
        "n_filters": 4,
        "downsampling_ratios": [2],
        "depths": [1],
    },
    semantic_tokenizer_config={
        "model_type": "vibevoice_semantic_tokenizer",
        "hidden_size": 32, 
        "n_filters": 4,
        "downsampling_ratios": [2],
        "depths": [1],
    },
    diffusion_head_config={
        "model_type": "vibevoice_diffusion_head",
        "hidden_size": 32,
        "latent_size": 16,
    },
    speech_start_id=3,
    speech_end_id=4,
    speech_diffusion_id=5,
)

# Create model and force eager attention
model = VibeVoiceForConditionalGeneration(config)
model.config._attn_implementation = 'eager'
model.eval()

# Create inputs
input_ids = torch.randint(0, 99, (1, 5))
attention_mask = torch.ones(1, 5)

# Test 1: Only attentions (should NOT include hidden_states)
print("=== Test 1: output_attentions=True, output_hidden_states=False ===")
with torch.no_grad():
    outputs1 = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=False)

print(f"Length: {len(outputs1)}")
print(f"Fields: {list(outputs1.keys())}")
print(f"loss: {outputs1.loss}")
print(f"diffusion_loss: {outputs1.diffusion_loss}")
print(f"logits: {outputs1.logits is not None}")
print(f"past_key_values: {outputs1.past_key_values is not None}")
print(f"hidden_states: {outputs1.hidden_states is not None}")
print(f"attentions: {outputs1.attentions is not None}")

# Test 2: Both attentions and hidden states
print("\n=== Test 2: output_attentions=True, output_hidden_states=True ===")
with torch.no_grad():
    outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)

print(f"Length: {len(outputs2)}")
print(f"Fields: {list(outputs2.keys())}")
print(f"loss: {outputs2.loss}")
print(f"diffusion_loss: {outputs2.diffusion_loss}")
print(f"logits: {outputs2.logits is not None}")
print(f"past_key_values: {outputs2.past_key_values is not None}")
print(f"hidden_states: {outputs2.hidden_states is not None}")
print(f"attentions: {outputs2.attentions is not None}")

print(f"\nDifference in length: {len(outputs2) - len(outputs1)}")

# Test 3: Neither attentions nor hidden states  
print("\n=== Test 3: output_attentions=False, output_hidden_states=False ===")
with torch.no_grad():
    outputs3 = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False)

print(f"Length: {len(outputs3)}")
print(f"Fields: {list(outputs3.keys())}")
print(f"hidden_states: {outputs3.hidden_states is not None}")
print(f"attentions: {outputs3.attentions is not None}")