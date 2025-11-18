#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/eric_bezzam/FORKS/transformers-vibevoice/src')

import torch
from transformers.models.vibevoice import VibeVoiceConfig, VibeVoiceForConditionalGeneration

# Create a simple config
config = VibeVoiceConfig(
    text_config={
        "model_type": "qwen2",
        "intermediate_size": 36,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,  # Add this for GQA
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

# Create model
model = VibeVoiceForConditionalGeneration(config)
model.set_attn_implementation('eager')  # Force eager attention for output_attentions
model.eval()

# Create inputs
input_ids = torch.randint(0, 99, (1, 5))
attention_mask = torch.ones(1, 5)

# Test 1: Only attentions
print("=== Test 1: output_attentions=True, output_hidden_states=False ===")
with torch.no_grad():
    outputs1 = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=False)

print(f"Type: {type(outputs1)}")
print(f"Length: {len(outputs1)}")
print(f"Fields: {list(outputs1.keys()) if hasattr(outputs1, 'keys') else 'No keys method'}")

# Test 2: Both attentions and hidden states
print("\n=== Test 2: output_attentions=True, output_hidden_states=True ===")
with torch.no_grad():
    outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)

print(f"Type: {type(outputs2)}")
print(f"Length: {len(outputs2)}")
print(f"Fields: {list(outputs2.keys()) if hasattr(outputs2, 'keys') else 'No keys method'}")

print(f"\nDifference in length: {len(outputs2) - len(outputs1)}")