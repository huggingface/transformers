#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/eric_bezzam/FORKS/transformers-vibevoice/src')

from transformers.models.vibevoice import VibeVoiceForConditionalGeneration

model_path = "bezzam/VibeVoice-1.5B"

# Load model
model = VibeVoiceForConditionalGeneration.from_pretrained(model_path)

print("Model's generation_config attributes:")
print(f"output_hidden_states: {getattr(model.generation_config, 'output_hidden_states', 'NOT SET')}")
print(f"return_dict_in_generate: {getattr(model.generation_config, 'return_dict_in_generate', 'NOT SET')}")
print(f"cfg_scale: {getattr(model.generation_config, 'cfg_scale', 'NOT SET')}")
print(f"ddpm_inference_steps: {getattr(model.generation_config, 'ddmp_inference_steps', 'NOT SET')}")

print("\nFull generation_config:")
print(model.generation_config)