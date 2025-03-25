# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. Team and AIRAS Inc. Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from ctransformers import AutoModelForCausalLM as GGUFModel
from models.sapnous import SapnousT1Config

def convert_to_gguf(model_path, output_path):
    # Load the model and tokenizer with vision-language support
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16  # Use FP16 for memory efficiency
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Get model configuration
    config = model.config
    if not isinstance(config, SapnousT1Config):
        raise ValueError("Model must be a SapnousT1 model")
    
    # Save in intermediate format
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Convert to GGUF using custom SapnousT1 architecture settings
    gguf_model = GGUFModel.from_pretrained(
        output_path,
        model_type='sapnous_t1',  # Custom architecture type
        gpu_layers=0,  # CPU only for conversion
        config={
            'context_length': config.sliding_window,
            'attention_type': 'multihead',  # Custom attention implementation
            'num_attention_heads': config.num_attention_heads,
            'num_key_value_heads': config.num_key_value_heads,
            'hidden_size': config.hidden_size,
            'intermediate_size': config.intermediate_size,
            'max_position_embeddings': config.max_position_embeddings,
            'vocab_size': config.vocab_size,
            'num_hidden_layers': config.num_hidden_layers,
            'rms_norm_eps': config.rms_norm_eps,
            'rope_theta': config.rope_theta,
            # Vision model parameters
            'vision_config': {
                'hidden_size': config.vision_hidden_size,
                'num_hidden_layers': config.vision_layers,
                'num_attention_heads': config.vision_heads,
                'intermediate_size': config.vision_intermediate_size,
                'patch_size': config.patch_size,
                'image_size': config.image_size
            }
        }
    )
    
    print(f"Model converted and saved to {output_path}")
    return gguf_model

def convert_to_hf(gguf_path, output_path):
    """Convert GGUF model back to Hugging Face format"""
    # Load GGUF model configuration
    config_path = Path(gguf_path) / "config.json"
    with open(config_path, 'r') as f:
        gguf_config = json.load(f)
    
    # Create SapnousT1 configuration
    config = SapnousT1Config(
        vocab_size=gguf_config['vocab_size'],
        hidden_size=gguf_config['hidden_size'],
        num_hidden_layers=gguf_config['num_hidden_layers'],
        num_attention_heads=gguf_config['num_attention_heads'],
        num_key_value_heads=gguf_config['num_key_value_heads'],
        intermediate_size=gguf_config['intermediate_size'],
        max_position_embeddings=gguf_config['max_position_embeddings'],
        rms_norm_eps=gguf_config['rms_norm_eps'],
        rope_theta=gguf_config['rope_theta'],
        # Vision configuration
        vision_hidden_size=gguf_config['vision_config']['hidden_size'],
        vision_layers=gguf_config['vision_config']['num_hidden_layers'],
        vision_heads=gguf_config['vision_config']['num_attention_heads'],
        vision_intermediate_size=gguf_config['vision_config']['intermediate_size'],
        patch_size=gguf_config['vision_config']['patch_size'],
        image_size=gguf_config['vision_config']['image_size']
    )
    
    # Load GGUF model
    gguf_model = GGUFModel.from_pretrained(gguf_path)
    
    # Convert weights to HF format
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(gguf_model.state_dict())
    
    # Save converted model
    model.save_pretrained(output_path)
    print(f"Model converted back to Hugging Face format at {output_path}")
    return model

if __name__ == '__main__':
    model_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(model_path, 'gguf_model')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    convert_to_gguf(model_path, output_path)