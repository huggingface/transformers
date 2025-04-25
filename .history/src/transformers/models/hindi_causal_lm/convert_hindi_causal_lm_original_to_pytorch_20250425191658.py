# src/transformers/models/hindi_causal_lm/convert_hindi_causal_lm_original_to_pytorch.py
import argparse
import json
import os
import torch
from pathlib import Path

from transformers import HindiCausalLMConfig, HindiCausalLMHeadModel


def convert_hindi_causal_lm_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Convert a HindiCausalLM checkpoint to the HuggingFace format.
    """
    # Load the config from the checkpoint or a provided config file
    if config_path is None:
        config_path = os.path.join(checkpoint_path, "config.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # Create the HF model config
    config = HindiCausalLMConfig(**config_data)
    
    # Create the model
    model = HindiCausalLMHeadModel(config)
    
    # Load the checkpoint
    if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(os.path.join(checkpoint_path, "model.safetensors"))
    elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
    else:
        raise ValueError(f"No model weights found in {checkpoint_path}")
    
    # Map the state dict keys to match the HF model
    # Note: This will depend on the exact architecture of your model
    # and will need to be adjusted based on the original model structure
    new_state_dict = {}
    
    # Map embedding weights
    if "token_embeddings.weight" in state_dict:
        new_state_dict["transformer.word_embeddings.weight"] = state_dict["token_embeddings.weight"]
    
    if "position_embeddings.weight" in state_dict:
        new_state_dict["transformer.position_embeddings.weight"] = state_dict["position_embeddings.weight"]
    
    # Map transformer layers
    for i in range(config.num_hidden_layers):
        # Attention layer
        if f"layers.{i}.attention.query.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.attention.query.weight"] = state_dict[f"layers.{i}.attention.query.weight"]
        if f"layers.{i}.attention.key.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.attention.key.weight"] = state_dict[f"layers.{i}.attention.key.weight"]
        if f"layers.{i}.attention.value.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.attention.value.weight"] = state_dict[f"layers.{i}.attention.value.weight"]
        if f"layers.{i}.attention.output.0.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.attention.output.0.weight"] = state_dict[f"layers.{i}.attention.output.0.weight"]
        
        # Layer norms
        if f"layers.{i}.attention_layernorm.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.attention_layernorm.weight"] = state_dict[f"layers.{i}.attention_layernorm.weight"]
        if f"layers.{i}.attention_layernorm.bias" in state_dict:
            new_state_dict[f"transformer.layers.{i}.attention_layernorm.bias"] = state_dict[f"layers.{i}.attention_layernorm.bias"]
        if f"layers.{i}.ffn_layernorm.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.ffn_layernorm.weight"] = state_dict[f"layers.{i}.ffn_layernorm.weight"]
        if f"layers.{i}.ffn_layernorm.bias" in state_dict:
            new_state_dict[f"transformer.layers.{i}.ffn_layernorm.bias"] = state_dict[f"layers.{i}.ffn_layernorm.bias"]
        
        # FFN
        if f"layers.{i}.ffn.0.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.ffn.0.weight"] = state_dict[f"layers.{i}.ffn.0.weight"]
        if f"layers.{i}.ffn.0.bias" in state_dict:
            new_state_dict[f"transformer.layers.{i}.ffn.0.bias"] = state_dict[f"layers.{i}.ffn.0.bias"]
        if f"layers.{i}.ffn.2.weight" in state_dict:
            new_state_dict[f"transformer.layers.{i}.ffn.2.weight"] = state_dict[f"layers.{i}.ffn.2.weight"]
        if f"layers.{i}.ffn.2.bias" in state_dict:
            new_state_dict[f"transformer.layers.{i}.ffn.2.bias"] = state_dict[f"layers.{i}.ffn.2.bias"]
    
    # Final layer norm
    if "final_layernorm.weight" in state_dict:
        new_state_dict["transformer.final_layernorm.weight"] = state_dict["final_layernorm.weight"]
    if "final_layernorm.bias" in state_dict:
        new_state_dict["transformer.final_layernorm.bias"] = state_dict["final_layernorm.bias"]
    
    # LM head
    if "lm_head.weight" in state_dict:
        new_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    
    # Verify all keys are used
    missing_keys = set(state_dict.keys()) - set([key.replace("transformer.", "") for key in new_state_dict.keys()])
    if missing_keys:
        print(f"Warning: Some keys in original state dict were not used: {missing_keys}")
    
    # Load the mapped state dict into the model
    model.load_state_dict(new_state_dict, strict=False)
    
    # Save the model and tokenizer to the specified path
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    
    # Also copy the tokenizer model file
    tokenizer_path = os.path.join(checkpoint_path, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        import shutil
        shutil.copy(tokenizer_path, os.path.join(pytorch_dump_folder_path, "tokenizer.model"))
    
    print(f"Model converted and saved to {pytorch_dump_folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original model checkpoint")
    parser.add_argument("--pytorch_dump_folder_path", type=str, required=True, help="Path to dump the HF model")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the config file (optional)")
    args = parser.parse_args()
    
    convert_hindi_causal_lm_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)