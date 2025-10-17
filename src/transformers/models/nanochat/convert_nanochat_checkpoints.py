# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import gc
import os
from pathlib import Path

import torch

from transformers import AutoTokenizer, NanoChatConfig, NanoChatForCausalLM


def infer_kv_heads(config: NanoChatConfig, state_dict: dict[str, torch.Tensor]) -> int:
    key_weight = state_dict.get("transformer.h.0.attn.c_k.weight")
    if key_weight is None:
        return config.num_key_value_heads
    rows = key_weight.shape[0]
    head_dim = config.hidden_size // config.num_attention_heads
    if rows % head_dim != 0:
        return config.num_key_value_heads
    inferred = rows // head_dim
    print(f"Inferred {inferred} key_value heads from checkpoint")
    return max(inferred, 1)


def convert_layer(old_prefix: str, new_prefix: str) -> dict[str, str]:
    return {
        f"{old_prefix}.attn.c_q.weight": f"{new_prefix}.self_attn.q_proj.weight",
        f"{old_prefix}.attn.c_k.weight": f"{new_prefix}.self_attn.k_proj.weight",
        f"{old_prefix}.attn.c_v.weight": f"{new_prefix}.self_attn.v_proj.weight",
        f"{old_prefix}.attn.c_proj.weight": f"{new_prefix}.self_attn.o_proj.weight",
        f"{old_prefix}.mlp.c_fc.weight": f"{new_prefix}.mlp.fc.weight",
        f"{old_prefix}.mlp.c_proj.weight": f"{new_prefix}.mlp.proj.weight",
    }


def write_model(input_dir, output_dir, safe_serialization=True):
    """Convert NanoChat model from original checkpoint format to HuggingFace format."""
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    
    # Load config
    config = NanoChatConfig.from_pretrained(input_path)
    print(f"Loaded config hidden_size={config.hidden_size} num_layers={config.num_hidden_layers}")

    # Load checkpoint
    print(f"Fetching all parameters from the checkpoint at {input_path}...")
    old_state = torch.load(input_path / "pytorch_model.bin", map_location="cpu")
    
    # Original nanochat weights are in bfloat16
    for key in old_state:
        if old_state[key].dtype == torch.float32:
            old_state[key] = old_state[key].to(torch.bfloat16)
    
    # Infer key-value heads from checkpoint
    inferred_kv = infer_kv_heads(config, old_state)
    config.num_key_value_heads = inferred_kv
    if config.num_attention_heads % config.num_key_value_heads != 0:
        print(
            f"Adjusting num_attention_heads from {config.num_attention_heads} to {config.num_key_value_heads}"
        )
        config.num_attention_heads = config.num_key_value_heads

    print("Converting model...")
    state_dict = {}
    rename_map = {}

    def assign(old_key: str, new_key: str) -> None:
        tensor = old_state.get(old_key)
        if tensor is None:
            return
        state_dict[new_key] = tensor.clone()
        rename_map[old_key] = new_key

    # Convert embeddings and head
    assign("transformer.wte.weight", "model.embed_tokens.weight")
    assign("lm_head.weight", "lm_head.weight")

    # Convert layers
    for layer_idx in range(config.num_hidden_layers):
        old_prefix = f"transformer.h.{layer_idx}"
        new_prefix = f"model.layers.{layer_idx}"
        mapping = convert_layer(old_prefix, new_prefix)
        for old_key, new_key in mapping.items():
            assign(old_key, new_key)

    missing = [key for key in old_state.keys() if key not in rename_map]
    if missing:
        print(f"Skipped {len(missing)} legacy entries that have no equivalent in the shared implementation")

    del old_state
    gc.collect()

    # Update config
    config.torch_dtype = torch.bfloat16
    config.tie_word_embeddings = False
    
    # Load the checkpoint into the model
    print("Loading the checkpoint in a NanoChat model.")
    with torch.device("meta"):
        model = NanoChatForCausalLM(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    
    if hasattr(model.config, "_name_or_path"):
        del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    del state_dict, model
    
    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    NanoChatForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")


def write_tokenizer(input_dir, output_dir):
    """Convert and save the tokenizer."""
    input_path = Path(input_dir)
    
    # Convert the pickle tokenizer to HF format
    tokenizer_pkl = input_path / "tokenizer.pkl"
    if tokenizer_pkl.exists():
        try:
            import pickle

            from transformers.integrations.tiktoken import convert_tiktoken_to_fast

            with open(tokenizer_pkl, "rb") as f:
                tok_pkl = pickle.load(f)
            convert_tiktoken_to_fast(tok_pkl, output_dir)
            print("Converted tokenizer.pkl to HuggingFace format")
        except Exception as e:
            print(f"Warning: Failed to convert tokenizer.pkl: {e}")
            # Fallback: copy tokenizer files if they exist
            for filename in ("tokenizer.json", "tokenizer_config.json"):
                src = input_path / filename
                if src.exists():
                    (Path(output_dir) / filename).write_bytes(src.read_bytes())
    else:
        # No pickle tokenizer, copy JSON files
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = input_path / filename
            if src.exists():
                (Path(output_dir) / filename).write_bytes(src.read_bytes())
    
    print("Tokenizer saved successfully.")


def run_test(output_dir: str, prompt: str, max_new_tokens: int = 64) -> None:
    """Run a quick generation test to verify the converted model works correctly."""
    print(f"Running quick generation test with prompt: {prompt}")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = NanoChatForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(f"Generated text: {generated}")


def main():
    parser = argparse.ArgumentParser(description="Convert NanoChat checkpoints to HuggingFace format")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the original checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Whether or not to save using `safetensors`.",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default=None,
        help="Optional prompt for a quick generation test",
    )
    args = parser.parse_args()
    
    write_model(
        args.input_dir,
        args.output_dir,
        safe_serialization=args.safe_serialization,
    )
    
    write_tokenizer(args.input_dir, args.output_dir)
    
    if args.test_prompt:
        run_test(args.output_dir, args.test_prompt)


if __name__ == "__main__":
    main()
