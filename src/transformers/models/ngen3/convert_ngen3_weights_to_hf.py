# Copyright 2024 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from tokenizers import Tokenizer

from transformers import NGEN3Config, NGEN3ForCausalLM
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

"""
Sample usage:

python src/transformers/models/ngen3/convert_ngen3_weights_to_hf.py \
    --input_dir /path/to/ngen3/weights --output_dir /output/path

After conversion, you can load the model as:

    from transformers import NGEN3ForCausalLM, AutoTokenizer
    model = NGEN3ForCausalLM.from_pretrained("/output/path")
    tokenizer = AutoTokenizer.from_pretrained("/output/path")
    
Important note: you need to have enough RAM to load the full checkpoint.
"""

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def write_json(content: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(content, f)

def write_model(
    model_path: str,
    input_base_path: str,
    include_tokenizer: bool = True,
    tokenizer_path: Path | None = None,
    safe_serialization: bool = True,
    tmp_cleanup: bool = True,
) -> None:
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.yaml"
    ngen3_config_yaml = yaml.safe_load(config_path.read_text())["model"]

    # Extract necessary hyperparameters from config
    n_layers = ngen3_config_yaml["n_layer"]
    n_heads = ngen3_config_yaml["n_head"]
    dim = ngen3_config_yaml["n_embd"]
    block_size = ngen3_config_yaml["block_size"]
    vocab_size = ngen3_config_yaml.get("vocab_size", 50257)
    dropout = ngen3_config_yaml.get("dropout", 0.1)
    instruct = ngen3_config_yaml.get("instruct", False)
    use_moe = ngen3_config_yaml.get("use_moe", False)
    num_experts = ngen3_config_yaml.get("num_experts", 4)

    # Precompute any positional parameters needed for RoPE
    dims_per_head = dim // n_heads
    base = 10000.0  # fixed base for RoPE
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = ngen3_config_yaml.get("max_position_embeddings", block_size)

    print(f"Loading all NGEN3 weights from checkpoint at {input_base_path}.")

    # Load the checkpoint (assumed unsharded in a single model.pt file)
    loaded = torch.load(os.path.join(input_base_path, "model.pt"), map_location="cpu")

    param_count = 0
    index_dict: Dict[str, Any] = {"weight_map": {}}
    # For each transformer block layer, extract and save weights
    for layer in range(n_layers):
        filename = f"pytorch_model-{layer + 1}-of-{n_layers + 1}.bin"
        # Here we assume the checkpoint uses a naming convention similar to:
        # "transformer.blocks.{layer}.att_proj.weight", "transformer.blocks.{layer}.ff_proj.weight", etc.
        # ...existing code to split and map weights...
        fused_dims = [dim, dims_per_head, dims_per_head]  # simplified assumption for projections
        qkv_weight = loaded[f"transformer.blocks.{layer}.att_proj.weight"]
        # Split qkv weight evenly
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(qkv_weight, dim, dim=0)
        up_proj_weight, gate_proj_weight = torch.chunk(loaded[f"transformer.blocks.{layer}.ff_proj.weight"], 2, dim=0)
        state_dict = {
            f"model.layers.{layer}.self_attn.q_proj.weight": q_proj_weight,
            f"model.layers.{layer}.self_attn.k_proj.weight": k_proj_weight,
            f"model.layers.{layer}.self_attn.v_proj.weight": v_proj_weight,
            f"model.layers.{layer}.self_attn.o_proj.weight": loaded[f"transformer.blocks.{layer}.attn_out.weight"],
            f"model.layers.{layer}.ln1.weight": loaded[f"transformer.blocks.{layer}.attn_norm.weight"],
            f"model.layers.{layer}.mlp.gate_proj.weight": gate_proj_weight,
            f"model.layers.{layer}.mlp.up_proj.weight": up_proj_weight,
            f"model.layers.{layer}.ln2.weight": loaded[f"transformer.blocks.{layer}.ff_norm.weight"],
        }
        # Save rotary inv_freq into the corresponding attention module
        state_dict[f"model.layers.{layer}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Save final model weights (embedding, final norm, output head)
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "model.norm.weight": loaded["transformer.ln_f.weight"],
        "lm_head.weight": loaded.get("transformer.ff_out.weight", loaded["transformer.wte.weight"]),
    }
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    # Instantiate and save a new NGEN3 config using the hyperparameters
    config = NGEN3Config(
        vocab_size=vocab_size,
        n_embd=dim,
        n_layer=n_layers,
        n_head=n_heads,
        block_size=block_size,
        dropout=dropout,
        instruct=instruct,
        use_moe=use_moe,
        num_experts=num_experts,
    )
    config.save_pretrained(tmp_model_path)

    # Cleanup loaded checkpoint to free memory
    del loaded
    gc.collect()

    if include_tokenizer:
        _write_tokenizer(model_path, config, input_base_path, tokenizer_path)

    print("Loading the checkpoint into a NGEN3 model.")
    model = NGEN3ForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    if tmp_cleanup:
        shutil.rmtree(tmp_model_path)

def _write_tokenizer(
    output_path: str,
    config: NGEN3Config,
    checkpoint_dir: str,
    input_tokenizer_path: Path | None,
) -> None:
    print(f"Saving a {GPT2TokenizerFast.__name__} to {output_path}.")
    if input_tokenizer_path is not None:
        base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))
    else:
        config_path = Path(checkpoint_dir) / "config.yaml"
        tokenizer_config = yaml.safe_load(config_path.read_text())["tokenizer"]
        if Path(tokenizer_config["identifier"]).is_file():
            base_tokenizer = Tokenizer.from_file(tokenizer_config["identifier"])
        else:
            base_tokenizer = Tokenizer.from_pretrained(tokenizer_config["identifier"])

    eos_token_id = config.eos_token_id if hasattr(config, "eos_token_id") and config.eos_token_id is not None else base_tokenizer.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else eos_token_id
    tokenizer = GPT2TokenizerFast(
        tokenizer_object=base_tokenizer,
        eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
    )
    tokenizer.save_pretrained(output_path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of NGEN3 weights, which should contain config.yaml and model.pt.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not convert NGEN3 tokenizer to HF tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        type=Path,
        default=None,
        help="Location of NGEN3 tokenizer json file. Defaults to what is set in the config file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write the HF model and tokenizer.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_false",
        dest="safe_serialization",
        help="If set, do not use safetensors for saving.",
    )
    parser.add_argument(
        "--no_tmp_cleanup",
        action="store_false",
        dest="tmp_cleanup",
        help="If set, do not remove the temporary directory after conversion.",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        include_tokenizer=args.include_tokenizer,
        tokenizer_path=args.tokenizer_json_path,
        tmp_cleanup=args.tmp_cleanup,
    )

if __name__ == "__main__":
    main()
