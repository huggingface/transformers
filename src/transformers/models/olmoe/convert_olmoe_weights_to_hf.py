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
"""
Example for running:
0. Cp ckpts to local
aws s3 cp --recursive s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-final-annealFrom1200000/step23842 /data/niklas/llm/checkpoints/olmoe-8x1b-newhp-newds-final-annealFrom1200000_step23842
1. Unshard your OLMoE checkpoint using https://github.com/allenai/OLMo/blob/7d63fe09d23cf23714da5aa633a44a90180195da/scripts/unshard.py
python OLMo/scripts/unshard.py /data/niklas/llm/checkpoints/23485/step954000 /data/niklas/llm/checkpoints/1b-954000-unsharded --model-only
python OLMo/scripts/unshard.py /data/niklas/llm/checkpoints/23485/step954000 /data/niklas/llm/checkpoints/1b-954000-unsharded --model-only
python OLMo/scripts/unshard.py /data/niklas/llm/checkpoints/olmoe-8x1b-newhp-newds-final-annealFrom1200000_step23842 /data/niklas/llm/checkpoints/olmoe-8x1b-newhp-newds-final-annealFrom1200000_step23842-unsharded --model-only
2. Convert to transformers
rm -rf olmoe; mkdir olmoe; python /data/niklas/transformers/src/transformers/models/olmoe/convert_olmoe_weights_to_hf.py --input_dir /data/niklas/llm/checkpoints/olmoe-8x1b-newhp-newds-final-annealFrom1200000_step23842-unsharded --tokenizer_json_path /data/niklas/llm/checkpoints/olmoe-step1200000-unsharded/tokenizer.json --output_dir olmoe
3. Load model via:
```
from transformers import OlmoeForCausalLM, AutoTokenizer
import torch
model = OlmoeForCausalLM.from_pretrained("../transformers/olmoe", dtype=torch.bfloat16).cuda()
model = OlmoeForCausalLM.from_pretrained("../transformers/olmoe").cuda()
tokenizer = AutoTokenizer.from_pretrained("../transformers/olmoe")
inputs = tokenizer("Bitcoin is", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}
out = model.generate(**inputs, max_length=64)
print(tokenizer.decode(out[0]))
# > # Bitcoin is a digital currency that is created and held electronically. No one controls it. Bitcoins aren’t printed, like dollars or euros – they’re produced by people and businesses running computers all around the world, using software that solves mathematical
# Or quick sanity check:
o = model(torch.tensor([[0, 1]]).cuda())
# If the checkpoint is not converted to BF16 but kept in FP32:
# > # Bitcoin is a digital currency that is not controlled by any central authority. It is a peer-to-peer payment system that allows users to send and receive payments from anywhere in the world. Bitcoin is also known as a cryptocurrency because it uses cryptography to secure transactions and prevent fraud.
```

Note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).

Compare with OLMo codebase:
```
from olmo.model import OLMo
import torch
model = OLMo.from_checkpoint("/data/niklas/llm/checkpoints/olmoe-step1200000-unsharded-pt")
model = model.cuda()
model = model.to(torch.bfloat16)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../transformers/olmoe")
inputs = tokenizer("Bitcoin is", return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}
out = model.generate(**inputs)
print(tokenizer.decode(out[0][0][0]))
# Bitcoin is a digital currency that is created and held electronically. No one controls it. Bitcoins aren’t printed, like dollars or euros – they’re produced by people and businesses running computers all around the world, using software that solves mathematical problems. It’s the first example of a growing category of money
# Or quick sanity check:
o = model(torch.tensor([[0, 1]]).cuda())
```
"""

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from transformers import OlmoeConfig, OlmoeForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, tokenizer_path=None, safe_serialization=True, fix_eos_token_id=True):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.yaml"
    olmoe_config = yaml.safe_load(config_path.read_text())["model"]

    if fix_eos_token_id:
        olmoe_config["eos_token_id"] = 50279

    n_layers = olmoe_config["n_layers"]
    n_heads = olmoe_config["n_heads"]
    dim = olmoe_config["d_model"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = olmoe_config["max_sequence_length"]

    vocab_size = olmoe_config.get("embedding_size", olmoe_config["vocab_size"])

    if olmoe_config.get("n_kv_heads", None) is not None:
        num_key_value_heads = olmoe_config["n_kv_heads"]  # for GQA / MQA
    elif olmoe_config["multi_query_attention"]:  # compatibility with other checkpoints
        num_key_value_heads = 1
    else:
        num_key_value_heads = n_heads

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

    # Not sharded
    loaded = torch.load(os.path.join(input_base_path, "model.pt"), map_location="cpu", weights_only=True)

    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        fused_dims = [dim, dims_per_head * num_key_value_heads, dims_per_head * num_key_value_heads]
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            loaded[f"transformer.blocks.{layer_i}.att_proj.weight"], fused_dims, dim=0
        )
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight": v_proj_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"transformer.blocks.{layer_i}.attn_out.weight"],
            f"model.layers.{layer_i}.self_attn.q_norm.weight": loaded[f"transformer.blocks.{layer_i}.q_norm.weight"],
            f"model.layers.{layer_i}.self_attn.k_norm.weight": loaded[f"transformer.blocks.{layer_i}.k_norm.weight"],
            f"model.layers.{layer_i}.mlp.gate.weight": loaded[f"transformer.blocks.{layer_i}.ffn.router.layer.weight"],
            f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"transformer.blocks.{layer_i}.attn_norm.weight"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                f"transformer.blocks.{layer_i}.ff_norm.weight"
            ],
        }

        num_experts = loaded[f"transformer.blocks.{layer_i}.ffn.router.layer.weight"].shape[0]
        dim_per_expert = loaded[f"transformer.blocks.{layer_i}.ffn.experts.mlp.w1"].shape[0] // num_experts
        for expert_i in range(num_experts):
            state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.gate_proj.weight"] = loaded[
                f"transformer.blocks.{layer_i}.ffn.experts.mlp.w1"
            ][dim_per_expert * expert_i : dim_per_expert * (expert_i + 1), :]
            state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.up_proj.weight"] = loaded[
                f"transformer.blocks.{layer_i}.ffn.experts.mlp.v1"
            ][dim_per_expert * expert_i : dim_per_expert * (expert_i + 1), :]
            state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.down_proj.weight"] = loaded[
                f"transformer.blocks.{layer_i}.ffn.experts.mlp.w2"
            ][dim_per_expert * expert_i : dim_per_expert * (expert_i + 1), :].T.contiguous()

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"

    # Unsharded
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "lm_head.weight": loaded["transformer.ff_out.weight"],
        "model.norm.weight": loaded["transformer.ln_f.weight"],
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = OlmoeConfig(
        vocab_size=vocab_size,
        hidden_size=dim,
        intermediate_size=dim_per_expert,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=olmoe_config["pad_token_id"],
        bos_token_id=None,
        eos_token_id=olmoe_config["eos_token_id"],
        tie_word_embeddings=olmoe_config["weight_tying"],
        rope_theta=base,
        clip_qkv=olmoe_config.get("clip_qkv"),
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    if tokenizer_path is not None:
        _write_tokenizer(model_path, config, tokenizer_path, fix_eos_token_id)

    print("Loading the checkpoint in a OLMoE model.")
    model = OlmoeForCausalLM.from_pretrained(tmp_model_path, dtype=torch.bfloat16)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


def _write_tokenizer(
    output_path: Path, config: OlmoeConfig, input_tokenizer_path: Path, fix_eos_token_id: bool = True
) -> None:
    print(f"Saving a {GPTNeoXTokenizerFast.__name__} to {output_path}.")

    base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))

    eos_token_id = config.eos_token_id if config.eos_token_id is not None else base_tokenizer.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else eos_token_id

    if fix_eos_token_id and eos_token_id == 0:
        # Fixing a bug in OLMo where eos token id was incorrectly set
        print("Changing eos_token_id from 0 to 50279.")
        eos_token_id = 50279

    tokenizer = GPTNeoXTokenizerFast(
        tokenizer_object=base_tokenizer,
        eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
        unk_token=None,
        bos_token=None,
    )

    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMoE weights, which contains config.yaml and model.pt.",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        default=None,
        help="Location of OLMoE tokenizer json file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--no_fix_eos_token_id",
        action="store_false",
        dest="fix_eos_token_id",
        help="If set, does not change eos token id from 0 to 50279 if it is 0. Changing 0 to 50279 is a bug fix, so use this option with care.",
    )
    parser.add_argument(
        "--safe_serialization", type=bool, default=True, help="Whether or not to save using `safetensors`."
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        tokenizer_path=args.tokenizer_json_path,
        fix_eos_token_id=args.fix_eos_token_id,
    )


if __name__ == "__main__":
    main()
