# Copyright 2025 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
import json
import os

import requests
import torch
import yaml
from accelerate import init_empty_weights
from PIL import Image

from transformers import (
    AnoleConfig,
    AnoleForConditionalGeneration,
    ChameleonImageProcessor,
    ChameleonProcessor,
)


try:
    from transformers import LlamaTokenizerFast
except ImportError:
    raise ValueError(
        "Anole conversion supports only FastTokenizer and LlamaTokenizerFast can't be imported! "
        "Update your `tokenizers` library and re-run the tokenizer conversion."
    )

"""
Sample usage:

```
python src/transformers/models/anole/convert_anole_weights_to_hf.py \
    --input_dir /path/to/downloaded/anole/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import AnoleForConditionalGeneration, LlamaTokenizerFast

model = AnoleForConditionalGeneration.from_pretrained("/output/path")
tokenizer = LlamaTokenizerFast.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

NUM_SHARDS = {
    "7B": 1,
    "30B": 4,
}

VOCAB_SIZE = 65536


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, anole_version=1):
    os.makedirs(model_path, exist_ok=True)
    input_model_path = os.path.join(input_base_path, "models", model_size.lower())
    params_path = os.path.join(input_model_path, "params.json")
    consolidate_params_path = os.path.join(input_model_path, "consolidate_params.json")

    params = read_json(params_path)
    if os.path.isfile(consolidate_params_path):
        params = {**params, **read_json(consolidate_params_path)}
    num_shards = NUM_SHARDS[model_size]
    model_parallel_size = params["model_parallel_size"]
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    swin_norm = params["swin_norm"]
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        # Depending on the Anole version, the default max_position_embeddings has different values.
        if anole_version == 1:
            max_position_embeddings = 4096
        else:
            raise NotImplementedError(
                f"Version {anole_version} of anole is not supported yet. "
                "Current supported versions of anole are [1]."
            )

    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    print(f"Fetching all parameters from the checkpoint at {input_model_path}.")
    # Load weights
    if num_shards == 1:
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = None
        for possible_name in ["consolidated.pth", "consolidated.00.pth"]:
            possible_path = os.path.join(input_model_path, possible_name)
            if os.path.exists(possible_path):
                loaded = torch.load(possible_path, map_location="cpu")
                break
        assert loaded is not None
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_model_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Load weights to the state dict
    state_dict = {}
    for layer_i in range(n_layers):
        if num_shards == 1:
            # Unsharded
            state_dict.update(
                {
                    f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                        loaded[f"layers.{layer_i}.attention.wq.weight"], n_heads=n_heads
                    ),
                    f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                        loaded[f"layers.{layer_i}.attention.wk.weight"],
                        n_heads=num_key_value_heads,
                        dim1=key_value_dim,
                    ),
                    f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                    f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                    f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                    f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                    f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                    f"model.layers.{layer_i}.input_layernorm.weight": loaded[
                        f"layers.{layer_i}.attention_norm.weight"
                    ],
                    f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                        f"layers.{layer_i}.ffn_norm.weight"
                    ],
                }
            )
            # qk_layernorm (see https://github.com/huggingface/transformers/pull/31534#issuecomment-2207354677)
            state_dict[f"model.layers.{layer_i}.self_attn.q_norm.weight"] = (
                loaded[f"layers.{layer_i}.attention.q_normalization.weight"]
                .view(dims_per_head // 2, 2)
                .t()
                .reshape(1, -1)
                .repeat_interleave(n_heads, 0)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.q_norm.bias"] = (
                loaded[f"layers.{layer_i}.attention.q_normalization.bias"]
                .view(dims_per_head // 2, 2)
                .t()
                .reshape(1, -1)
                .repeat_interleave(n_heads, 0)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_norm.weight"] = (
                loaded[f"layers.{layer_i}.attention.k_normalization.weight"]
                .view(dims_per_head // 2, 2)
                .t()
                .reshape(1, -1)
                .repeat_interleave(num_key_value_heads, 0)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_norm.bias"] = (
                loaded[f"layers.{layer_i}.attention.k_normalization.bias"]
                .view(dims_per_head // 2, 2)
                .t()
                .reshape(1, -1)
                .repeat_interleave(num_key_value_heads, 0)
            )

        else:
            # Sharded
            state_dict.update(
                {
                    f"model.layers.{layer_i}.input_layernorm.weight": torch.stack(
                        [l[f"layers.{layer_i}.attention_norm.weight"] for l in loaded]
                    ).mean(dim=0),
                    f"model.layers.{layer_i}.post_attention_layernorm.weight": torch.stack(
                        [l[f"layers.{layer_i}.ffn_norm.weight"] for l in loaded]
                    ).mean(dim=0),
                }
            )
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim),
                n_heads=n_heads,
            )

            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                            num_local_key_value_heads, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim),
                n_heads=num_key_value_heads,
                dim1=key_value_dim,
            )

            # qk_layernorm (see https://github.com/huggingface/transformers/pull/31534#issuecomment-2207354677)
            state_dict[f"model.layers.{layer_i}.self_attn.q_norm.weight"] = (
                torch.cat([l[f"layers.{layer_i}.attention.q_normalization.weight"].unsqueeze(0) for l in loaded])
                .view(num_shards, dims_per_head // 2, 2)
                .transpose(1, 2)
                .reshape(num_shards, -1)
                .repeat_interleave(n_heads // num_shards, 0)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.q_norm.bias"] = (
                torch.cat([l[f"layers.{layer_i}.attention.q_normalization.bias"].unsqueeze(0) for l in loaded])
                .view(num_shards, dims_per_head // 2, 2)
                .transpose(1, 2)
                .reshape(num_shards, -1)
                .repeat_interleave(n_heads // num_shards, 0)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_norm.weight"] = (
                torch.cat([l[f"layers.{layer_i}.attention.k_normalization.weight"].unsqueeze(0) for l in loaded])
                .view(num_shards, dims_per_head // 2, 2)
                .transpose(1, 2)
                .reshape(num_shards, -1)
                .repeat_interleave(num_key_value_heads // num_shards, 0)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_norm.bias"] = (
                torch.cat([l[f"layers.{layer_i}.attention.k_normalization.bias"].unsqueeze(0) for l in loaded])
                .view(num_shards, dims_per_head // 2, 2)
                .transpose(1, 2)
                .reshape(num_shards, -1)
                .repeat_interleave(num_key_value_heads // num_shards, 0)
            )

            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                        num_local_key_value_heads, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

    if num_shards == 1:
        # Unsharded
        state_dict.update(
            {
                "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
                "model.norm.weight": loaded["norm.weight"],
                "lm_head.weight": loaded["output.weight"],
            }
        )
    else:
        state_dict.update(
            {
                "model.embed_tokens.weight": torch.cat(
                    [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
                ),
                "model.norm.weight": torch.stack([loaded[i]["norm.weight"] for i in range(num_shards)]).mean(dim=0),
                "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
            }
        )

    # Load VQGAN weights
    vqgan_path = os.path.join(input_base_path, "tokenizer/vqgan.ckpt")
    vqgan_state_dict = torch.load(vqgan_path, map_location="cpu")["state_dict"]
    for k, v in vqgan_state_dict.items():
        state_dict[f"model.vqmodel.{k}"] = v

    # Write configs
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256

    with open(os.path.join(input_base_path, "tokenizer/text_tokenizer.json")) as tokenizer_file:
        tokenizer_config = json.load(tokenizer_file)
        vocabulary_map = tokenizer_config["model"]["vocab"]
        # use a reserved token instead of adding a new one
        vocabulary_map["<image>"] = vocabulary_map["<reserved08707>"]
        del vocabulary_map["<reserved08707>"]

        for token in tokenizer_config["added_tokens"]:
            if token["content"] == "<reserved08707>":
                token["content"] = "<image>"

    with open(os.path.join(input_base_path, "tokenizer/text_tokenizer_modified.json"), "w") as f:
        json.dump(tokenizer_config, f)  # save the new file to init tokenizer later

    vq_keys_to_replace = [
        ("ch", "base_channels"),
        ("out_ch", "out_channels"),
        ("n_embed", "num_embeddings"),
        ("ch_mult", "channel_multiplier"),
        ("double_z", "double_latent"),
        ("z_channels", "latent_channels"),
    ]
    with open(os.path.join(input_base_path, "tokenizer/vqgan.yaml")) as vqgan_cfg_file:
        vq_config = yaml.safe_load(vqgan_cfg_file)["model"]["params"]
        vq_config.update(**vq_config["ddconfig"])
        for old, new in vq_keys_to_replace:
            vq_config[new] = vq_config[old]
        del vq_config["ddconfig"]
        del vq_config["ckpt_path"]
        del vq_config["lossconfig"]

    config = AnoleConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=VOCAB_SIZE,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        model_parallel_size=model_parallel_size,
        swin_norm=swin_norm,
        vq_config=vq_config,
        vocabulary_map=vocabulary_map,
        image_token_id=vocabulary_map["<image>"],
        boi_token_id=vocabulary_map["<racm3:break>"],
        eoi_token_id=vocabulary_map["<eoss>"],
    )
    with init_empty_weights():
        model = AnoleForConditionalGeneration(config)

    model.load_state_dict(state_dict, assign=True, strict=False)
    model.save_pretrained(model_path, safe_serialization=True)

    # Load and save the processor
    extra_special_tokens = {
        "image_token": "<image>",
        "boi_token": "<racm3:break>",
        "eoi_token": "<eoss>",
    }
    tokenizer = LlamaTokenizerFast(
        tokenizer_file=os.path.join(input_base_path, "tokenizer/text_tokenizer_modified.json"),
        legacy=False,
        extra_special_tokens=extra_special_tokens,
    )
    tokenizer.sep_token_id = 8710  # assign <reserved08706> to sep so that we can append it after input text
    tokenizer.pad_token_id = 1  # assing <pad> to special pad_token
    image_processor = ChameleonImageProcessor()
    processor = ChameleonProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    del vqgan_state_dict
    gc.collect()

    # Short inference on a few examples to check if generation makes sense
    # taken from https://github.com/facebookresearch/anole/blob/7a72f40aa5f462965c8374f25257f55b65b25ff4/data/prompts_for_human_evaluations.jsonl
    print("Loading the checkpoint in a Anole model...")
    print("*" * 100)
    model = AnoleForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = ChameleonProcessor.from_pretrained(model_path)

    prompt = "I'm very intrigued by this work of art:<image>Please tell me about the artist."
    image = Image.open(
        requests.get(
            "https://uploads4.wikiart.org/images/paul-klee/death-for-the-idea-1915.jpg!Large.jpg", stream=True
        ).raw
    )
    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)
    length = inputs.input_ids.shape[1]

    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    generated_text = processor.batch_decode(out[:, length:], skip_special_tokens=True)[0]

    print(f"Generation for single-image: {generated_text}")
    print("*" * 100)

    # Multi-image example
    prompt = "I used to know a lot about constellations when I was younger, but as I grew older, I forgot most of what I knew. These are the only two constellations that I really remember now.<image><image>I would like for you to tell me about 3 more constellations and give me a little bit of history about the constellation."
    image = Image.open(
        requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
    )
    image_2 = Image.open(
        requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw
    )

    inputs = processor(prompt, images=[image, image_2], return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    length = inputs.input_ids.shape[1]
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated_text = processor.batch_decode(out[:, length:], skip_special_tokens=True)[0]

    print(f"Generation for multi-image: {generated_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Anole weights",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "30B"],
        help=""
        " models correspond to the finetuned versions, and are specific to the Anole official release. For more details on Anole, checkout the original repo: https://github.com/facebookresearch/anole",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model",
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Whether to load the model for generation to test it's converted correctly.",
    )
    # Different Anole versions used different default values for max_position_embeddings, hence the need to be able to specify which version is being used.
    parser.add_argument(
        "--anole_version",
        choices=[1],
        default=1,
        type=int,
        help="Version of the Anole model to convert",
    )

    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        anole_version=args.anole_version,
    )


if __name__ == "__main__":
    main()
