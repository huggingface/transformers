# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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
"""Convert the original MiniCPM-Llama3-V 2.5 checkpoint (remote code) to the Transformers format.

    python src/transformers/models/minicpmv2_5/convert_minicpmv2_5_weights_to_hf.py \
        --input_dir /path/to/MiniCPM-Llama3-V-2_5 --output_dir /path/to/converted
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

from transformers import (
    GenerationConfig,
    MiniCPMV2_5Config,
    MiniCPMV2_5ForConditionalGeneration,
    MiniCPMV2_5ImageProcessor,
    MiniCPMV2_5Processor,
    PreTrainedTokenizerFast,
)


# Original key prefix -> Transformers key prefix (first match wins). The resampler's packed q/k/v is handled separately.
KEY_RENAMES = [
    ("llm.model.", "model.language_model."),
    ("llm.lm_head.", "lm_head."),
    ("vpm.", "model.vision_tower."),
    ("resampler.attn.out_proj.", "model.resampler.o_proj."),
    ("resampler.", "model.resampler."),
]

TEXT_CONFIG_KEYS = [
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "hidden_act", "max_position_embeddings", "rms_norm_eps", "attention_bias",
    "attention_dropout", "bos_token_id", "eos_token_id", "initializer_range", "tie_word_embeddings",
]  # fmt: skip

# Same format as the original Llama 3 template; images (one `<unk>` each) and text items are joined with newlines,
# matching the original `chat()`, which put "(<image>./</image>)\n" before the question.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['content'] is string -%}{%- set content = message['content'] -%}"
    "{%- else -%}{%- set ns = namespace(parts=[]) -%}"
    "{%- for item in message['content'] -%}"
    "{%- if item['type'] == 'image' -%}{%- set ns.parts = ns.parts + ['<unk>'] -%}"
    "{%- elif item['type'] == 'text' -%}{%- set ns.parts = ns.parts + [item['text']] -%}{%- endif -%}"
    "{%- endfor -%}{%- set content = ns.parts | join('\n') -%}{%- endif -%}"
    "{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' -%}"
    "{%- if loop.index0 == 0 -%}{%- set content = bos_token + content -%}{%- endif -%}"
    "{{- content -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}{%- endif -%}"
)


def convert_config(original: dict) -> MiniCPMV2_5Config:
    text_config = {key: original[key] for key in TEXT_CONFIG_KEYS if key in original}
    text_config["model_type"] = "llama"
    text_config["rope_parameters"] = {"rope_type": "default", "rope_theta": original["rope_theta"]}
    vision_config = {key: value for key, value in original["vision_config"].items() if key != "model_type"}
    return MiniCPMV2_5Config(
        text_config=text_config,
        vision_config=vision_config,
        query_num=original["query_num"],
        tie_word_embeddings=original.get("tie_word_embeddings", False),
    )


def convert_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    with open(os.path.join(input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    state_dict = {}
    for shard in shards:
        with safe_open(os.path.join(input_dir, shard), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in ("resampler.attn.in_proj_weight", "resampler.attn.in_proj_bias"):
                    # nn.MultiheadAttention packs q, k and v along dim 0, in that order.
                    suffix = "weight" if key.endswith("weight") else "bias"
                    for name, part in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
                        state_dict[f"model.resampler.{name}.{suffix}"] = part.clone()
                    continue
                new_key = next((key.replace(old, new, 1) for old, new in KEY_RENAMES if key.startswith(old)), None)
                if new_key is None:
                    raise ValueError(f"Unexpected key in the original checkpoint: {key}")
                state_dict[new_key] = tensor
    return state_dict


def convert(input_dir: str, output_dir: str):
    with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as f:
        config = convert_config(json.load(f))

    with torch.device("meta"):
        model = MiniCPMV2_5ForConditionalGeneration(config)
    state_dict = convert_state_dict(input_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"Weight mismatch. Missing: {missing}. Unexpected: {unexpected}")
    model.to(torch.float16)
    model.generation_config = GenerationConfig(bos_token_id=128000, eos_token_id=[128001, 128009], pad_token_id=128001)
    model.save_pretrained(output_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(input_dir, clean_up_tokenization_spaces=False)
    processor = MiniCPMV2_5Processor(
        image_processor=MiniCPMV2_5ImageProcessor(), tokenizer=tokenizer, chat_template=CHAT_TEMPLATE
    )
    processor.save_pretrained(output_dir)
    print(f"Saved converted model and processor to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Local copy of openbmb/MiniCPM-Llama3-V-2_5.")
    parser.add_argument("--output_dir", required=True, help="Where to write the converted checkpoint.")
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
