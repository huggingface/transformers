# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

"""Convert Qianfan-OCR weights to HuggingFace Transformers format.

The original model uses the same InternVL architecture with a Qwen3 LLM backend.
The vision encoder weights need QKV splitting and renaming; the LLM weights only
need prefix remapping; the MLP projector gets renamed from `mlp1.*` to the
`model.multi_modal_projector.*` convention.

Usage:
    python convert_qianfan_ocr_weights_to_hf.py \
        --input_dir /path/to/Qianfan-OCR \
        --output_dir /path/to/output
"""

import argparse
import gc
import os
import re

import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3Config,
)


UNNECESSARY_CONFIG_KEYS = [
    "_name_or_path",
    "_attn_implementation_autoset",
    "auto_map",
    "use_bfloat16",
    "use_flash_attn",
    "use_fa3",
    "bias",
    "drop_path_rate",
    "debug",
    "ep_size",
    "micro_forward",
    "skip_checkpoint",
    "use_deepep",
]

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION = {
    # Top-level prefix: vision_model.* → model.vision_tower.*
    r"^vision_model\.":                             r"model.vision_tower.",
    # Encoder layer list: encoder.layers.N → encoder.layer.N
    r"encoder\.layers\.":                           r"encoder.layer.",
    # NOTE: class_embedding, patch_embedding, position_embedding keep their
    #       original names because QianfanOCRVisionEmbeddings uses the same attribute names.
    # Layer scale params: ls1/ls2 → lambda_1/lambda_2
    r"\.ls(\d+)":                                   r".lambda_\1",
    # Attention projection: attn.proj → attention.projection_layer
    r"\.attn\.proj\.":                              r".attention.projection_layer.",
    # Attention dropout: attn.dropout → attention.projection_dropout
    r"\.attn\.dropout\.":                           r".attention.projection_dropout.",
    # Remaining attention submodules: attn.* → attention.*
    r"\.attn\.":                                    r".attention.",
    # Layer norms: norm1 → layernorm_before, norm2 → layernorm_after
    r"\.norm1\.":                                   r".layernorm_before.",
    r"\.norm2\.":                                   r".layernorm_after.",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT = {
    # language_model.model.* → model.language_model.* (AutoModel returns the backbone directly)
    r"^language_model\.model\.":                    r"model.language_model.",
    # lm_head is separated out of the language_model wrapper
    r"^language_model\.lm_head\.":                 r"lm_head.",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI = {
    # mlp1.0.* → model.multi_modal_projector.layer_norm.*
    r"^mlp1\.0\.":                                  r"model.multi_modal_projector.layer_norm.",
    # mlp1.1.* → model.multi_modal_projector.linear_1.*
    r"^mlp1\.1\.":                                  r"model.multi_modal_projector.linear_1.",
    # mlp1.3.* → model.multi_modal_projector.linear_2.*
    r"^mlp1\.3\.":                                  r"model.multi_modal_projector.linear_2.",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys, input_base_path=None):
    output_dict = {}

    def apply_mappings(text, mapping):
        for pattern, replacement in mapping.items():
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        return text

    # Vision keys
    vision_keys = [key for key in state_dict_keys if key.startswith("vision_model")]
    old_text_vision = "\n".join(vision_keys)
    new_text = apply_mappings(old_text_vision, ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION)
    output_dict.update(dict(zip(old_text_vision.split("\n"), new_text.split("\n"))))

    # Language model keys
    language_keys = [key for key in state_dict_keys if key.startswith("language_model")]
    old_text_language = "\n".join(language_keys)
    new_text = apply_mappings(old_text_language, ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT)
    output_dict.update(dict(zip(old_text_language.split("\n"), new_text.split("\n"))))

    # Multi-modal projector keys (everything else: mlp1.*)
    multi_keys = [
        key for key in state_dict_keys if not (key.startswith("language_model") or key.startswith("vision_model"))
    ]
    old_text_multi = "\n".join(multi_keys)
    new_text = apply_mappings(old_text_multi, ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI)
    output_dict.update(dict(zip(old_text_multi.split("\n"), new_text.split("\n"))))

    return output_dict


def load_original_state_dict(input_base_path):
    """Load state dict directly from safetensors files (no model instantiation)."""
    import glob

    from safetensors import safe_open

    state_dict = {}
    # Prefer sharded files (model-00001-of-*.safetensors) over the merged model.safetensors
    shard_files = sorted(glob.glob(os.path.join(input_base_path, "model-*-of-*.safetensors")))
    if not shard_files:
        # Fall back to single-file model.safetensors
        shard_files = sorted(glob.glob(os.path.join(input_base_path, "model.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {input_base_path}")
    print(f"Loading from {len(shard_files)} shard(s): {[os.path.basename(p) for p in shard_files]}")
    for shard_path in shard_files:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def get_qianfan_ocr_config(input_base_path, tokenizer):
    from transformers import QianfanOCRConfig, QianfanOCRVisionConfig

    # Use AutoConfig so we never instantiate the original model
    base_config = AutoConfig.from_pretrained(input_base_path, trust_remote_code=True)

    # Vision config
    vision_dict = base_config.vision_config.to_dict()
    vision_dict = {k: v for k, v in vision_dict.items() if k not in UNNECESSARY_CONFIG_KEYS}
    if "qk_normalization" in vision_dict:
        vision_dict["use_qk_norm"] = vision_dict.pop("qk_normalization")
    if "qkv_bias" in vision_dict:
        vision_dict["attention_bias"] = vision_dict.pop("qkv_bias")
    if "dropout" in vision_dict:
        vision_dict["hidden_dropout_prob"] = vision_dict.pop("dropout")
    if "attention_probs_dropout_prob" in vision_dict:
        attn_dropout = vision_dict.pop("attention_probs_dropout_prob")
        vision_dict["attention_dropout"] = attn_dropout
        vision_dict["projection_dropout"] = attn_dropout
    vision_dict["use_absolute_position_embeddings"] = True
    # Remove keys that don't map to QianfanOCRVisionConfig
    for k in ["architectures", "model_type", "torch_dtype", "initializer_factor"]:
        vision_dict.pop(k, None)

    # Text config (Qwen3)
    llm_dict = base_config.llm_config.to_dict()
    llm_dict = {k: v for k, v in llm_dict.items() if k not in UNNECESSARY_CONFIG_KEYS}
    llm_dict["use_cache"] = True
    text_config = Qwen3Config(**llm_dict)

    # Top-level config
    return QianfanOCRConfig(
        text_config=text_config,
        vision_config=QianfanOCRVisionConfig(**vision_dict),
        image_token_id=tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"),
        downsample_ratio=getattr(base_config, "downsample_ratio", 0.5),
        force_image_size=getattr(base_config, "force_image_size", 448),
        dynamic_image_size=getattr(base_config, "dynamic_image_size", True),
        use_thumbnail=getattr(base_config, "use_thumbnail", True),
        ps_version=getattr(base_config, "ps_version", "v2"),
        min_dynamic_patch=getattr(base_config, "min_dynamic_patch", 1),
        max_dynamic_patch=getattr(base_config, "max_dynamic_patch", 12),
    )


def write_model(model_path, input_base_path, push_to_hub=False):
    from transformers import QianfanOCRForConditionalGeneration, QianfanOCRImageProcessor, QianfanOCRProcessor

    os.makedirs(model_path, exist_ok=True)

    # Load tokenizer first so image_token_id can be derived from it
    tokenizer = AutoTokenizer.from_pretrained(input_base_path, trust_remote_code=True)

    config = get_qianfan_ocr_config(input_base_path, tokenizer)
    config.architectures = ["QianfanOCRForConditionalGeneration"]
    config.save_pretrained(model_path)
    print("Config saved.")

    # Convert weights
    print(f"Loading state dict from {input_base_path}...")
    state_dict_old = load_original_state_dict(input_base_path)
    all_keys = list(state_dict_old.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)
    dim = config.vision_config.hidden_size

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        if "attn.qkv" in key:
            # Split fused QKV into separate Q, K, V
            state_dict[new_key.replace("attention.qkv", "attention.q_proj")] = state_dict_old[key][:dim]
            state_dict[new_key.replace("attention.qkv", "attention.k_proj")] = state_dict_old[key][dim : 2 * dim]
            state_dict[new_key.replace("attention.qkv", "attention.v_proj")] = state_dict_old[key][-dim:]
        else:
            state_dict[new_key] = state_dict_old[key]

    del state_dict_old
    gc.collect()

    print("Loading into QianfanOCRForConditionalGeneration...")
    model = QianfanOCRForConditionalGeneration(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    model.save_pretrained(model_path, max_shard_size="4GB")
    print("Model saved.")

    if push_to_hub:
        model_name = model_path.split(os.path.sep)[-1]
        model.push_to_hub(model_name)

    # Save processor
    image_processor = QianfanOCRImageProcessor(
        do_resize=True,
        size={"height": 448, "width": 448},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_convert_rgb=True,
    )
    # Add special image token attributes required by InternVLProcessor
    tokenizer.start_image_token = "<img>"
    tokenizer.end_image_token = "</img>"
    tokenizer.context_image_token = "<IMG_CONTEXT>"
    tokenizer.video_token = "<video>"
    tokenizer.start_image_token_id = tokenizer.convert_tokens_to_ids("<img>")
    tokenizer.end_image_token_id = tokenizer.convert_tokens_to_ids("</img>")
    tokenizer.context_image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    processor = QianfanOCRProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(model_path)

    # Overwrite chat_template.jinja with a multimodal-aware template that handles
    # content as both string and list-of-dicts (image/video/text blocks),
    # and supports enable_thinking for Qwen3-style chain-of-thought.
    multimodal_chat_template = """\
{%- if messages[0]['role'] == 'system' %}
{{- '<|im_start|>system\\n' }}
{%- if messages[0]['content'] is string %}
{{- messages[0]['content'] }}
{%- else %}
{%- for item in messages[0]['content'] %}
{%- if item['type'] == 'text' %}{{- item['text'] }}
{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- else %}
{{- '<|im_start|>system\\n你是千帆VL，由百度智能云研发的多模态大语言模型。<|im_end|>\\n' }}
{%- endif %}
{%- set ns = namespace(found_last_user=false, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
{%- set index = (messages|length - 1) - loop.index0 %}
{%- if not ns.found_last_user and message['role'] == 'user' %}
{%- set ns.found_last_user = true %}
{%- set ns.last_query_index = index %}
{%- endif %}
{%- endfor %}
{%- for message in messages %}
{%- if messages[0]['role'] != 'system' or not loop.first %}
{%- if message['role'] == 'user' or (message['role'] == 'system' and not loop.first) %}
{%- set append_think = (enable_thinking is defined and enable_thinking and message['role'] == 'user' and loop.index0 == ns.last_query_index) %}
{{- '<|im_start|>' + message['role'] + '\\n' }}
{%- if message['content'] is string %}
{{- message['content'] }}
{%- else %}
{%- for item in message['content'] %}
{%- if item['type'] == 'image' %}{{- '<IMG_CONTEXT>\\n' }}
{%- elif item['type'] == 'video' %}{{- '<video>\\n' }}
{%- elif item['type'] == 'text' %}{{- item['text'] }}
{%- endif %}
{%- endfor %}
{%- endif %}
{%- if append_think %}{{- '\\n<think>' }}{%- endif %}
{{- '<|im_end|>\\n' }}
{%- elif message['role'] == 'assistant' %}
{%- if message['content'] is string %}
{%- set raw_content = message['content'] %}
{%- else %}
{%- set content_ns = namespace(raw='') %}
{%- for item in message['content'] %}
{%- if item['type'] == 'text' %}
{%- set content_ns.raw = content_ns.raw + item['text'] %}
{%- endif %}
{%- endfor %}
{%- set raw_content = content_ns.raw %}
{%- endif %}
{%- set content = raw_content %}
{%- set reasoning_content = '' %}
{%- if 'reasoning_content' in message and message['reasoning_content'] is not none %}
{%- set reasoning_content = message['reasoning_content'] %}
{%- elif '</think>' in raw_content %}
{%- set content = raw_content.split('</think>')[-1].lstrip('\\n') %}
{%- set reasoning_content = raw_content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
{%- endif %}
{%- if loop.index0 > ns.last_query_index and reasoning_content %}
{{- '<|im_start|>' + message['role'] + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
{%- else %}
{{- '<|im_start|>' + message['role'] + '\\n' + content }}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- if enable_thinking is defined and enable_thinking %}{{- '<think>\\n' }}{%- endif %}
{%- endif %}
"""
    with open(os.path.join(model_path, "chat_template.jinja"), "w") as f:
        f.write(multimodal_chat_template)

    # Persist special token attributes into tokenizer_config.json so they survive round-trip loading
    import json

    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    with open(tokenizer_config_path) as f:
        tokenizer_config = json.load(f)
    tokenizer_config["start_image_token"] = "<img>"
    tokenizer_config["end_image_token"] = "</img>"
    tokenizer_config["context_image_token"] = "<IMG_CONTEXT>"
    tokenizer_config["video_token"] = "<video>"
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print("Processor saved.")

    if push_to_hub:
        processor.push_to_hub(model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to original Qianfan-OCR model directory")
    parser.add_argument("--output_dir", required=True, help="Path to save converted HF model")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
