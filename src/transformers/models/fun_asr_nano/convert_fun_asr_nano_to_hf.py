#!/usr/bin/env python3
# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Convert Fun-ASR-Nano checkpoint to Hugging Face Transformers format.

Usage:

1) Run the conversion script, downloading the original checkpoint directly from the Hub:

```bash
python src/transformers/models/fun_asr_nano/convert_fun_asr_nano_to_hf.py \
    --model_id FunAudioLLM/Fun-ASR-Nano-2512 \
    --output_path ./fun-asr-nano-hf
```

Alternatively, convert an already-downloaded local checkpoint:

```bash
hf download FunAudioLLM/Fun-ASR-Nano-2512 \
    --local-dir /path/to/Fun-ASR-Nano-2512

python src/transformers/models/fun_asr_nano/convert_fun_asr_nano_to_hf.py \
    --model_path /path/to/Fun-ASR-Nano-2512 \
    --output_path ./fun-asr-nano-hf
```

2) Verify the converted checkpoint can be loaded:

```bash
python - <<'PY'
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "./fun-asr-nano-hf"
AutoProcessor.from_pretrained(model_id)
AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
PY
```

3) Optionally push the converted checkpoint to the Hub:

```bash
python src/transformers/models/fun_asr_nano/convert_fun_asr_nano_to_hf.py \
    --model_id FunAudioLLM/Fun-ASR-Nano-2512 \
    --output_path ./fun-asr-nano-hf \
    --push_to_hub your-username/Fun-ASR-Nano-2512-hf
```
"""

import argparse
import json
import os
import re

import torch
import yaml
from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    FunAsrNanoConfig,
    FunAsrNanoEncoderConfig,
    FunAsrNanoFeatureExtractor,
    FunAsrNanoForConditionalGeneration,
    FunAsrNanoProcessor,
    Qwen3Config,
)


ROOT_STATE_DICT_MAPPING = (
    (r"^audio_encoder\.encoders0\.0\.", "model.audio_tower.stem."),
    (r"^audio_encoder\.encoders\.", "model.audio_tower.layers."),
    (r"^audio_encoder\.tp_encoders\.", "model.audio_tower.timestamp_prediction_layers."),
    (r"^audio_encoder\.after_norm\.", "model.audio_tower.layer_norm."),
    (r"^audio_encoder\.tp_norm\.", "model.audio_tower.timestamp_prediction_layer_norm."),
    (r"^audio_adaptor\.blocks\.", "model.audio_adaptor.blocks."),
    (r"^audio_adaptor\.linear1\.", "model.multi_modal_projector.linear_1."),
    (r"^audio_adaptor\.linear2\.", "model.multi_modal_projector.linear_2."),
    # Keep lm_head.weight explicitly. Although tie_word_embeddings=True, this model load path
    # does not retie lm_head from the embeddings, and the source already stores lm_head == embeddings.
    # safetensors deduplicates the shared storage, so this adds no extra disk over the embeddings.
    (r"^llm\.lm_head\.", "lm_head."),
    (r"^llm\.model\.", "model.language_model."),
)

COMPONENT_STATE_DICT_MAPPING = (
    (r"\.feed_forward\.w_1\.", ".fc1."),
    (r"\.feed_forward\.w_2\.", ".fc2."),
    (r"\.norm1\.", ".self_attn_layer_norm."),
    (r"\.norm2\.", ".final_layer_norm."),
    (r"\.self_attn\.linear_q\.", ".self_attn.q_proj."),
    (r"\.self_attn\.linear_k\.", ".self_attn.k_proj."),
    (r"\.self_attn\.linear_v\.", ".self_attn.v_proj."),
    (r"\.self_attn\.linear_out\.", ".self_attn.out_proj."),
    (r"\.self_attn\.fsmn_block\.", ".feedforward_sequential_memory.conv."),
)


# Chat template stored in the checkpoint so that `processor.apply_chat_template` builds the checkpoint's
# transcription prompt. Audio elements are replaced with the `<|object_ref_start|>` placeholder token,
# which the processor later expands to the right number of audio tokens.
# fmt: off
CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if loop.first and message['role'] != 'system' %}"
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "{% endif %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is string %}"
            "{{ message['content'] }}<|im_end|>\n"
        "{% else %}"
            "{% set audio_items = message['content'] | selectattr('type', 'equalto', 'audio') | list %}"
            "{% if message['role'] == 'user' and audio_items %}"
                "{% set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list %}"
                "{% set keyword_items = message['content'] | selectattr('type', 'equalto', 'keywords') | list %}"
                "{% set language_items = message['content'] | selectattr('type', 'equalto', 'language') | list %}"
                "{% if text_items or keyword_items %}"
                    "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
                    "**上下文信息：**\n\n\n"
                    "{% for item in text_items %}{{ item['text'] }}\n{% endfor %}"
                    "{% set keyword_namespace = namespace(items=[]) %}"
                    "{% for item in keyword_items %}"
                        "{% set keyword_namespace.items = keyword_namespace.items + item['keywords'] %}"
                    "{% endfor %}"
                    "{% if keyword_namespace.items %}"
                        "热词列表：[{{ keyword_namespace.items | join(', ') }}]\n"
                    "{% endif %}"
                "{% endif %}"
                "语音转写{% if language_items %}成{{ language_items[0]['language'] }}{% endif %}："
                "{% for content in audio_items %}<|object_ref_start|>{% endfor %}"
            "{% else %}"
                "{% for content in message['content'] %}"
                    "{% if content['type'] == 'audio' %}"
                        "<|object_ref_start|>"
                    "{% elif content['type'] == 'text' %}"
                        "{{ content['text'] }}"
                    "{% endif %}"
                "{% endfor %}"
            "{% endif %}"
            "<|im_end|>\n"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
    "{% endif %}"
)
# fmt: on


def load_original_checkpoint(checkpoint_path: str) -> dict:
    """Load the original FunASR model.pt checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def convert_key(key: str) -> str | None:
    """Map one original FunASR checkpoint key to the HF layout."""
    mapped_key = None
    for pattern, replacement in ROOT_STATE_DICT_MAPPING:
        if re.match(pattern, key):
            mapped_key = re.sub(pattern, replacement, key)
            break
    if mapped_key is None or ".linear_q_k_v." in mapped_key:
        return None

    for pattern, replacement in COMPONENT_STATE_DICT_MAPPING:
        mapped_key = re.sub(pattern, replacement, mapped_key)
    return mapped_key


def convert_state_dict(original_state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Convert checkpoint keys and split each fused encoder QKV projection."""
    converted_state_dict = {}
    unconverted_keys = []

    for key, value in original_state_dict.items():
        if ".self_attn.linear_q_k_v." in key:
            if value.shape[0] % 3 != 0:
                raise ValueError(f"Fused QKV tensor must be divisible by 3 along dim 0, got {key}: {value.shape}.")
            projection_values = value.chunk(3, dim=0)
            projection_keys = [
                convert_key(key.replace("linear_q_k_v", f"{projection}_proj")) for projection in ("q", "k", "v")
            ]
            if any(projection_key is None for projection_key in projection_keys):
                unconverted_keys.append(key)
                continue
            for projection_key, projection_value in zip(projection_keys, projection_values):
                converted_state_dict[projection_key] = projection_value.to(torch.bfloat16)
            continue

        new_key = convert_key(key)
        if new_key is None:
            unconverted_keys.append(key)
        else:
            converted_state_dict[new_key] = value.to(torch.bfloat16)

    return converted_state_dict, unconverted_keys


def build_config_from_yaml(config_yaml_path: str, qwen3_config_path: str) -> FunAsrNanoConfig:
    """Build HF config from original config.yaml."""
    with open(config_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Audio encoder config (standalone encoder model -> standalone config, Parakeet-style).
    enc_conf = cfg.get("audio_encoder_conf", {})
    encoder_config = FunAsrNanoEncoderConfig(
        num_mel_bins=80,
        num_stacked_frames=7,
        d_model=enc_conf.get("output_size", 512),
        encoder_attention_heads=enc_conf.get("attention_heads", 4),
        encoder_ffn_dim=enc_conf.get("linear_units", 2048),
        encoder_layers=enc_conf.get("num_blocks", 50),
        num_timestamp_prediction_blocks=enc_conf.get("tp_blocks", 20),
        dropout=enc_conf.get("dropout_rate", 0.1),
        attention_dropout=enc_conf.get("attention_dropout_rate", 0.1),
        activation_dropout=enc_conf.get("dropout_rate", 0.1),
        activation_function="relu",
        kernel_size=enc_conf.get("kernel_size", 11),
    )

    # Text (LLM) config
    with open(os.path.join(qwen3_config_path, "config.json"), "r") as f:
        qwen3_cfg = json.load(f)
    text_config = Qwen3Config(**qwen3_cfg)

    config = FunAsrNanoConfig(
        encoder_config=encoder_config,
        text_config=text_config,
    )

    return config


def convert_checkpoint(
    output_path: str,
    model_path: str | None = None,
    model_id: str | None = None,
    push_to_hub: str | None = None,
):
    """Convert Fun-ASR-Nano checkpoint to HuggingFace format.

    Args:
        output_path: Output directory for the converted model.
        model_path: Path to the original Fun-ASR-Nano model directory (containing config.yaml, model.pt, Qwen3-0.6B/).
        model_id: Hub repo ID to download the original Fun-ASR-Nano checkpoint from, as an alternative to
            `model_path`.
        push_to_hub: Hub model ID to push the converted model to, or None to skip pushing.
    """
    if model_id is not None:
        print(f"Downloading original checkpoint from {model_id}...")
        model_path = snapshot_download(model_id)
    elif model_path is None:
        raise ValueError("Either `model_path` or `model_id` must be provided.")

    config_yaml_path = os.path.join(model_path, "config.yaml")
    checkpoint_path = os.path.join(model_path, "model.pt")
    qwen3_path = os.path.join(model_path, "Qwen3-0.6B")

    print(f"Building config from {config_yaml_path}...")
    config = build_config_from_yaml(config_yaml_path, qwen3_path)

    print(f"Loading original checkpoint from {checkpoint_path}...")
    original_state_dict = load_original_checkpoint(checkpoint_path)

    print("Converting state dict keys...")
    converted_state_dict, unconverted_keys = convert_state_dict(original_state_dict)

    if unconverted_keys:
        print(f"Skipping {len(unconverted_keys)} keys not used by the HF generation model (e.g. CTC branch):")
        for k in unconverted_keys[:20]:
            print(f"  - {k}")
        if len(unconverted_keys) > 20:
            print(f"  ... and {len(unconverted_keys) - 20} more")

    # Keep the serialized config dtype consistent with the bf16 weights.
    config.dtype = "bfloat16"
    print("Initializing HF model...")
    model = FunAsrNanoForConditionalGeneration(config).to(torch.bfloat16)

    print("Loading converted weights...")
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)

    if missing:
        print(f"Missing keys ({len(missing)}):")
        for k in missing[:20]:
            print(f"  - {k}")

    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:20]:
            print(f"  - {k}")
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint conversion mismatch: missing={missing}, unexpected={unexpected}")

    print(f"Saving model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    # Build and save the processor; this also writes the chat template to its own file in the checkpoint
    # (`chat_template.jinja`), so `processor.apply_chat_template` works without any Python-side default.
    print("Building processor (with chat template)...")

    tokenizer = AutoTokenizer.from_pretrained(
        qwen3_path,
        padding=True,
    )
    feature_extractor = FunAsrNanoFeatureExtractor()
    processor = FunAsrNanoProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )
    processor.save_pretrained(output_path)

    print("Done!")

    if push_to_hub:
        print(f"Pushing to hub: {push_to_hub}")
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Fun-ASR-Nano to HuggingFace format")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the original Fun-ASR-Nano model directory (alternative to --model_id)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Hugging Face Hub repo ID to download the original Fun-ASR-Nano checkpoint from",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push converted model to this HuggingFace Hub repo ID",
    )
    args = parser.parse_args()

    convert_checkpoint(
        output_path=args.output_path,
        model_path=args.model_path,
        model_id=args.model_id,
        push_to_hub=args.push_to_hub,
    )
