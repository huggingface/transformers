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

1) Download the original Fun-ASR-Nano checkpoint:

```bash
huggingface-cli download FunAudioLLM/Fun-ASR-Nano-2512 \
    --local-dir /path/to/Fun-ASR-Nano-2512
```

2) Run the conversion script:

```bash
python src/transformers/models/fun_asr_nano/convert_fun_asr_nano_to_hf.py \
    --model_path /path/to/Fun-ASR-Nano-2512 \
    --output_path ./fun-asr-nano-hf
```

3) Verify the converted checkpoint can be loaded:

```bash
python - <<'PY'
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "./fun-asr-nano-hf"
AutoProcessor.from_pretrained(model_id)
AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
PY
```

4) Optionally push the converted checkpoint to the Hub:

```bash
python src/transformers/models/fun_asr_nano/convert_fun_asr_nano_to_hf.py \
    --model_path /path/to/Fun-ASR-Nano-2512 \
    --output_path ./fun-asr-nano-hf \
    --push_to_hub \
    --hub_model_id your-username/Fun-ASR-Nano-2512-hf
```
"""

import argparse
import json
import os
import re

import torch
import yaml

from transformers import (
    AutoTokenizer,
    Qwen3Config,
)

from .configuration_fun_asr_nano import (
    FunAsrNanoConfig,
    FunAsrNanoEncoderConfig,
)
from .modeling_fun_asr_nano import FunAsrNanoForConditionalGeneration


# fmt: off
STATE_DICT_MAPPING = {
    r"^audio_encoder\.": r"model.audio_encoder.",
    r"^audio_adaptor\.": r"model.audio_adaptor.",
    # Keep lm_head.weight explicitly. Although tie_word_embeddings=True, this model load path
    # does not retie lm_head from the embeddings, and the source already stores lm_head == embeddings.
    # safetensors deduplicates the shared storage, so this adds no extra disk over the embeddings.
    r"^llm\.lm_head\.": r"lm_head.",
    r"^llm\.model\.": r"model.language_model.",
}
# fmt: on


# Chat template stored in the checkpoint so that `processor.apply_chat_template` works without any
# Python-side default. Audio elements are replaced with the `<|object_ref_start|>` placeholder token,
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
            "{% for content in message['content'] %}"
                "{% if content['type'] == 'audio' %}"
                    "<|object_ref_start|>"
                "{% elif content['type'] == 'text' %}"
                    "{{ content['text'] }}"
                "{% endif %}"
            "{% endfor %}"
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
    """Map an original FunASR checkpoint key to the HF (split) layout.

    The HF model is split into a base [`FunAsrNanoModel`] (holding the audio encoder, adaptor and the *headless*
    language model) plus a separate `lm_head`, mirroring AudioFlamingo3 / Voxtral. The mapping is therefore:

        audio_encoder.*       -> model.audio_encoder.*
        audio_adaptor.*       -> model.audio_adaptor.*
        llm.model.*           -> model.language_model.*
        llm.lm_head.weight    -> lm_head.weight

    Returns `None` for keys that are not used by the generation path (e.g. the CTC / timestamp branch).
    """
    for pattern, replacement in STATE_DICT_MAPPING.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return None


def build_config_from_yaml(config_yaml_path: str, qwen3_config_path: str) -> FunAsrNanoConfig:
    """Build HF config from original config.yaml."""
    with open(config_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Audio encoder config (standalone encoder model -> standalone config, Parakeet-style).
    enc_conf = cfg.get("audio_encoder_conf", {})
    audio_encoder_config = FunAsrNanoEncoderConfig(
        input_size=enc_conf.get("input_layer_size", 560),  # 80 * 7 (lfr_m)
        output_dim=enc_conf.get("output_size", 512),
        num_attention_heads=enc_conf.get("attention_heads", 4),
        intermediate_size=enc_conf.get("linear_units", 2048),
        encoder_layers=enc_conf.get("num_blocks", 50),
        tp_blocks=enc_conf.get("tp_blocks", 20),
        dropout=enc_conf.get("dropout_rate", 0.1),
        attention_dropout=enc_conf.get("attention_dropout_rate", 0.0),
        kernel_size=enc_conf.get("kernel_size", 11),
        sanm_shift=enc_conf.get("sanm_shfit", 0),
    )

    # Adaptor params live directly on the main config (the adaptor is not a standalone model).
    adp_conf = cfg.get("audio_adaptor_conf", {})

    # Text (LLM) config
    with open(os.path.join(qwen3_config_path, "config.json"), "r") as f:
        qwen3_cfg = json.load(f)
    text_config = Qwen3Config(**qwen3_cfg)

    config = FunAsrNanoConfig(
        audio_encoder_config=audio_encoder_config,
        text_config=text_config,
        adaptor_downsample_rate=adp_conf.get("downsample_rate", 1),
        adaptor_intermediate_size=adp_conf.get("ffn_dim", 2048),
        adaptor_num_hidden_layers=adp_conf.get("n_layer", 2),
        adaptor_num_attention_heads=8,
        adaptor_dropout=0.0,
    )

    return config


def convert_checkpoint(
    model_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """Convert Fun-ASR-Nano checkpoint to HuggingFace format.

    Args:
        model_path: Path to the original Fun-ASR-Nano model directory (containing config.yaml, model.pt, Qwen3-0.6B/).
        output_path: Output directory for the converted model.
        push_to_hub: Whether to push the converted model to HuggingFace Hub.
        hub_model_id: Hub model ID for pushing.
    """
    config_yaml_path = os.path.join(model_path, "config.yaml")
    checkpoint_path = os.path.join(model_path, "model.pt")
    qwen3_path = os.path.join(model_path, "Qwen3-0.6B")

    print(f"Building config from {config_yaml_path}...")
    config = build_config_from_yaml(config_yaml_path, qwen3_path)

    print(f"Loading original checkpoint from {checkpoint_path}...")
    original_state_dict = load_original_checkpoint(checkpoint_path)

    print("Converting state dict keys...")
    converted_state_dict = {}
    unconverted_keys = []

    for key, value in original_state_dict.items():
        new_key = convert_key(key)
        if new_key is not None:
            # Cast every weight to bfloat16 so the checkpoint is dtype-consistent with the
            # bf16 LLM (and the old hub layout); the source SAN-M encoder/adaptor are stored in F32.
            converted_state_dict[new_key] = value.to(torch.bfloat16)
        else:
            # CTC / timestamp branch is not used for the generation path and is intentionally dropped.
            unconverted_keys.append(key)

    if unconverted_keys:
        print(f"Skipping {len(unconverted_keys)} keys not used by the HF model (e.g. CTC branch):")
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

    print(f"Saving model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    # Build and save the processor; this also writes the chat template to its own file in the checkpoint
    # (`chat_template.jinja`), so `processor.apply_chat_template` works without any Python-side default.
    print("Building processor (with chat template)...")
    from .feature_extraction_fun_asr_nano import FunAsrNanoFeatureExtractor
    from .processing_fun_asr_nano import FunAsrNanoProcessor

    tokenizer = AutoTokenizer.from_pretrained(qwen3_path)
    feature_extractor = FunAsrNanoFeatureExtractor()
    processor = FunAsrNanoProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
        audio_downsample_rate=config.adaptor_downsample_rate,
    )
    processor.save_pretrained(output_path)

    print("Done!")

    if push_to_hub and hub_model_id:
        print(f"Pushing to hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        processor.push_to_hub(hub_model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Fun-ASR-Nano to HuggingFace format")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the original Fun-ASR-Nano model directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID",
    )
    args = parser.parse_args()

    convert_checkpoint(
        model_path=args.model_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
