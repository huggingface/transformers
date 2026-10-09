# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import dataclasses
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    MossTranscribeDiarizeAudioConfig,
    MossTranscribeDiarizeConfig,
    MossTranscribeDiarizeFeatureExtractor,
    MossTranscribeDiarizeForConditionalGeneration,
    MossTranscribeDiarizeProcessor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# fmt: off
STATE_DICT_MAPPING = {
    r"^model\.whisper_encoder\.":     r"model.audio_tower.",
    r"^model\.vq_adaptor\.layers\.0\.": r"model.multi_modal_projector.linear_1.",
    r"^model\.vq_adaptor\.layers\.2\.": r"model.multi_modal_projector.linear_2.",
    r"^model\.vq_adaptor\.layers\.3\.": r"model.multi_modal_projector.norm.",
}
# fmt: on


# Add the original system prompt and hotword list to the system message.
CHAT_TEMPLATE = """{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- set ns = namespace(has_audio=false, text=none) -%}
        {%- set keyword_namespace = namespace(items=[]) -%}
        {%- for item in content -%}
            {%- if item.type == 'audio' or 'audio' in item or 'audio_url' in item -%}
                {{- '<|audio_start|><|audio_pad|><|audio_end|>\n' -}}
                {%- set ns.has_audio = true -%}
            {%- elif item.type == 'text' and ns.text is none -%}
                {%- set ns.text = item.text -%}
            {%- elif item.type == 'keywords' -%}
                {%- set keyword_namespace.items = keyword_namespace.items + item.keywords -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if ns.has_audio -%}
            {%- if ns.text -%}
                {{- '补充信息：' + ns.text + '\n\n' -}}
            {%- endif -%}
            {%- if keyword_namespace.items -%}
                {{- '热词列表：[' + keyword_namespace.items|join(', ') + ']\n\n' -}}
            {%- endif -%}
            {{- '请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。' -}}
        {%- elif ns.text -%}
            {{- ns.text -}}
        {%- endif -%}
    {%- endif -%}
{%- endmacro -%}
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- render_content(messages[0].content) + '\n\n' }}
    {%- else %}
        {{- 'You are a helpful assistant.\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + render_content(messages[0].content) + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- set content = render_content(message.content) %}
    {%- if ns.multi_step_tool and message.role == "user" and content is string and not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = render_content(message.content) %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


def map_old_key_to_new(old_key: str) -> str:
    new_key = old_key
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key = re.sub(pattern, replacement, new_key)
    return new_key


def convert_state_dict(original_state_dict: dict[str, Any]) -> dict[str, Any]:
    new_state_dict = {}
    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")
    return new_state_dict


def convert_config(original_config_dict: dict[str, Any]) -> MossTranscribeDiarizeConfig:
    config_dict = dict(original_config_dict)
    # `adaptor_input_dim` is derived.
    config_dict.pop("adaptor_input_dim", None)

    # The original `audio_config` is a full `WhisperConfig`, keep only the encoder fields
    attribute_map = MossTranscribeDiarizeAudioConfig.attribute_map
    encoder_fields = {field.name for field in dataclasses.fields(MossTranscribeDiarizeAudioConfig)}
    audio_config = {attribute_map.get(key, key): value for key, value in config_dict["audio_config"].items()}
    audio_config = {key: value for key, value in audio_config.items() if key in encoder_fields}
    audio_config["model_type"] = MossTranscribeDiarizeAudioConfig.model_type
    config_dict["audio_config"] = audio_config

    return MossTranscribeDiarizeConfig.from_dict(config_dict)


def load_original_state_dict(checkpoint_dir: Path) -> dict[str, Any]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]

    state_dict = {}
    for shard_file in shard_files:
        state_dict.update(load_file(checkpoint_dir / shard_file))
    return state_dict


def convert_checkpoint(checkpoint_dir, push_to_hub, bfloat16):
    dtype = torch.bfloat16 if bfloat16 else torch.float32
    checkpoint_dir = Path(checkpoint_dir)

    # 1) Load original state dict, config, generation config and processor.
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    original_state_dict = load_original_state_dict(checkpoint_dir)
    with open(checkpoint_dir / "config.json", "r", encoding="utf-8") as f:
        config = convert_config(json.load(f))
    processor = MossTranscribeDiarizeProcessor.from_pretrained(checkpoint_dir)
    # The original `preprocessor_config.json` declares `WhisperFeatureExtractor`, which lacks the chunking and `padding_mask`
    processor.feature_extractor = MossTranscribeDiarizeFeatureExtractor.from_pretrained(checkpoint_dir)
    processor.chat_template = CHAT_TEMPLATE

    processor.tokenizer.padding_side = "left"
    processor.tokenizer.init_kwargs["padding_side"] = "left"
    processor.tokenizer.init_kwargs["padding"] = True
    processor.tokenizer.init_kwargs["return_tensors"] = "pt"
    processor.feature_extractor.return_attention_mask = True

    # 2) Convert state dict to match HF model structure
    logger.info("Converting state dict")
    converted_state_dict = convert_state_dict(original_state_dict)

    # 3) Create model and load weights
    logger.info("Creating MossTranscribeDiarizeForConditionalGeneration model")
    model = MossTranscribeDiarizeForConditionalGeneration(config).to(dtype)

    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    # `lm_head.weight` is tied to `model.language_model.embed_tokens.weight` and isn't saved in the checkpoint.
    missing = [key for key in missing if key != "lm_head.weight"]
    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if len(missing) != 0:
        raise ValueError(f"Missing keys: {missing}")
    model.tie_weights()

    generation_config_path = checkpoint_dir / "generation_config.json"
    if generation_config_path.exists():
        model.generation_config = GenerationConfig.from_pretrained(checkpoint_dir)

    if push_to_hub:
        logger.info(f"Pushing to hub as {push_to_hub}")
        processor.push_to_hub(push_to_hub)
        model.push_to_hub(push_to_hub)

        gc.collect()
        logger.info("Verifying conversion by reloading model")
        AutoProcessor.from_pretrained(push_to_hub)
        AutoModelForCausalLM.from_pretrained(push_to_hub, dtype=torch.bfloat16, device_map="auto")
        logger.info("Model reloaded successfully!")
        logger.info("Conversion complete!")


"""
Conversion script to convert the original MOSS-Transcribe-Diarize checkpoint (Whisper encoder + VQAdaptor + Qwen3,
using the original repo's `trust_remote_code` module names) into an `MossTranscribeDiarizeForConditionalGeneration`
checkpoint using the natively-supported 🤗 Transformers modeling code.

1) download the original checkpoint (config, weights, tokenizer and processor files) locally, e.g.:
```bash
huggingface-cli download itazap/MOSS-Transcribe-Diarize-HF --local-dir /raid/moss_transcribe_diarize/original
```

2) run conversion with:
```bash
python src/transformers/models/moss_transcribe_diarize/convert_moss_transcribe_diarize_to_hf.py \
    --checkpoint_dir /raid/moss_transcribe_diarize/original \
    --push_to_hub itazap/MOSS-Transcribe-Diarize-HF
```

A checkpoint will be pushed to `itazap/MOSS-Transcribe-Diarize-HF` on the HF Hub.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        default=None,
        type=str,
        help="Path to a local directory with the original MOSS-Transcribe-Diarize checkpoint (config, safetensors "
        "weights, tokenizer and processor files).",
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--float32", action="store_true", help="Whether to use float32 precision. Default is bfloat16."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_dir,
        args.push_to_hub,
        bfloat16=not args.float32,
    )
