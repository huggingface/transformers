# Copyright 2025 BosonAI and The HuggingFace Inc. team. All rights reserved.
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
import glob
import re

import safetensors.torch
import torch
from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    DacFeatureExtractor,
    HiggsAudioV2Config,
    HiggsAudioV2ForConditionalGeneration,
    HiggsAudioV2Processor,
    HiggsAudioV2TokenizerModel,
)


CHAT_TEMPLATE = "{{- bos_token }}\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- if messages[0]['content'] is string %}\n        {%- set system_message = messages[0]['content']|trim %}\n    {%- elif messages[0]['content'] is iterable and messages[0]['content'][0]['type'] == 'text' %}\n        {%- set system_message = messages[0]['content'][0]['text']|trim %}\n    {%- else %}\n        {{- raise_exception(\"System message content must be a string or contain text type!\") }}\n    {%- endif %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {{- raise_exception(\"A system message is required but not provided!\") }}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{{- system_message }}\n\n{#- Check for scene message and handle it specially #}\n{%- if messages and messages[0]['role'] == 'scene' %}\n    {{- \"\\n\\n<|scene_desc_start|>\\n\" }}\n    {%- if messages[0]['content'] is string %}\n        {{- messages[0]['content'] | trim }}\n    {%- elif messages[0]['content'] is iterable %}\n        {%- for content_item in messages[0]['content'] %}\n            {%- if content_item['type'] == 'text' %}\n                {%- set text_content = content_item['text'] | trim %}\n                {{- text_content }}\n                {%- if loop.first and not loop.last %}\n                    {{- \"\\n\\n\" }}\n                {%- endif %}\n                {%- if not loop.first and not loop.last and messages[0]['content'][loop.index]['type'] != 'audio' %}\n                    {{- \"\\n\" }}\n                {%- endif %}\n            {%- elif content_item['type'] == 'audio' %}\n                {{- ' <|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>' }}\n                {%- if not loop.last %}\n                    {{- \"\\n\" }}\n                {%- endif %}\n            {%- endif %}\n        {%- endfor %}\n    {%- endif %}\n    {{- \"\\n<|scene_desc_end|>\" }}\n    {%- set messages = messages[1:] %}\n{%- endif %}\n\n{{- \"<|eot_id|>\" }}\n\n{#- Loop through all messages #}\n{%- for message in messages %}\n    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}\n    {%- if message['role'] == 'assistant' %}\n        {%- if message['content'] is not iterable or message['content'][0]['type'] != 'audio' %}\n            {{- raise_exception(\"Assistant messages must contain audio content only!\") }}\n        {%- endif %}\n        {{- '<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>' }}\n    {%- else %}\n        {%- if message['content'] is string %}\n            {{- message['content'] | trim }}\n        {%- elif message['content'] is iterable %}\n            {%- for content_item in message['content'] %}\n                {%- if content_item['type'] == 'text' %}\n                    {{- content_item['text'] | trim }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n    {%- endif %}\n    {{- '<|eot_id|>' }}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n<|audio_out_bos|>' }}\n{%- endif %}"

KEYS_TO_IGNORE = {
    "audio_codebook_weights",
    "rotary_emb.inv_freq",
    "rotary_emb.original_inv_freq",
}

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^audio_codebook_embeddings\.":          "model.embed_audio_tokens.embed_audio_tokens.",
    r"^(embed_tokens|layers|norm)":                                           r"model.\1",
    r"^audio_decoder_proj\.":                                                         "",
}
# fmt: on


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def convert_model(input_path_or_repo, revision=None):
    original_directory = snapshot_download(
        repo_id=input_path_or_repo, revision=revision, allow_patterns=["*.safetensors"]
    )

    # Load and merge original state dict
    original_state_dict = {}
    for path in sorted(glob.glob(f"{original_directory}/*.safetensors")):
        with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                original_state_dict[key] = f.get_tensor(key)

    # Merge PartiallyFrozenEmbedding weights
    if "embed_tokens.embedding_frozen.weight" in original_state_dict:
        original_state_dict["embed_tokens.weight"] = torch.cat(
            [
                original_state_dict.pop("embed_tokens.embedding_frozen.weight"),
                original_state_dict.pop("embed_tokens.embedding_trainable.weight"),
            ],
            dim=0,
        )

    # Merge PartiallyFrozenLinear weights for audio_lm_head
    if "audio_decoder_proj.audio_lm_head.linear_frozen.weight" in original_state_dict:
        original_state_dict["audio_decoder_proj.audio_lm_head.weight"] = torch.cat(
            [
                original_state_dict.pop("audio_decoder_proj.audio_lm_head.linear_frozen.weight"),
                original_state_dict.pop("audio_decoder_proj.audio_lm_head.linear_trainable.weight"),
            ],
            dim=0,
        )

    # Merge PartiallyFrozenLinear weights for text_lm_head
    if "audio_decoder_proj.text_lm_head.linear_frozen.weight" in original_state_dict:
        original_state_dict["audio_decoder_proj.text_lm_head.weight"] = torch.cat(
            [
                original_state_dict.pop("audio_decoder_proj.text_lm_head.linear_frozen.weight"),
                original_state_dict.pop("audio_decoder_proj.text_lm_head.linear_trainable.weight"),
            ],
            dim=0,
        )

    # Convert keys
    state_dict = {}
    for key, tensor in original_state_dict.items():
        if any(key.endswith(ignored) for ignored in KEYS_TO_IGNORE):
            continue
        state_dict[convert_key(key, ORIGINAL_TO_CONVERTED_KEY_MAPPING)] = tensor

    # Keep audio_decoder_proj-prefixed lm_head weights alongside the stripped versions
    if "audio_lm_head.weight" in state_dict:
        state_dict["audio_decoder_proj.audio_lm_head.weight"] = state_dict["audio_lm_head.weight"]
    if "text_lm_head.weight" in state_dict:
        state_dict["audio_decoder_proj.text_lm_head.weight"] = state_dict["text_lm_head.weight"]

    # Load into model (use_text_head=True to include text_lm_head)
    config = HiggsAudioV2Config(codebook_size=1026)
    with torch.device("meta"):
        model = HiggsAudioV2ForConditionalGeneration(config, use_text_head=True)
    model._keys_to_ignore_on_load_unexpected = [
        "audio_decoder_proj.audio_lm_head.weight",
        "audio_decoder_proj.text_lm_head.weight",
    ]
    model.load_state_dict(state_dict, strict=False, assign=True)

    model.generation_config._from_model_config = False
    model.generation_config.bos_token_id = 1
    model.generation_config.eos_token_id = 128009
    model.generation_config.pad_token_id = 128001
    model.generation_config.ras_win_len = 7
    model.generation_config.ras_win_max_num_repeat = 2
    model.generation_config.use_text_head = True

    print("Model converted successfully.")

    return model


def create_processor(
    input_path_or_repo, audio_tokenizer_path_or_repo, input_revision=None, audio_tokenizer_revision=None
):
    tokenizer = AutoTokenizer.from_pretrained(input_path_or_repo, revision=input_revision)
    tokenizer.pad_token = tokenizer.eos_token
    feature_extractor = DacFeatureExtractor(
        feature_size=1,
        hop_length=1,
        padding_side="right",
        padding_value=0.0,
        sampling_rate=24000,
        return_attention_mask=True,
    )
    audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        audio_tokenizer_path_or_repo, revision=audio_tokenizer_revision
    )

    processor = HiggsAudioV2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        chat_template=CHAT_TEMPLATE,
        audio_token="<|AUDIO_OUT|>",
        audio_bos_token="<|audio_out_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
    )
    print("Processor created successfully.")

    return processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_or_repo", default="bosonai/higgs-audio-v2-generation-3B-base")
    parser.add_argument("--input_revision", type=str, default="10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub_path", type=str, default=None)
    parser.add_argument("--audio_tokenizer_path_or_repo", default="eustlb/higgs-audio-v2-tokenizer")
    parser.add_argument("--audio_tokenizer_revision", type=str, default="todo")
    args = parser.parse_args()

    if args.output_dir is None and args.push_to_hub_path is None:
        raise ValueError("Either --output_dir or --push_to_hub_path must be provided.")

    model = convert_model(args.input_path_or_repo, revision=args.input_revision)
    processor = create_processor(
        args.input_path_or_repo,
        args.audio_tokenizer_path_or_repo,
        input_revision=args.input_revision,
        audio_tokenizer_revision=args.audio_tokenizer_revision,
    )

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Model and processor saved to {args.output_dir}")

    if args.push_to_hub_path is not None:
        model.push_to_hub(args.push_to_hub_path)
        processor.push_to_hub(args.push_to_hub_path)
        print(f"Model and processor pushed to {args.push_to_hub_path}")


if __name__ == "__main__":
    main()
