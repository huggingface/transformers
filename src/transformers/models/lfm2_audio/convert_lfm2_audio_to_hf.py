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

"""Convert Liquid Audio checkpoints to native Transformers format (local output by default)."""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    Lfm2AudioConfig,
    Lfm2AudioDetokenizer,
    Lfm2AudioFeatureExtractor,
    Lfm2AudioForConditionalGeneration,
    Lfm2AudioProcessor,
    Lfm2Config,
    ParakeetEncoderConfig,
)


DEFAULT_CHAT_TEMPLATE = r"""{{- bos_token -}}
{%- set ns = namespace(system_prompt="") -%}
{%- if messages and messages[0]["role"] == "system" -%}
    {%- set system_content = messages[0]["content"] -%}
    {%- if system_content is string -%}
        {%- set ns.system_prompt = system_content -%}
    {%- else -%}
        {%- for part in system_content -%}
            {%- if part["type"] == "text" -%}
                {%- set ns.system_prompt = ns.system_prompt + part["text"] -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if ns.system_prompt -%}
    {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
    {{- "<|im_start|>" + message["role"] + "\n" -}}
    {%- if message["content"] is string -%}
        {{- message["content"] -}}
    {%- else -%}
        {%- for part in message["content"] -%}
            {%- if part["type"] == "audio" -%}
                {{- "<|reserved_123|>" -}}
            {%- elif part["type"] == "text" -%}
                {{- part["text"] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n" -}}
{%- endif -%}"""


def convert_encoder_config(encoder):
    """Translate the original NeMo FastConformer configuration once, at conversion time."""
    for key, expected in {
        "subsampling": "dw_striding",
        "self_attention_model": "rel_pos",
        "conv_norm_type": "batch_norm",
    }.items():
        if encoder.get(key, expected) != expected:
            raise ValueError(f"Unsupported encoder {key}: {encoder[key]!r}")
    if encoder.get("reduction") is not None or encoder.get("reduction_factor", 1) != 1:
        raise ValueError("Additional encoder reduction is not supported.")
    return ParakeetEncoderConfig(
        hidden_size=encoder["d_model"],
        num_hidden_layers=encoder["n_layers"],
        num_attention_heads=encoder["n_heads"],
        intermediate_size=encoder["d_model"] * encoder.get("ff_expansion_factor", 4),
        num_mel_bins=encoder["feat_in"],
        subsampling_factor=encoder.get("subsampling_factor", 8),
        subsampling_conv_channels=encoder.get("subsampling_conv_channels", 256),
        conv_kernel_size=encoder.get("conv_kernel_size", 9),
        dropout=encoder.get("dropout_pre_encoder", 0.1),
        dropout_positions=encoder.get("dropout_emb", 0.0),
        layerdrop=0.0,
        activation_dropout=encoder.get("dropout", 0.1),
        attention_dropout=encoder.get("dropout_att", 0.1),
        max_position_embeddings=encoder.get("pos_emb_max_len", 5000),
        scale_input=encoder.get("xscaling", False),
    )


def convert_state_dict(state_dict, config):
    """Rename NeMo modules and split the depth attention's packed QKV projection."""
    replacements = {
        "audio_embedding.embedding.": "audio_embedding.",
        "conformer.pre_encode.conv.": "conformer.subsampling.layers.",
        "conformer.pre_encode.out.": "conformer.subsampling.linear.",
        ".self_attn.linear_q.": ".self_attn.q_proj.",
        ".self_attn.linear_k.": ".self_attn.k_proj.",
        ".self_attn.linear_v.": ".self_attn.v_proj.",
        ".self_attn.linear_out.": ".self_attn.o_proj.",
        ".self_attn.linear_pos.": ".self_attn.relative_k_proj.",
        ".self_attn.pos_bias_u": ".self_attn.bias_u",
        ".self_attn.pos_bias_v": ".self_attn.bias_v",
        ".conv.batch_norm.": ".conv.norm.",
        "audio_adapter.model.0.": "audio_adapter_norm.",
        "audio_adapter.model.1.": "audio_adapter_linear_1.",
        "audio_adapter.model.3.": "audio_adapter_linear_2.",
        ".operator.out_proj.": ".operator.o_proj.",
        ".operator.bounded_attention.q_layernorm.": ".operator.q_norm.",
        ".operator.bounded_attention.k_layernorm.": ".operator.k_norm.",
    }
    converted = {}
    depth = config.depth_config
    kv_size = depth.num_key_value_heads * (depth.dim // depth.num_attention_heads)
    for key, value in state_dict.items():
        if key in {
            "codebook_offsets",
            "audio_loss_weights",
            "audio_embedding.embedding_norm.weight",
            "audio_embedding.to_logits.weight",
        }:
            continue
        for old, new in replacements.items():
            key = key.replace(old, new)
        key = "model." + key
        if re.search(r"depthformer\.layers\.\d+\.operator\.qkv_proj\.weight$", key):
            for name, weight in zip(
                ("q_proj", "k_proj", "v_proj"), value.split((depth.dim, kv_size, kv_size)), strict=True
            ):
                converted[key.replace("qkv_proj", name)] = weight.contiguous()
        else:
            converted[key] = value
    return converted


def load_state_dict(checkpoint_path):
    index = checkpoint_path / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    state_dict = {}
    for filename in files:
        state_dict.update(load_file(str(checkpoint_path / filename)))
    return state_dict


def create_processor(checkpoint_path, frontend, decoder_model_id):
    for key, expected in {
        "normalize": "per_feature",
        "window": "hann",
        "log": True,
        "frame_splicing": 1,
        "pad_to": 0,
    }.items():
        if frontend.get(key, expected) != expected:
            raise ValueError(f"Unsupported frontend {key}: {frontend[key]!r}")
    sampling_rate = frontend["sample_rate"]
    feature_extractor = Lfm2AudioFeatureExtractor(
        feature_size=frontend["features"],
        sampling_rate=sampling_rate,
        hop_length=round(frontend["window_stride"] * sampling_rate),
        n_fft=frontend["n_fft"],
        win_length=round(frontend["window_size"] * sampling_rate),
        padding_value=frontend.get("pad_value", 0.0),
    )
    return Lfm2AudioProcessor(
        feature_extractor=feature_extractor,
        tokenizer=AutoTokenizer.from_pretrained(checkpoint_path),
        chat_template=DEFAULT_CHAT_TEMPLATE,
        decoder_model_id=decoder_model_id,
    )


def convert_checkpoint(checkpoint_path, output_dir, revision=None, push_to_hub=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        checkpoint_path = Path(snapshot_download(str(checkpoint_path), revision=revision))
    output_dir = Path(output_dir).resolve()
    if output_dir == checkpoint_path.resolve():
        raise ValueError("Output directory must differ from the original checkpoint.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    original_config = json.loads((checkpoint_path / "config.json").read_text())
    frontend = original_config.pop("preprocessor")
    original_config["encoder"] = convert_encoder_config(original_config["encoder"])
    config = Lfm2AudioConfig(**original_config, dtype="bfloat16")
    state_dict = convert_state_dict(load_state_dict(checkpoint_path), config)
    with torch.device("meta"):
        model = Lfm2AudioForConditionalGeneration(config)
    # The original safetensors omits shared depth embedding weights.
    for target, source in model.all_tied_weights_keys.items():
        if target not in state_dict and source in state_dict:
            state_dict[target] = state_dict[source]
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.tie_weights()
    model.save_pretrained(output_dir)

    detokenizer_path = checkpoint_path / "audio_detokenizer"
    if detokenizer_path.is_dir():
        detokenizer_config = Lfm2Config.from_pretrained(detokenizer_path)
        # The explicit sliding-window mask in Lfm2AudioDetokenizer handles these legacy layers.
        detokenizer_config.layer_types = [
            "full_attention" if layer_type == "sliding_attention" else layer_type
            for layer_type in detokenizer_config.layer_types
        ]
        detokenizer, loading_info = Lfm2AudioDetokenizer.from_pretrained(
            detokenizer_path, config=detokenizer_config, dtype=torch.float32, output_loading_info=True
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            if loading_info[key]:
                raise ValueError(f"Detokenizer has {key}: {loading_info[key]}")
        detokenizer.save_pretrained(output_dir / "audio_detokenizer")
    decoder_model_id = (push_to_hub or str(output_dir)) if detokenizer_path.is_dir() else None
    processor = create_processor(checkpoint_path, frontend, decoder_model_id)
    processor.save_pretrained(output_dir)
    if (checkpoint_path / "LICENSE").is_file():
        shutil.copy2(checkpoint_path / "LICENSE", output_dir / "LICENSE")
    # Upload is strictly opt-in; the default conversion only writes local files.
    if push_to_hub is not None:
        api = HfApi()
        api.create_repo(push_to_hub, exist_ok=True)
        api.upload_folder(repo_id=push_to_hub, folder_path=output_dir)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--push_to_hub",
        metavar="REPO_ID",
        default=None,
        help="Optional destination, e.g. kadirnar/LFM2.5-Audio-1.5B-hf",
    )
    args = parser.parse_args()
    convert_checkpoint(**vars(args))
