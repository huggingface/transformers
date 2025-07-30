# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

r"""Utility to convert Gemma models from Orbax to HF Transformers checkpoint.

python src/transformers/models/gemma3n/convert_gemma3n_weights.py \
    --variant='gemma3n_e4b' \
    --tokenizer_path="$HOME/tokenizers/gemma-3n-tokenizer.model" \
    --checkpoint_path="$HOME/checkpoints/gemma-3n-orbax/" \
    --output_path="$HOME/checkpoints/gemma-3n-safetensors/"
"""

import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import accelerate
import numpy as np
import torch
import tree
from absl import app, flags, logging
from orbax import checkpoint as obc

from transformers import (
    Gemma3nAudioConfig,
    Gemma3nAudioFeatureExtractor,
    Gemma3nConfig,
    Gemma3nForConditionalGeneration,
    Gemma3nProcessor,
    Gemma3nTextConfig,
    Gemma3nVisionConfig,
    GemmaTokenizerFast,
    GenerationConfig,
    SiglipImageProcessorFast,
)
from transformers.image_utils import PILImageResampling


# ==== Internal Constants and Classes ====


_CHAT_TEMPLATE = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'audio' -%}
                {{ '<audio_soft_token>' }}
            {%- elif item['type'] == 'image' -%}
                {{ '<image_soft_token>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""

_DTYPES = {"float32", "bfloat16", "float16"}

_SLIDING_WINDOW_PATTERN = 5

_AUDIO_ENCODER_PARAMETER = "AudioEncoder/encoder"
_AUDIO_ENCODER_CONFORMER = f"{_AUDIO_ENCODER_PARAMETER}/conformer/stacked_layers"
_AUDIO_ENCODER_SSCP = f"{_AUDIO_ENCODER_PARAMETER}/feature"

_TRANSFORMER_PARAMETER = "transformer"
_TRANSFORMER_ALTUP_PROJ = f"{_TRANSFORMER_PARAMETER}/altup_projection_"
_TRANSFORMER_ALTUP_UNEMB = f"{_TRANSFORMER_PARAMETER}/altup_unembed_projection_"
_TRANSFORMER_DECODER_BLOCK = f"{_TRANSFORMER_PARAMETER}/stacked_layers/attention_type_"
_TRANSFORMER_DECODER_BLOCK_LEN = len(_TRANSFORMER_DECODER_BLOCK)
_TRANSFORMER_EMBEDDER = f"{_TRANSFORMER_PARAMETER}/embedder"
_TRANSFORMER_FINAL_NORM = "transformer/final_norm"
_TRANSFORMER_POST_TRAINING_PREFIX = "rlx_networks/policy_network/"
_TRANSFORMER_POST_TRAINING_PREFIX_LEN = len(_TRANSFORMER_POST_TRAINING_PREFIX)

# _MOBILE_NET_CONFIG = Gemma3nVisionConfig.from_pretrained("")

_MOBILE_NET_PREFIX = "mobilenet"
_MOBILE_NET_TIMM_SUMMED_BLOCK_SIZES = [3, 8, 45, 84]
_MOBILE_NET_CONV = "block_group_conv2d_"
_MOBILE_NET_FIB = "block_group_fused_ib_"
_MOBILE_NET_MQA = "block_group_mmqa_"
_MOBILE_NET_MSFA = "block_adapter_"
_MOBILE_NET_UIB = "block_group_uib_"
_MOBILE_NET_UIB_HAS_DW_START = {
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (3, 0),
}
_MOBILE_NET_UIB_HAS_DW_MID = {
    (1, 0),
    (2, 0),
    (3, 0),
}

_VARIANT_GEMMA_3_2B = "gemma3n_e2b"
_VARIANT_GEMMA_3_4B = "gemma3n_e4b"
_VARIANTS: Mapping[str, Gemma3nConfig] = {
    _VARIANT_GEMMA_3_2B: Gemma3nConfig(
        text_config=Gemma3nTextConfig(
            intermediate_size=2048 * 4,
            num_hidden_layers=30,
            activation_sparsity_pattern=(0.95,) * 10 + (0.0,) * 20,
            num_kv_shared_layers=10,
        ),
        vision_config=Gemma3nVisionConfig(),
        audio_config=Gemma3nAudioConfig(),
    ),
    _VARIANT_GEMMA_3_4B: Gemma3nConfig(
        text_config=Gemma3nTextConfig(),
        vision_config=Gemma3nVisionConfig(),
        audio_config=Gemma3nAudioConfig(),
    ),
}


# ==== Flags ====

_AUDIO_DTYPE = flags.DEFINE_enum(
    name="audio_dtype",
    default="bfloat16",
    help="The floating point precision (aka dtype) of the model.",
    enum_values=_DTYPES,
)

_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path",
    default=None,
    help="Path to the Orbax checkpoint.",
    required=True,
)

_INCLUDE_CHAT_TEMPLATE = flags.DEFINE_bool(
    name="include_chat_template", default=False, help="If true, will save the default chat template with the tokenizer"
)

_OUTPUT_PATH = flags.DEFINE_string(
    name="output_path",
    default=None,
    help="Path to store the HF checkpoint.",
    required=True,
)

_TRANSFORMER_DTYPE = flags.DEFINE_enum(
    name="text_dtype",
    default="bfloat16",
    help="The floating point precision (aka dtype) of the model.",
    enum_values=_DTYPES,
)

_TOKENIZER_PATH = flags.DEFINE_string(
    name="tokenizer_path",
    default=None,
    help="Path to the SentencePiece model file.",
    required=True,
)

_VARIANT = flags.DEFINE_enum(
    name="variant",
    default=_VARIANT_GEMMA_3_4B,
    help="The model variant to convert.",
    enum_values=set(_VARIANTS.keys()),
)

_VERBOSE = flags.DEFINE_bool(
    name="verbose",
    default=False,
    help="If true, log the path, shape, and dtype of every converted layer.",
)

_VISION_DTYPE = flags.DEFINE_enum(
    name="vision_dtype",
    default="bfloat16",
    help="The floating point precision (aka dtype) of the model.",
    enum_values=_DTYPES,
)


def convert_audio_encoder_weights(
    config: Gemma3nAudioConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    if path.startswith(_AUDIO_ENCODER_CONFORMER):
        assert weights.shape[0] == config.conf_num_hidden_layers

        for i, matrix in enumerate(weights):
            if "fflayer_end" in path:
                base = f"conformer.{i}.ffw_layer_end"

                if path.endswith("ffn_layer1"):
                    converted_paths.append(f"{base}.ffw_layer_1.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("ffn_layer2"):
                    converted_paths.append(f"{base}.ffw_layer_2.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("post_layer_norm"):
                    converted_paths.append(f"{base}.post_layer_norm.weight")
                    converted_weights.append(matrix)
                elif path.endswith("pre_layer_norm"):
                    converted_paths.append(f"{base}.pre_layer_norm.weight")
                    converted_weights.append(matrix)
            elif "fflayer_start" in path:
                base = f"conformer.{i}.ffw_layer_start"

                if path.endswith("ffn_layer1"):
                    converted_paths.append(f"{base}.ffw_layer_1.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("ffn_layer2"):
                    converted_paths.append(f"{base}.ffw_layer_2.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("post_layer_norm"):
                    converted_paths.append(f"{base}.post_layer_norm.weight")
                    converted_weights.append(matrix)
                elif path.endswith("pre_layer_norm"):
                    converted_paths.append(f"{base}.pre_layer_norm.weight")
                    converted_weights.append(matrix)
            elif path.endswith("final_ln"):
                converted_paths.append(f"conformer.{i}.norm.weight")
                converted_weights.append(matrix)
            elif "lconv" in path:
                base = f"conformer.{i}.lconv1d"

                if path.endswith("conv_norm"):
                    converted_paths.append(f"{base}.conv_norm.weight")
                    converted_weights.append(matrix)
                elif path.endswith("depthwise_conv1d"):
                    converted_paths.append(f"{base}.depthwise_conv1d.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("linear_end"):
                    converted_paths.append(f"{base}.linear_end.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("linear_start"):
                    converted_paths.append(f"{base}.linear_start.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("ln"):
                    converted_paths.append(f"{base}.pre_layer_norm.weight")
                    converted_weights.append(matrix)
            elif "trans_atten" in path:
                base = f"conformer.{i}.attention"

                if param == "per_dim_scale":
                    converted_paths.append(f"{base}.attn.per_dim_scale")
                    converted_weights.append(matrix)

                if path.endswith("query_key_value_projection"):
                    converted_paths.extend(
                        [f"{base}.attn.q_proj.weight", f"{base}.attn.k_proj.weight", f"{base}.attn.v_proj.weight"]
                    )
                    converted_weights.extend(
                        [
                            m.reshape(config.hidden_size, config.hidden_size).transpose()
                            for m in matrix.transpose(1, 0, 2, 3)
                        ]
                    )
                elif path.endswith("pos_proj"):
                    converted_paths.append(f"{base}.attn.relative_position_embedding.pos_proj.weight")
                    converted_weights.append(matrix.reshape(config.hidden_size, config.hidden_size).transpose())
                elif path.endswith("post"):
                    converted_paths.append(f"{base}.post.weight")
                    converted_weights.append(matrix.transpose(2, 0, 1).reshape(config.hidden_size, config.hidden_size))
                elif path.endswith("post_norm"):
                    converted_paths.append(f"{base}.post_norm.weight")
                    converted_weights.append(matrix)
                elif path.endswith("pre_norm"):
                    converted_paths.append(f"{base}.pre_attn_norm.weight")
                    converted_weights.append(matrix)
    elif path.startswith(_AUDIO_ENCODER_SSCP):
        if path.endswith("input_proj"):
            converted_paths.append("subsample_conv_projection.input_proj_linear.weight")
            converted_weights.append(
                weights.transpose(2, 0, 1).reshape(config.hidden_size, config.sscp_conv_channel_size[1] ** 2)
            )
        elif "norm_" in path:
            index = int(path[-1])
            converted_paths.append(f"subsample_conv_projection.conv_{index}.norm.weight")
            converted_weights.append(weights)
        elif "subsampling_" in path:
            index = int(path[-1])
            converted_paths.append(f"subsample_conv_projection.conv_{index}.conv.weight")
            converted_weights.append(weights.transpose(3, 2, 0, 1))

    if (cpl := len(converted_paths)) != (cwl := len(converted_weights)):
        raise ValueError(
            "The `converted_paths` and `converted_weights` should be the same "
            f"length. Got {cpl} and {cwl}, respectively, for {path}."
        )

    return zip(converted_paths, converted_weights)


def convert_transformer_weights(
    config: Gemma3nTextConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    if path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
        path = path[_TRANSFORMER_POST_TRAINING_PREFIX_LEN:]

    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    if path.startswith(_TRANSFORMER_ALTUP_PROJ):
        index = int(path[-1])
        converted_paths.append(f"altup_projections.{index}.weight")
        converted_weights.append(weights.transpose())
    elif path.startswith(_TRANSFORMER_ALTUP_UNEMB):
        index = int(path[-1])
        converted_paths.append(f"altup_unembed_projections.{index}.weight")
        converted_weights.append(weights.transpose())
    elif path.startswith(_TRANSFORMER_DECODER_BLOCK):
        attention_type_index = int(path[_TRANSFORMER_DECODER_BLOCK_LEN])
        assert weights.shape[0] == config.num_hidden_layers / _SLIDING_WINDOW_PATTERN

        for i, matrix in enumerate(weights):
            layer_idx = _SLIDING_WINDOW_PATTERN * i + attention_type_index
            base_path = f"layers.{layer_idx}"

            if "altup" in path:
                altup_path = f"{base_path}.altup"

                if param == "correct_output_scale":
                    converted_paths.append(f"{altup_path}.correct_output_scale")
                    converted_weights.append(matrix)
                elif param == "correction_coefs":
                    converted_paths.append(f"{altup_path}.correction_coefs.weight")
                    converted_weights.append(matrix.transpose())
                elif param == "prediction_coefs":
                    converted_paths.append(f"{altup_path}.prediction_coefs.weight")
                    converted_weights.append(
                        np.clip(
                            matrix.reshape(config.altup_num_inputs, config.altup_num_inputs**2).transpose(),
                            -config.altup_coef_clip,
                            config.altup_coef_clip,
                        )
                    )

                if path.endswith("modality_router"):
                    converted_paths.append(f"{altup_path}.modality_router.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("router_norm_layer"):
                    converted_paths.append(f"{altup_path}.router_norm.weight")
                    converted_weights.append(matrix)
            elif path.endswith("attn/attn_vec_einsum"):
                converted_paths.append(f"{base_path}.self_attn.o_proj.weight")
                converted_weights.append(
                    matrix.transpose(2, 0, 1).reshape(config.hidden_size, config.num_attention_heads * config.head_dim)
                )
            elif path.endswith("attn/kv_einsum"):
                converted_paths.extend(
                    [
                        f"{base_path}.self_attn.k_proj.weight",
                        f"{base_path}.self_attn.v_proj.weight",
                    ]
                )
                k_proj_weights, v_proj_weights = matrix.transpose(0, 2, 1, 3)
                kv_proj_shape = (config.hidden_size, config.num_key_value_heads * config.head_dim)
                converted_weights.extend(
                    [
                        k_proj_weights.reshape(kv_proj_shape).transpose(),
                        v_proj_weights.reshape(kv_proj_shape).transpose(),
                    ]
                )
            elif path.endswith("attn/q_einsum"):
                converted_paths.append(f"{base_path}.self_attn.q_proj.weight")
                converted_weights.append(
                    matrix.transpose(1, 0, 2)
                    .reshape(config.hidden_size, config.num_attention_heads * config.head_dim)
                    .transpose()
                )
            elif path.endswith("attn/query_norm"):
                converted_paths.append(f"{base_path}.self_attn.q_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("attn/key_norm"):
                converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("laurel_block/linear_left"):
                converted_paths.append(f"{base_path}.laurel.linear_left.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("laurel_block/linear_right"):
                converted_paths.append(f"{base_path}.laurel.linear_right.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("mlp/gating_einsum"):
                converted_paths.extend([f"{base_path}.mlp.gate_proj.weight", f"{base_path}.mlp.up_proj.weight"])
                gate_proj_weight, up_proj_weight = matrix
                converted_weights.extend([gate_proj_weight, up_proj_weight])
            elif path.endswith("mlp/linear"):
                converted_paths.append(f"{base_path}.mlp.down_proj.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("per_layer_input_gate"):
                converted_paths.append(f"{base_path}.per_layer_input_gate.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("per_layer_projection"):
                converted_paths.append(f"{base_path}.per_layer_projection.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("post_attention_norm"):
                converted_paths.append(f"{base_path}.post_attention_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_ffw_norm"):
                converted_paths.append(f"{base_path}.post_feedforward_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_laurel_norm"):
                converted_paths.append(f"{base_path}.laurel.post_laurel_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_per_layer_input_norm"):
                converted_paths.append(f"{base_path}.post_per_layer_input_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("pre_attention_norm"):
                converted_paths.append(f"{base_path}.input_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("pre_ffw_norm"):
                converted_paths.append(f"{base_path}.pre_feedforward_layernorm.weight")
                converted_weights.append(matrix)
    elif path == _TRANSFORMER_EMBEDDER:
        if param == "input_embedding":
            converted_paths.append("embed_tokens.weight")
            # Gemma 3n model doesn't have soft tokens or "end of" tokens for images and audio in its input and output
            # embeddings, so we resize to avoid bugs observed with Mllama
            pre_expansion_embeddings = weights
            pad_token_slice = slice(config.pad_token_id, config.pad_token_id + 1)
            new_embeddings = np.repeat(pre_expansion_embeddings[pad_token_slice], 256, axis=0)
            weights = np.vstack([pre_expansion_embeddings, new_embeddings])
            converted_weights.append(weights)
        elif param == "per_layer_embeddings":
            converted_paths.append("embed_tokens_per_layer.weight")
            converted_weights.append(
                weights.reshape(
                    config.vocab_size_per_layer_input, config.num_hidden_layers * config.hidden_size_per_layer_input
                )
            )
    elif path.startswith(_TRANSFORMER_EMBEDDER):
        # TODO: ryanmullins - support multimodal norms and projections
        if path.endswith("per_layer_model_projection"):
            converted_paths.append("per_layer_model_projection.weight")
            converted_weights.append(
                weights.reshape(
                    config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input
                ).transpose()
            )
        elif path.endswith("per_layer_projection_norm"):
            converted_paths.append("per_layer_projection_norm.weight")
            converted_weights.append(weights)
    elif path == _TRANSFORMER_FINAL_NORM:
        converted_paths = ["norm.weight"]
        converted_weights = [weights]

    if (cpl := len(converted_paths)) != (cwl := len(converted_weights)):
        raise ValueError(
            "The `converted_paths` and `converted_weights` should be the same "
            f"length. Got {cpl} and {cwl}, respectively, for {path}."
        )

    return zip(converted_paths, converted_weights)


def convert_vision_weights(
    config: Gemma3nVisionConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    def generate_base_path(path: str, block_type: str) -> tuple[str, tuple[int, int]]:
        re_str = r"{}(\d+)/".format(block_type)
        re_pattern = re.compile(re_str)
        match = re.search(re_pattern, path).group(1)
        idx = abs(int(match)) - 1

        for block_idx, v in enumerate(_MOBILE_NET_TIMM_SUMMED_BLOCK_SIZES):
            if v > idx:
                offset = _MOBILE_NET_TIMM_SUMMED_BLOCK_SIZES[block_idx - 1] if block_idx > 0 else 0
                layer_idx = idx - offset
                return f"blocks.{block_idx}.{layer_idx}", (block_idx, layer_idx)

        raise ValueError(f"could not extract a base path from {path}")

    if _MOBILE_NET_MSFA in path:
        converted_path = "msfa"

        if "ffn/Normalize_0" in path:
            converted_path += ".ffn.pw_exp.bn.weight"
            converted_weight = weights
        elif "ffn/Normalize_1" in path:
            converted_path += ".ffn.pw_proj.bn.weight"
            converted_weight = weights
        elif "ffn/expand" in path:
            converted_path += ".ffn.pw_exp.conv.weight"
            converted_weight = weights.transpose()[:, :, None, None]
        elif "ffn/project" in path:
            converted_path += ".ffn.pw_proj.conv.weight"
            converted_weight = weights.transpose()[:, :, None, None]
        elif "Normalize_0" in path:
            converted_path += ".norm.weight"
            converted_weight = weights
    elif _MOBILE_NET_CONV in path:
        if "Conv_0" in path:
            converted_path = ("conv_stem.conv.weight", "conv_stem.conv.bias")
            converted_weight = weights.transpose(3, 2, 0, 1)
            converted_weight = (converted_weight, np.zeros(converted_weight.shape[0]))
        elif "Normalize_0" in path:
            converted_path = "conv_stem.bn.weight"
            converted_weight = weights
    elif _MOBILE_NET_FIB in path:
        converted_path, _ = generate_base_path(path, _MOBILE_NET_FIB)
        if "Normalize_0" in path:
            converted_path += ".bn1.weight"
            converted_weight = weights
        elif "Normalize_1" in path:
            converted_path += ".bn2.weight"
            converted_weight = weights
        elif "expand_conv" in path:
            converted_path += ".conv_exp.weight"
            converted_weight = weights.transpose(3, 2, 0, 1)
        else:
            converted_path += ".conv_pwl.weight"
            converted_weight = weights.transpose()[:, :, None, None]
    elif _MOBILE_NET_MQA in path:
        converted_path, _ = generate_base_path(path, _MOBILE_NET_MQA)

        if "LayerScale_0" in path:
            converted_path += ".layer_scale.gamma"
            converted_weight = weights
        elif "Normalize_0" in path:
            converted_path += ".norm.weight"
            converted_weight = weights
        elif "Normalize_1" in path:
            converted_path += ".attn.key.norm.weight"
            converted_weight = weights
        elif "Normalize_2" in path:
            converted_path += ".attn.value.norm.weight"
            converted_weight = weights
        elif "key_dwconv" in path:
            converted_path += ".attn.key.down_conv.weight"
            converted_weight = weights.transpose(3, 2, 0, 1)
        elif "key_proj" in path:
            converted_path += ".attn.key.proj.weight"
            converted_weight = weights.transpose()[:, :, None, None]
        elif "output_proj" in path:
            converted_path += ".attn.output.proj.weight"
            converted_weight = weights.transpose()[:, :, None, None]
        elif "query_proj" in path:
            converted_path += ".attn.query.proj.weight"
            converted_weight = weights.transpose()[:, :, None, None]
        elif "value_dwconv" in path:
            converted_path += ".attn.value.down_conv.weight"
            converted_weight = weights.transpose(3, 2, 0, 1)
        elif "value_proj" in path:
            converted_path += ".attn.value.proj.weight"
            converted_weight = weights.transpose()[:, :, None, None]
    elif _MOBILE_NET_UIB in path:
        converted_path, idx_key = generate_base_path(path, _MOBILE_NET_UIB)

        has_dw_start = idx_key in _MOBILE_NET_UIB_HAS_DW_START
        has_dw_mid = idx_key in _MOBILE_NET_UIB_HAS_DW_MID

        if "LayerScale_0" in path:
            converted_path += ".layer_scale.gamma"
            converted_weight = weights
        elif "Normalize_0" in path:
            converted_path += ".dw_start.bn.weight" if has_dw_start else ".pw_exp.bn.weight"
            converted_weight = weights
        elif "Normalize_1" in path:
            converted_path += ".pw_exp.bn.weight" if has_dw_start else ".pw_proj.bn.weight"
            converted_weight = weights
        elif "Normalize_2" in path:
            converted_path += ".dw_mid.bn.weight" if has_dw_mid else ".pw_proj.bn.weight"
            converted_weight = weights
        elif "Normalize_3" in path:
            converted_path += ".pw_proj.bn.weight"
            converted_weight = weights
        elif "expand" in path:
            converted_path += ".pw_exp.conv.weight"
            converted_weight = weights.transpose()[:, :, None, None]
        elif "middle_dwconv" in path:
            converted_path += ".dw_mid.conv.weight"
            converted_weight = weights.transpose(3, 2, 0, 1)
        elif "project" in path:
            converted_path += ".pw_proj.conv.weight"
            converted_weight = weights.transpose()[:, :, None, None]
        elif "start_dwconv" in path:
            converted_path += ".dw_start.conv.weight"
            converted_weight = weights.transpose(3, 2, 0, 1)

    if isinstance(converted_path, (tuple, list)):
        return zip(converted_path, converted_weight)
    else:
        return [(converted_path, converted_weight)]


def convert(checkpoint_path: str, config: Gemma3nConfig) -> dict[str, torch.Tensor]:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    checkpointer = obc.PyTreeCheckpointer()
    ckpt = checkpointer.restore(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    def update_tree(path: str, weights: np.ndarray, target_dtype: torch.dtype) -> None:
        hf_tree[path] = torch.from_numpy(weights.astype("float32")).type(target_dtype)
        if _VERBOSE.value:
            logging.info(
                "%s converted shape=%s with dtype=%s",
                path,
                weights.shape,
                target_dtype,
            )

    for (path, param), value in tree.flatten_with_path(ckpt):
        if param == "audio_input_embedding_extra":
            update_tree("model.embed_audio.embedding.weight", value, config.audio_config.dtype)
        elif path.endswith("audio_embedding_norm"):
            update_tree("model.embed_audio.hard_embedding_norm.weight", value, config.audio_config.dtype)
        elif path.endswith("audio_input_projection"):
            update_tree(
                "model.embed_audio.embedding_projection.weight", value.transpose(), config.audio_config.dtype
            )
        elif path.endswith("audio_soft_embedding_norm"):
            update_tree("model.embed_audio.soft_embedding_norm.weight", value, config.audio_config.dtype)
        elif param == "mm_input_embedding_extra":
            update_tree("model.embed_vision.embedding.weight", value, config.vision_config.dtype)
        elif path.endswith("mm_hard_embedding_norm"):
            update_tree("model.embed_vision.hard_embedding_norm.weight", value, config.vision_config.dtype)
        elif path.endswith("mm_input_projection"):
            update_tree(
                "model.embed_vision.embedding_projection.weight", value.transpose(), config.vision_config.dtype
            )
        elif path.endswith("mm_soft_embedding_norm"):
            update_tree("model.embed_vision.soft_embedding_norm.weight", value, config.vision_config.dtype)
        elif path.startswith(_TRANSFORMER_PARAMETER):
            for path, weights in convert_transformer_weights(config.text_config, path, param, value):
                update_tree(f"model.language_model.{path}", weights, config.text_config.dtype)
        elif _MOBILE_NET_PREFIX in path:
            mobilenet_prefix_idx = path.index(_MOBILE_NET_PREFIX)
            path = path[mobilenet_prefix_idx:]
            for path, weights in convert_vision_weights(config.vision_config, path, param, value):
                update_tree(f"model.vision_tower.timm_model.{path}", weights, config.vision_config.dtype)
        elif path.startswith(_AUDIO_ENCODER_PARAMETER):
            for path, weights in convert_audio_encoder_weights(config.audio_config, path, param, value):
                update_tree(f"model.audio_tower.{path}", weights, config.audio_config.dtype)

    hf_tree["lm_head.weight"] = hf_tree["model.language_model.embed_tokens.weight"]

    return hf_tree


def main(*args):
    del args

    output_path = _OUTPUT_PATH.value
    variant = _VARIANT.value

    config = _VARIANTS[variant]
    config.audio_config.dtype = getattr(torch, _AUDIO_DTYPE.value)
    config.text_config.dtype = getattr(torch, _TRANSFORMER_DTYPE.value)
    config.vision_config.dtype = getattr(torch, _VISION_DTYPE.value)
    if _INCLUDE_CHAT_TEMPLATE.value:
        # Chat template is included for instruction tuned models, which treat
        # both "<eos>" and "<end_of_turn>" as generation stoppers.
        config.eos_token_id = [1, 106]

    logging.info(
        "Converting Gemma 3 (%s) @ %s (language) and %s (vision)",
        variant,
        _TRANSFORMER_DTYPE.value,
        _VISION_DTYPE.value,
    )
    state_tree = convert(_CHECKPOINT_PATH.value, config)
    logging.info("Converted Gemma 3 (%s) state tree from Orbax to Hugging Face.", variant)

    with accelerate.init_empty_weights():
        model = Gemma3nForConditionalGeneration(config=config)

    model.load_state_dict(state_tree, assign=True, strict=True)
    logging.info(
        "Loaded Gemma 3 (%s) in Hugging Face Transformers as a %s instance.",
        variant,
        type(model).__name__,
    )
    model.save_pretrained(output_path, state_dict=state_tree, safe_serialization=True)
    logging.info(
        "Saved Gemma 3 (%s) to SafeTensors in %s using %s",
        variant,
        output_path,
        type(model).__name__,
    )
    del model
    del state_tree

    chat_template_kwargs = {"chat_template": _CHAT_TEMPLATE} if _INCLUDE_CHAT_TEMPLATE.value else {}

    tokenizer = GemmaTokenizerFast(
        _TOKENIZER_PATH.value,
        add_bos_token=True,
        extra_special_tokens={
            "image_token": "<image_soft_token>",  # Should be ID=262_145
            "boi_token": "<start_of_image>",  # Should be ID=255_999
            "eoi_token": "<end_of_image>",  # Should be ID=262_144
            "audio_token": "<audio_soft_token>",  # Should be ID=262_273
            "boa_token": "<start_of_audio>",  # Should be ID=256_000
            "eoa_token": "<end_of_audio>",  # Should be ID=262_272
        },
        **chat_template_kwargs,
    )
    tokenizer.save_pretrained(output_path)
    logging.info("Saved GemmaTokenizer for %s to %s", variant, output_path)

    feature_extractor = Gemma3nAudioFeatureExtractor()
    image_processor = SiglipImageProcessorFast(
        image_seq_length=256,
        image_mean=(0.5,) * 3,
        image_std=(0.5,) * 3,
        size={"height": 768, "width": 768},
        resample=PILImageResampling.BILINEAR,
        do_normalize=False,
    )
    processor = Gemma3nProcessor(
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        **chat_template_kwargs,
    )
    processor.save_pretrained(output_path)

    logging.info("Saved Gemma3nProcessor for %s to %s", variant, output_path)

    # NOTE: feature_extractor and image_processor both use the same filename, preprocessor_config.json, when saved to
    # disk, but the files are overwritten by processor.save_pretrained(). However, the configs can be unioned, saved,
    # and loaded from the same preprocessor_config.json file, so we do that explicitly here.
    feature_extractor_config = json.loads(feature_extractor.to_json_string())
    image_processor_config = json.loads(image_processor.to_json_string())
    preprocessor_config = {**feature_extractor_config, **image_processor_config}
    with open(os.path.join(output_path, "preprocessor_config.json"), "w", encoding="utf-8") as writer:
        writer.write(json.dumps(preprocessor_config, indent=2, sort_keys=True) + "\n")

    logging.info("Saved joint preprocessor_config.json for %s to %s", variant, output_path)

    del feature_extractor, image_processor, processor, tokenizer

    generation_config = GenerationConfig(
        pad_token_id=config.text_config.pad_token_id,
        bos_token_id=config.text_config.bos_token_id,
        eos_token_id=(
            [config.text_config.eos_token_id, 106] if _INCLUDE_CHAT_TEMPLATE.value else config.text_config.eos_token_id
        ),
        cache_implementation="hybrid",
        temperature=1.0,
        do_sample=True,
        top_k=64,
        top_p=0.95,
    )
    generation_config.save_pretrained(output_path)


if __name__ == "__main__":
    app.run(main)
