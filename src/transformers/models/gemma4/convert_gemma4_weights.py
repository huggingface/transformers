# Copyright 2026 Google Inc. HuggingFace Inc. team. All rights reserved.
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

python src/transformers/models/gemma4/convert_gemma4_weights.py \
    --variant='gemma-4-e2b' \
    --include_chat_template \
    --include_response_schema \
    --tokenizer_path="$HOME/tokenizers/gemma4/gemma4_cleaned_262144.model" \
    --checkpoint_path="$HOME/gemma4/checkpoints/gemma_e2b_it_orbax" \
    --output_path="$HOME/gemma4/checkpoints/gemma_e2b_it_safetensors"
"""

import ast
import json
import os
import pathlib
from collections.abc import Iterable, Mapping
from typing import Any

import accelerate
import jax
import numpy as np
import torch
import tree
from absl import app, flags, logging
from jax.sharding import SingleDeviceSharding
from orbax import checkpoint as obc
from orbax.checkpoint import args as obc_args
from orbax.checkpoint import type_handlers

from transformers import (
    Gemma4AudioConfig,
    Gemma4AudioFeatureExtractor,
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4ImageProcessor,
    Gemma4Processor,
    Gemma4TextConfig,
    Gemma4VideoProcessor,
    Gemma4VisionConfig,
    GemmaTokenizer,
    GenerationConfig,
    RopeParameters,
)
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor
from transformers.utils.hub import cached_file


# ==== Internal Constants and Classes ====

# The correct chat templates were already uploaded to those 2 repos, so download from there
_CHAT_TEMPLATE = pathlib.Path(cached_file("gg-hf-gg/gemma-4-E4B-it", "chat_template.jinja")).read_text()
_CHAT_TEMPLATE_LARGE = pathlib.Path(cached_file("gg-hf-gg/gemma-4-31B-it", "chat_template.jinja")).read_text()

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "thinking": {
            "type": "string",
        },
        "content": {
            "type": "string",
        },
        "tool_calls": {
            "x-regex-iterator": r"<\|tool_call>(.*?)<tool_call\|>",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "x-regex": r"call\:(?P<name>\w+)(?P<arguments>\{.*\})",
                        "properties": {
                            "name": {
                                "type": "string",
                            },
                            "arguments": {
                                "type": "object",
                                "x-parser": "gemma4-tool-call",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
    "x-regex": r"(\<\|channel\>thought\n(?P<thinking>.*?)\<channel\|\>)?(?P<tool_calls>\<\|tool_call\>.*\<tool_call\|\>)?(?P<content>(?:(?!\<turn\|\>)(?!\<\|tool_response\>).)+)?(?:\<turn\|\>|\<\|tool_response\>)?",
}

_DTYPES = {"float32", "bfloat16", "float16"}

_SLIDING_WINDOW_PATTERN = 6

_AUDIO_ENCODER_PARAMETER = "AudioEncoder/encoder"
_AUDIO_ENCODER_CONFORMER = f"{_AUDIO_ENCODER_PARAMETER}/conformer/stacked_layers"
_AUDIO_ENCODER_SSCP = f"{_AUDIO_ENCODER_PARAMETER}/feature"

_TRANSFORMER_PARAMETER = "transformer"
_TRANSFORMER_DECODER_BLOCK = f"{_TRANSFORMER_PARAMETER}/stacked_layers/attention_type_"
_TRANSFORMER_DECODER_BLOCK_LEN = len(_TRANSFORMER_DECODER_BLOCK)
_TRANSFORMER_EMBEDDER = f"{_TRANSFORMER_PARAMETER}/embedder"
_TRANSFORMER_FINAL_NORM = "transformer/final_norm"
_TRANSFORMER_POST_TRAINING_PREFIX = "rlx_networks/policy_network/"
_TRANSFORMER_POST_TRAINING_PREFIX_LEN = len(_TRANSFORMER_POST_TRAINING_PREFIX)

_VISION_ENCODER_PARAMETER = "PatchInputVariablePoolingEncoder_0"
_VISION_ENCODER_VIT_PARAMETER = f"{_VISION_ENCODER_PARAMETER}/_model/vit"
_VISION_ENCODER_ENTRY = f"{_VISION_ENCODER_VIT_PARAMETER}/entry"
_VISION_ENCODER_EXIT = f"{_VISION_ENCODER_VIT_PARAMETER}/exit"
_VISION_ENCODER_STANDARDIZE = f"{_VISION_ENCODER_PARAMETER}/standardize"
_VISION_ENCODER_TRANSFORMER = f"{_VISION_ENCODER_VIT_PARAMETER}/transformer/stacked_layers/block"

_VARIANT_GEMMA_4_E2B = "gemma-4-e2b"
_VARIANT_GEMMA_4_E4B = "gemma-4-e4b"
_VARIANT_GEMMA_4_26B_A4B = "gemma-4-26b-a4b"
_VARIANT_GEMMA_4_31B = "gemma-4-31b"

_LARGE_MODEL_VARIANTS = {
    _VARIANT_GEMMA_4_31B,
    _VARIANT_GEMMA_4_26B_A4B,
}

_ON_DEVICE_VISION_CONFIG = Gemma4VisionConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=16,
    num_attention_heads=12,
    num_key_value_heads=12,
    head_dim=64,
    global_head_dim=64,
    default_output_length=280,
    pooling_kernel_size=3,
    use_clipped_linears=True,
)

_LARGE_MODEL_VISION_CONFIG = Gemma4VisionConfig(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=72,
    global_head_dim=72,
    default_output_length=280,
    pooling_kernel_size=3,
    use_clipped_linears=False,
    standardize=True,
)

_DEFAULT_AUDIO_CONFIG = Gemma4AudioConfig()

_ROPE_PARAMS: dict[str, RopeParameters] = {
    "full_attention": RopeParameters(
        rope_theta=1_000_000.0,
        rope_type="proportional",
        partial_rotary_factor=0.25,
    ),
    "sliding_attention": RopeParameters(
        rope_theta=10000.0,
        rope_type="default",
    ),
}

_DEFAULT_LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]
_GEMMA_4_E2B_LAYER_TYPES = ["sliding_attention"] * 4 + ["full_attention"]


_VARIANTS: Mapping[str, Gemma4Config] = {
    _VARIANT_GEMMA_4_E2B: Gemma4Config(
        text_config=Gemma4TextConfig(
            hidden_size=1536,
            hidden_size_per_layer_input=256,
            intermediate_size=4 * 1536,
            num_hidden_layers=35,
            layer_types=_GEMMA_4_E2B_LAYER_TYPES * 7,
            num_attention_heads=8,
            num_key_value_heads=1,
            num_global_key_value_heads=None,
            attention_k_eq_v=False,
            use_bidirectional_attention=None,
            num_kv_shared_layers=20,
            use_double_wide_mlp=True,
            final_logit_softcapping=30.0,
            rope_parameters=_ROPE_PARAMS,
        ),
        vision_config=_ON_DEVICE_VISION_CONFIG,
        audio_config=_DEFAULT_AUDIO_CONFIG,
        vision_soft_tokens_per_image=280,
    ),
    _VARIANT_GEMMA_4_E4B: Gemma4Config(
        text_config=Gemma4TextConfig(
            hidden_size=2560,
            hidden_size_per_layer_input=256,
            intermediate_size=4 * 2560,
            num_hidden_layers=42,
            layer_types=_DEFAULT_LAYER_TYPES * 7,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_global_key_value_heads=None,
            global_head_dim=512,  # Global attention layers use 512-dim heads
            attention_k_eq_v=False,
            use_bidirectional_attention=None,
            num_kv_shared_layers=18,
            final_logit_softcapping=30.0,
            rope_parameters=_ROPE_PARAMS,
        ),
        vision_config=_ON_DEVICE_VISION_CONFIG,
        vision_soft_tokens_per_image=280,
        audio_config=_DEFAULT_AUDIO_CONFIG,
    ),
    _VARIANT_GEMMA_4_31B: Gemma4Config(
        text_config=Gemma4TextConfig(
            hidden_size=5376,
            hidden_size_per_layer_input=0,
            intermediate_size=4 * 5376,
            num_hidden_layers=60,
            layer_types=_DEFAULT_LAYER_TYPES * 10,
            num_attention_heads=32,
            num_key_value_heads=16,
            num_global_key_value_heads=4,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            num_kv_shared_layers=0,
            sliding_window=1024,
            final_logit_softcapping=30.0,
            rope_parameters=_ROPE_PARAMS,
            max_position_embeddings=262_144,
        ),
        vision_config=_LARGE_MODEL_VISION_CONFIG,
        vision_soft_tokens_per_image=280,
    ),
    _VARIANT_GEMMA_4_26B_A4B: Gemma4Config(
        text_config=Gemma4TextConfig(
            hidden_size=2816,
            hidden_size_per_layer_input=0,
            intermediate_size=2112,  # Shared expert FFW
            num_hidden_layers=30,
            layer_types=_DEFAULT_LAYER_TYPES * 5,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_global_key_value_heads=2,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            num_kv_shared_layers=0,
            enable_moe_block=True,
            num_experts=128,
            moe_intermediate_size=704,
            top_k_experts=8,
            sliding_window=1024,
            final_logit_softcapping=30.0,
            rope_parameters=_ROPE_PARAMS,
            max_position_embeddings=262_144,
        ),
        vision_config=_LARGE_MODEL_VISION_CONFIG,
        vision_soft_tokens_per_image=280,
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

_INCLUDE_RESPONSE_SCHEMA = flags.DEFINE_bool(
    name="include_response_schema",
    default=False,
    help="If true, will save the default response schema with the tokenizer",
)

_OUTPUT_PATH = flags.DEFINE_string(
    name="output_path",
    default=None,
    help="Path to store the HF checkpoint.",
    required=True,
)

_TEXT_DTYPE = flags.DEFINE_enum(
    name="text_dtype",
    default="bfloat16",
    help="The floating point precision (aka dtype) of the model.",
    enum_values=_DTYPES,
)

_TEXT_ONLY = flags.DEFINE_bool(
    name="text_only",
    default=False,
    help="If True, saves a Gemma4ForCasualLM model instead of a Gemma4ForConditionalGeneration model.",
)

_TOKENIZER_PATH = flags.DEFINE_string(
    name="tokenizer_path",
    default=None,
    help="Path to the SentencePiece model file.",
    required=True,
)

_VARIANT = flags.DEFINE_enum(
    name="variant",
    default=None,
    help="The model variant to convert.",
    enum_values=set(_VARIANTS.keys()),
    required=True,
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
    config,  # Gemma4AudioConfig
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    # The conformer uses its own internal dimension (1024 by default via conf_hidden_size).
    # Since we now use the default hidden_size=1024 (same as conf_hidden_size),
    # we use config.conf_hidden_size for reshaping conformer weights.

    if path.startswith(_AUDIO_ENCODER_CONFORMER):
        assert weights.shape[0] == config.num_hidden_layers

        for i, matrix in enumerate(weights):
            if "fflayer_end" in path:
                base = f"layers.{i}.feed_forward2"

                if path.endswith("ffn_layer1/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.ffw_layer_1.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                elif path.endswith("ffn_layer2/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.ffw_layer_2.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                elif path.endswith("ffn_layer1"):
                    converted_paths.append(f"{base}.ffw_layer_1.linear.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("ffn_layer2"):
                    converted_paths.append(f"{base}.ffw_layer_2.linear.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("post_layer_norm"):
                    converted_paths.append(f"{base}.post_layer_norm.weight")
                    converted_weights.append(matrix)
                elif path.endswith("pre_layer_norm"):
                    converted_paths.append(f"{base}.pre_layer_norm.weight")
                    converted_weights.append(matrix)
            elif "fflayer_start" in path:
                base = f"layers.{i}.feed_forward1"

                if path.endswith("ffn_layer1/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.ffw_layer_1.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                elif path.endswith("ffn_layer2/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.ffw_layer_2.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                elif path.endswith("ffn_layer1"):
                    converted_paths.append(f"{base}.ffw_layer_1.linear.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("ffn_layer2"):
                    converted_paths.append(f"{base}.ffw_layer_2.linear.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("post_layer_norm"):
                    converted_paths.append(f"{base}.post_layer_norm.weight")
                    converted_weights.append(matrix)
                elif path.endswith("pre_layer_norm"):
                    converted_paths.append(f"{base}.pre_layer_norm.weight")
                    converted_weights.append(matrix)
            elif path.endswith("final_ln"):
                converted_paths.append(f"layers.{i}.norm_out.weight")
                converted_weights.append(matrix)
            elif "lconv" in path:
                base = f"layers.{i}.lconv1d"

                if path.endswith("linear_start/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.linear_start.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                elif path.endswith("linear_end/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.linear_end.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                elif path.endswith("conv_norm"):
                    converted_paths.append(f"{base}.conv_norm.weight")
                    converted_weights.append(matrix)
                elif path.endswith("depthwise_conv1d"):
                    converted_paths.append(f"{base}.depthwise_conv1d.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("linear_end"):
                    converted_paths.append(f"{base}.linear_end.linear.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("linear_start"):
                    converted_paths.append(f"{base}.linear_start.linear.weight")
                    converted_weights.append(matrix.transpose())
                elif path.endswith("ln"):
                    converted_paths.append(f"{base}.pre_layer_norm.weight")
                    converted_weights.append(matrix)
            elif "trans_atten" in path:
                base = f"layers.{i}"

                if param == "per_dim_scale":
                    converted_paths.append(f"{base}.self_attn.per_dim_scale")
                    converted_weights.append(matrix)

                if path.endswith("query_key_value_projection/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.self_attn.q_proj.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                    converted_paths.append(f"{base}.self_attn.k_proj.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                    converted_paths.append(f"{base}.self_attn.v_proj.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)
                elif path.endswith("post/ClippedEinsum_0"):
                    converted_paths.append(f"{base}.self_attn.post.{param.removeprefix('clip_')}")
                    converted_weights.append(matrix)

                if path.endswith("query_key_value_projection"):
                    converted_paths.extend(
                        [
                            f"{base}.self_attn.q_proj.linear.weight",
                            f"{base}.self_attn.k_proj.linear.weight",
                            f"{base}.self_attn.v_proj.linear.weight",
                        ]
                    )
                    converted_weights.extend(
                        [
                            m.reshape(config.hidden_size, config.hidden_size).transpose()
                            for m in matrix.transpose(1, 0, 2, 3)
                        ]
                    )
                elif path.endswith("pos_proj"):
                    converted_paths.append(f"{base}.self_attn.relative_k_proj.weight")
                    converted_weights.append(matrix.reshape(config.hidden_size, config.hidden_size).transpose())
                elif path.endswith("post"):
                    converted_paths.append(f"{base}.self_attn.post.linear.weight")
                    converted_weights.append(matrix.transpose(2, 0, 1).reshape(config.hidden_size, config.hidden_size))
                elif path.endswith("post_norm"):
                    converted_paths.append(f"{base}.norm_post_attn.weight")
                    converted_weights.append(matrix)
                elif path.endswith("pre_norm"):
                    converted_paths.append(f"{base}.norm_pre_attn.weight")
                    converted_weights.append(matrix)
    elif path.startswith(_AUDIO_ENCODER_SSCP):
        if path.endswith("input_proj"):
            converted_paths.append("subsample_conv_projection.input_proj_linear.weight")
            converted_weights.append(
                weights.transpose(2, 0, 1).reshape(config.hidden_size, config.subsampling_conv_channels[1] ** 2)
            )
        elif "norm_" in path:
            index = int(path[-1])
            converted_paths.append(f"subsample_conv_projection.layer{index}.norm.weight")
            converted_weights.append(weights)
        elif "subsampling_" in path:
            index = int(path[-1])
            converted_paths.append(f"subsample_conv_projection.layer{index}.conv.weight")
            converted_weights.append(weights.transpose(3, 2, 0, 1))

    elif path.endswith("output_projection"):
        if param == "kernel":
            converted_paths.append("output_proj.weight")
            converted_weights.append(weights.transpose())
        elif param == "bias":
            converted_paths.append("output_proj.bias")
            converted_weights.append(weights)

    if (cpl := len(converted_paths)) != (cwl := len(converted_weights)):
        raise ValueError(
            "The `converted_paths` and `converted_weights` should be the same "
            f"length. Got {cpl} and {cwl}, respectively, for {path}."
        )

    return zip(converted_paths, converted_weights)


def convert_vision_encoder_weights(
    config,  # Gemma4VisionConfig
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    """Convert vision encoder weights from JAX checkpoint to HuggingFace format.

    Args:
        config: Vision config with num_hidden_layers, hidden_size, etc.
        path: Path in the JAX checkpoint (e.g., "VisionEncoder_0/entry/input_projection")
        param: Parameter type (e.g., "w", "scale", "pos_emb")
        weights: NumPy array of weights

    Returns:
        Iterable of (hf_path, converted_weights) tuples
    """
    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    # Patch Embedder - Entry
    # TODO(philculliton): These do not appear to be used currently - they should be loaded by Gemma4VisionPatchEmbedder, by all appearances, but are not currently.
    if path == f"{_VISION_ENCODER_ENTRY}/input_projection":
        if param == "w":
            converted_paths.append("patch_embedder.input_proj.weight")
            # Shape: (768, 768) -> transpose to (768, 768) for nn.Linear
            converted_weights.append(weights.transpose())
    elif path == _VISION_ENCODER_ENTRY:
        if param == "pos_emb":
            converted_paths.append("patch_embedder.position_embedding_table")
            # Shape: (10240, 2, 768) -> transpose to (2, 10240, 768)
            converted_weights.append(weights.transpose(1, 0, 2))

    # Pooler - Exit: convert the learnable scale parameter for vision output scaling
    elif path == _VISION_ENCODER_EXIT:
        if param == "scale":
            converted_paths.append("pooler.scale")
            # JAX shape is (1, 1, d_model), keep as-is for nn.Parameter
            converted_weights.append(weights)

    elif path == _VISION_ENCODER_STANDARDIZE:
        if param == "bias":
            converted_paths.append("std_bias")
            converted_weights.append(weights)
        else:
            converted_paths.append("std_scale")
            converted_weights.append(weights)

    # Transformer Layers (stacked format)
    elif path.startswith(_VISION_ENCODER_TRANSFORMER):
        # All vision transformer layers are stacked in dimension 0
        num_layers = weights.shape[0]
        assert num_layers == config.num_hidden_layers, f"Expected {config.num_hidden_layers} layers, got {num_layers}"

        for i, matrix in enumerate(weights):
            base_path = f"encoder.layers.{i}"

            # Handle clipped einsum states (`ClippedEinsum_0` target paths).
            if path.endswith("attn_vec_einsum/ClippedEinsum_0"):
                converted_paths.append(f"{base_path}.self_attn.o_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
            if path.endswith("kv_einsum/ClippedEinsum_0"):
                # NOTE: In JAX reference implementations of Gemma, k_proj and v_proj are performed with a single einsum
                # operation. We split this into two operations in Transformers, but they are passed the same input and
                # share the same activation bounds for clipping, thus we re-use the same matrix for both.
                converted_paths.append(f"{base_path}.self_attn.k_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
                converted_paths.append(f"{base_path}.self_attn.v_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
            if path.endswith("q_einsum/ClippedEinsum_0"):
                converted_paths.append(f"{base_path}.self_attn.q_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
            if path.endswith("gating_einsum/ClippedEinsum_0"):
                # NOTE: In JAX reference implementations of Gemma, gate_proj and up_proj are performed with a single
                # einsum operation. We split this into two operations in Transformers, but they are passed the same
                # input and share the same activation bounds for clipping, thus we re-use the same matrix for both.
                converted_paths.append(f"{base_path}.mlp.gate_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
                converted_paths.append(f"{base_path}.mlp.up_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
            if path.endswith("linear/ClippedEinsum_0"):
                converted_paths.append(f"{base_path}.mlp.down_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)

            # Handle clipped einsum states (`compression_einsum` target paths).
            # The target path specifies the activation direction (`input` or `output`),
            # and the parameter holds `clip_min` or `clip_max`.
            if "/compression_einsum/" in path:
                direction = path.split("/")[-1].split("_")[0]  # Extracts "input" or "output"
                hf_suffix = f"{direction}_{param.removeprefix('clip_')}"
                einsum_type = path.split("/compression_einsum/")[0].split("/")[-1]

                if einsum_type == "attn_vec_einsum":
                    converted_paths.append(f"{base_path}.self_attn.o_proj.{hf_suffix}")
                    converted_weights.append(matrix)
                elif einsum_type == "kv_einsum":
                    converted_paths.append(f"{base_path}.self_attn.k_proj.{hf_suffix}")
                    converted_weights.append(matrix)
                    converted_paths.append(f"{base_path}.self_attn.v_proj.{hf_suffix}")
                    converted_weights.append(matrix)
                elif einsum_type == "q_einsum":
                    converted_paths.append(f"{base_path}.self_attn.q_proj.{hf_suffix}")
                    converted_weights.append(matrix)
                elif einsum_type == "gating_einsum":
                    converted_paths.append(f"{base_path}.mlp.gate_proj.{hf_suffix}")
                    converted_weights.append(matrix)
                    converted_paths.append(f"{base_path}.mlp.up_proj.{hf_suffix}")
                    converted_weights.append(matrix)
                elif einsum_type == "linear":
                    converted_paths.append(f"{base_path}.mlp.down_proj.{hf_suffix}")
                    converted_weights.append(matrix)

            if path.endswith("attn/attn_vec_einsum"):
                # Shape: (12, 64, 768) -> reshape to (768, 768) for o_proj
                converted_paths.append(f"{base_path}.self_attn.o_proj.linear.weight")
                converted_weights.append(
                    matrix.transpose(2, 0, 1).reshape(config.hidden_size, config.num_attention_heads * config.head_dim)
                )
            elif path.endswith("attn/kv_einsum"):
                # Shape: (2, 12, 768, 64) -> split into k_proj and v_proj
                converted_paths.extend(
                    [
                        f"{base_path}.self_attn.k_proj.linear.weight",
                        f"{base_path}.self_attn.v_proj.linear.weight",
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
                # Shape: (12, 768, 64) -> reshape to (768, 768) for q_proj
                converted_paths.append(f"{base_path}.self_attn.q_proj.linear.weight")
                converted_weights.append(
                    matrix.transpose(1, 0, 2)
                    .reshape(config.hidden_size, config.num_attention_heads * config.head_dim)
                    .transpose()
                )
            elif path.endswith("mlp/gating_einsum"):
                # Shape: (2, 3072, 768) -> split into gate_proj and up_proj
                converted_paths.extend(
                    [
                        f"{base_path}.mlp.gate_proj.linear.weight",
                        f"{base_path}.mlp.up_proj.linear.weight",
                    ]
                )
                gate_proj_weight, up_proj_weight = matrix
                converted_weights.extend([gate_proj_weight, up_proj_weight])
            elif path.endswith("mlp/linear"):
                # Shape: (3072, 768) -> transpose for down_proj
                converted_paths.append(f"{base_path}.mlp.down_proj.linear.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("post_attention_norm"):
                converted_paths.append(f"{base_path}.post_attention_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_ffw_norm"):
                converted_paths.append(f"{base_path}.post_feedforward_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("pre_attention_norm"):
                converted_paths.append(f"{base_path}.input_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("pre_ffw_norm"):
                converted_paths.append(f"{base_path}.pre_feedforward_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("attn/query_norm/scale") or path.endswith("attn/query_norm"):
                # Vision Q/K norms: JAX trained scale values (~-0.6) are not directly
                # usable because the OSS modules expect different shapes and the HF
                # RMSNorm uses scale_shift=1.0 (formula: weight + 1.0).
                # We use zeros to get identity: (0 + 1.0) = 1.0, matching the blaze
                # reference which also uses zeros(head_dim) -> (1+0) = 1.0 identity.
                converted_paths.append(f"{base_path}.self_attn.q_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("attn/key_norm/scale") or path.endswith("attn/key_norm"):
                converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
                converted_weights.append(matrix)

    if (cpl := len(converted_paths)) != (cwl := len(converted_weights)):
        raise ValueError(
            "The `converted_paths` and `converted_weights` should be the same "
            f"length. Got {cpl} and {cwl}, respectively, for {path}."
        )

    return zip(converted_paths, converted_weights)


def convert_transformer_weights(
    config: Gemma4TextConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    if path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
        path = path[_TRANSFORMER_POST_TRAINING_PREFIX_LEN:]

    converted_paths: list[str] = []
    converted_weights: list[Any] = []
    first_kv_shared_layer_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)

    # Handle new checkpoint format: transformer/layer_N/...
    # TODO(philculliton):Direct handling for unstacked checkpoint type, needs to be merged to allow for unified tensor handling
    if path.startswith(f"{_TRANSFORMER_PARAMETER}/layer_"):
        # Extract layer number from path like "transformer/layer_0/attn/q_einsum"
        layer_str = path.split("/")[1]  # "layer_0"
        layer_idx = int(layer_str.replace("layer_", ""))  # 0
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        base_path = f"layers.{layer_idx}"

        # Determine head_dim from actual checkpoint weight dimensions
        # For q_einsum/key_norm, the last dimension tells us the head_dim
        # Otherwise fall back to config
        if path.endswith("attn/key_norm") or path.endswith("attn/query_norm"):
            head_dim = weights.shape[0]  # The norm dimension IS the head_dim
        elif path.endswith("attn/q_einsum"):
            head_dim = weights.shape[-1]  # Last dimension is head_dim
        else:
            # Fall back to config-based determination
            head_dim = (
                config.global_head_dim
                if config.layer_types[layer_idx] == "full_attention" and config.global_head_dim
                else config.head_dim
            )

        # Note: In new format, weights are per-layer (not batched), so no enumerate loop needed
        matrix = weights

        if path.endswith("attn/attn_vec_einsum"):
            converted_paths.append(f"{base_path}.self_attn.o_proj.weight")
            converted_weights.append(
                matrix.transpose(2, 0, 1).reshape(config.hidden_size, config.num_attention_heads * head_dim)
            )
        elif path.endswith("attn/kv_einsum") and not is_kv_shared_layer:
            converted_paths.extend(
                [
                    f"{base_path}.self_attn.k_proj.weight",
                    f"{base_path}.self_attn.v_proj.weight",
                ]
            )
            k_proj_weights, v_proj_weights = matrix.transpose(0, 2, 1, 3)
            kv_proj_shape = (config.hidden_size, config.num_key_value_heads * head_dim)
            converted_weights.extend(
                [
                    k_proj_weights.reshape(kv_proj_shape).transpose(),
                    v_proj_weights.reshape(kv_proj_shape).transpose(),
                ]
            )
        elif path.endswith("attn/k_einsum") and not is_kv_shared_layer:
            converted_paths.append(f"{base_path}.self_attn.k_proj.weight")
            converted_weights.append(
                matrix.transpose(1, 0, 2)
                .reshape(config.hidden_size, config.num_global_key_value_heads * head_dim)
                .transpose()
            )
        elif path.endswith("attn/q_einsum"):
            converted_paths.append(f"{base_path}.self_attn.q_proj.weight")
            converted_weights.append(
                matrix.transpose(1, 0, 2)
                .reshape(config.hidden_size, config.num_attention_heads * head_dim)
                .transpose()
            )
        elif path.endswith("attn/query_norm"):
            converted_paths.append(f"{base_path}.self_attn.q_norm.weight")
            converted_weights.append(matrix)
        elif path.endswith("attn/key_norm") and not is_kv_shared_layer:
            converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
            converted_weights.append(matrix)
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
        elif path.endswith("post_per_layer_input_norm"):
            converted_paths.append(f"{base_path}.post_per_layer_input_norm.weight")
            converted_weights.append(matrix)
        elif path.endswith("pre_attention_norm"):
            converted_paths.append(f"{base_path}.input_layernorm.weight")
            converted_weights.append(matrix)
        elif path.endswith("pre_ffw_norm"):
            converted_paths.append(f"{base_path}.pre_feedforward_layernorm.weight")
            converted_weights.append(matrix)
        elif path.endswith(layer_str) and param == "skip_scale":
            converted_paths.append(f"{base_path}.layer_scalar")
            converted_weights.append(matrix)

    # Handle old checkpoint format: transformer/stacked_layers/attention_type_N/...
    elif path.startswith(_TRANSFORMER_DECODER_BLOCK):
        attention_type_index = int(path[_TRANSFORMER_DECODER_BLOCK_LEN])
        expected_layers_per_group = config.num_hidden_layers / _SLIDING_WINDOW_PATTERN
        observed_layers_per_group = weights.shape[0]
        assert observed_layers_per_group == expected_layers_per_group, (
            f"Expected {observed_layers_per_group=} to be {expected_layers_per_group=}"
        )

        for i, matrix in enumerate(weights):
            layer_idx = _SLIDING_WINDOW_PATTERN * i + attention_type_index
            is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
            base_path = f"layers.{layer_idx}"
            head_dim = (
                config.global_head_dim
                if config.layer_types[layer_idx] == "full_attention" and config.global_head_dim
                else config.head_dim
            )

            if param == "skip_scale":
                converted_paths.append(f"{base_path}.layer_scalar")
                converted_weights.append(matrix)
            elif path.endswith("attn/attn_vec_einsum"):
                converted_paths.append(f"{base_path}.self_attn.o_proj.weight")
                converted_weights.append(
                    matrix.transpose(2, 0, 1).reshape(config.hidden_size, config.num_attention_heads * head_dim)
                )
            elif path.endswith("attn/kv_einsum") and not is_kv_shared_layer:
                converted_paths.extend(
                    [
                        f"{base_path}.self_attn.k_proj.weight",
                        f"{base_path}.self_attn.v_proj.weight",
                    ]
                )
                k_proj_weights, v_proj_weights = matrix.transpose(0, 2, 1, 3)
                kv_proj_shape = (config.hidden_size, config.num_key_value_heads * head_dim)
                converted_weights.extend(
                    [
                        k_proj_weights.reshape(kv_proj_shape).transpose(),
                        v_proj_weights.reshape(kv_proj_shape).transpose(),
                    ]
                )
            elif path.endswith("attn/k_einsum") and not is_kv_shared_layer:
                converted_paths.append(f"{base_path}.self_attn.k_proj.weight")
                converted_weights.append(
                    matrix.transpose(1, 0, 2)
                    .reshape(config.hidden_size, config.num_global_key_value_heads * head_dim)
                    .transpose()
                )
            elif path.endswith("attn/q_einsum"):
                converted_paths.append(f"{base_path}.self_attn.q_proj.weight")
                converted_weights.append(
                    matrix.transpose(1, 0, 2)
                    .reshape(config.hidden_size, config.num_attention_heads * head_dim)
                    .transpose()
                )
            elif path.endswith("attn/query_norm"):
                converted_paths.append(f"{base_path}.self_attn.q_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("attn/key_norm") and not is_kv_shared_layer:
                converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("mlp/gating_einsum"):
                # NOTE: The JAX implementations changes the type of the primary `mlp` for MOE models and adds a new
                # `mlp2` that operates _before_ `mlp`. In Hugging Face Transformers we keep the type of `mlp` constant
                # and add an `experts` that operates after `mlp`, so we need to invert this assignment when using MOE arch.
                if config.enable_moe_block:
                    # MoE expert weights: matrix shape [num_experts, 2, moe_intermediate_size, hidden_size]
                    # -> experts.gate_up_proj (nn.Parameter, shape [E, 2*moe_inter, hidden])
                    num_experts, _, expert_inter, hidden_size = matrix.shape
                    gate_up_proj_weight = np.asarray(matrix).reshape(num_experts, 2 * expert_inter, hidden_size)
                    converted_paths.append(f"{base_path}.experts.gate_up_proj")
                    converted_weights.append(gate_up_proj_weight)
                else:
                    # Dense MLP: matrix shape [2, intermediate_size, hidden_size]
                    gate_proj_weight, up_proj_weight = matrix
                    converted_paths.extend([f"{base_path}.mlp.gate_proj.weight", f"{base_path}.mlp.up_proj.weight"])
                    converted_weights.extend([gate_proj_weight, up_proj_weight])
            elif path.endswith("mlp/linear"):
                # NOTE: The JAX implementations changes the type of the primary `mlp` for MOE models and adds a new
                # `mlp2` that operates _before_ `mlp`. In Hugging Face Transformers we keep the type of `mlp` constant
                # and add an `experts` that operates after `mlp`, so we need to invert this assignment when using MOE arch.
                if config.enable_moe_block:
                    # MoE expert down_proj: matrix shape [num_experts, moe_inter, hidden]
                    # -> experts.down_proj (nn.Parameter, shape [E, hidden, moe_inter])
                    converted_paths.append(f"{base_path}.experts.down_proj")
                    converted_weights.append(matrix.transpose(0, 2, 1))
                else:
                    # Dense MLP down_proj
                    converted_paths.append(f"{base_path}.mlp.down_proj.weight")
                    converted_weights.append(matrix.transpose())
            elif path.endswith("mlp/router_logits"):
                # MoE router: matrix shape [hidden_size, num_experts]
                # -> router.proj.weight (nn.Linear, shape [num_experts, hidden_size])
                converted_paths.append(f"{base_path}.router.proj.weight")
                converted_weights.append(matrix.transpose())
            elif param == "router_scale" and path.endswith("mlp"):
                # MoE router scale: shape [hidden_size]
                converted_paths.append(f"{base_path}.router.scale")
                converted_weights.append(matrix)
            elif param == "per_expert_scale" and path.endswith("mlp"):
                # MoE per-expert scale: shape [num_experts]
                converted_paths.append(f"{base_path}.router.per_expert_scale")
                converted_weights.append(matrix)
            elif path.endswith("mlp2/gating_einsum"):
                # Shared expert: matrix shape [2, intermediate_size, hidden_size]
                # -> mlp.gate_proj.weight + mlp.up_proj.weight (nn.Linear)
                converted_paths.extend([f"{base_path}.mlp.gate_proj.weight", f"{base_path}.mlp.up_proj.weight"])
                gate_proj_weight, up_proj_weight = matrix
                converted_weights.extend([gate_proj_weight, up_proj_weight])
            elif path.endswith("mlp2/linear"):
                # Shared expert down_proj: matrix shape [intermediate_size, hidden_size]
                # -> mlp.down_proj.weight (nn.Linear, needs transpose)
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
            elif path.endswith("post_ffw1_norm"):
                converted_paths.append(f"{base_path}.post_feedforward_layernorm_2.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_ffw2_norm"):
                converted_paths.append(f"{base_path}.post_feedforward_layernorm_1.weight")
                converted_weights.append(matrix)
            elif path.endswith("pre_ffw2_norm"):
                converted_paths.append(f"{base_path}.pre_feedforward_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_per_layer_input_norm"):
                converted_paths.append(f"{base_path}.post_per_layer_input_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("pre_attention_norm"):
                converted_paths.append(f"{base_path}.input_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("pre_ffw_norm"):
                # NOTE: The JAX implementations changes the type of the primary `mlp` for MOE models and adds a new
                # `mlp2` that operates _before_ `mlp`. In Hugging Face Transformer we keep the type of `mlp` constant
                # and add an `mlp2` that operates after `mlp`, so we need to invert this assignment when using MOE arch.
                if config.enable_moe_block:
                    # pre_ffw_norm is the pre-norm for ffw1 (MoE); in HF, MoE is mlp_2
                    converted_paths.append(f"{base_path}.pre_feedforward_layernorm_2.weight")
                else:
                    converted_paths.append(f"{base_path}.pre_feedforward_layernorm.weight")
                converted_weights.append(matrix)
    elif path == _TRANSFORMER_EMBEDDER:
        if param == "input_embedding":
            converted_paths.append("embed_tokens.weight")
            converted_weights.append(weights)
        elif param == "per_layer_embeddings":
            converted_paths.append("embed_tokens_per_layer.weight")
            # JAX uses an einsum, but Transformers uses a Linear, so reshapes are required here and in modeling file.
            vocab_size, num_layers, hidden_dim = weights.shape
            converted_weights.append(weights.reshape(vocab_size, num_layers * hidden_dim))
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


def _restore_checkpoint(checkpoint_path: str) -> dict:
    """Restores an Orbax checkpoint, handling multi-device sharded checkpoints.

    Reads the checkpoint metadata to build a target tree structure and uses
    SingleDeviceSharding to consolidate all shards onto a single CPU device.
    """
    metadata_path = os.path.join(checkpoint_path, "_METADATA")
    with open(metadata_path, "rb") as f:
        metadata = json.loads(f.read())

    tree_metadata = metadata["tree_metadata"]

    # Build a nested dict matching the checkpoint's tree structure
    target = {}
    for key_str in tree_metadata:
        keys = ast.literal_eval(key_str)
        d = target
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = np.zeros(1)  # placeholder leaf

    device = jax.devices("cpu")[0]
    sharding = SingleDeviceSharding(device)

    restore_args_tree = tree.map_structure(lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding), target)
    restore = obc_args.PyTreeRestore(item=target, restore_args=restore_args_tree)

    checkpointer = obc.PyTreeCheckpointer()
    return checkpointer.restore(checkpoint_path, args=restore)


def convert(checkpoint_path: str, config: Gemma4Config) -> dict[str, torch.Tensor]:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    ckpt = _restore_checkpoint(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    text_path_prefix = "model"
    if not _TEXT_ONLY.value:
        text_path_prefix += ".language_model"

    def update_tree(path: str, weights: np.ndarray, target_dtype: torch.dtype) -> None:
        # Convert directly to float32 in a single step to avoid an extra intermediate copy.
        # The old code did np.asarray(weights) then .astype("float32"), keeping two full copies alive.
        weights_f32 = np.asarray(weights, dtype=np.float32)
        del weights  # allow GC of the input (JAX array or numpy view)
        t = torch.from_numpy(weights_f32)  # shares memory with weights_f32
        if t.dtype != target_dtype:
            hf_tree[path] = t.to(target_dtype)
            del t, weights_f32  # free the float32 intermediate
        else:
            hf_tree[path] = t
        if _VERBOSE.value:
            logging.info(
                "%s converted shape=%s with dtype=%s",
                path,
                hf_tree[path].shape,
                target_dtype,
            )

    for path_tuple, value in tree.flatten_with_path(ckpt):
        param = path_tuple[-1]
        if "params" in path_tuple:
            path_tuple = path_tuple[2:]
        path_tuple = path_tuple[:-1]
        path = "/".join(path_tuple) if len(path_tuple) > 1 else path_tuple[0]

        if path.endswith("audio_input_projection") and not _TEXT_ONLY.value:
            update_tree("model.embed_audio.embedding_projection.weight", value.transpose(), config.audio_config.dtype)
        elif path.endswith("mm_input_projection") and not _TEXT_ONLY.value:
            update_tree(
                "model.embed_vision.embedding_projection.weight", value.transpose(), config.vision_config.dtype
            )
        elif path.startswith(_TRANSFORMER_PARAMETER):
            for hf_path, weights in convert_transformer_weights(config.text_config, path, param, value):
                update_tree(f"{text_path_prefix}.{hf_path}", weights, config.text_config.dtype)
        elif path.startswith(_VISION_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
            for hf_path, weights in convert_vision_encoder_weights(config.vision_config, path, param, value):
                update_tree(f"model.vision_tower.{hf_path}", weights, config.vision_config.dtype)
        elif path.startswith(_AUDIO_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
            for hf_path, weights in convert_audio_encoder_weights(config.audio_config, path, param, value):
                update_tree(f"model.audio_tower.{hf_path}", weights, config.audio_config.dtype)

    hf_tree["lm_head.weight"] = hf_tree[f"{text_path_prefix}.embed_tokens.weight"]

    return hf_tree


def main(*args):
    del args

    output_path = _OUTPUT_PATH.value
    variant = _VARIANT.value

    config = _VARIANTS[variant]
    config.text_config.dtype = getattr(torch, _TEXT_DTYPE.value)
    config.vision_config.dtype = getattr(torch, _VISION_DTYPE.value)
    if (audio_config := config.audio_config) is not None:
        audio_config.dtype = getattr(torch, _AUDIO_DTYPE.value)

    if _INCLUDE_CHAT_TEMPLATE.value:
        # Chat template is included for instruction tuned models, which treat
        # both "<eos>" and "<end_of_turn>" as generation stoppers.
        config.eos_token_id = [1, 106]

    logging.info(
        "Converting Gemma 4 (%s) @ %s (language) and %s (vision)",
        variant,
        _TEXT_DTYPE.value,
        _VISION_DTYPE.value,
    )
    state_tree = convert(_CHECKPOINT_PATH.value, config)
    logging.info("Converted Gemma 4 (%s) state tree from Orbax to Hugging Face.", variant)

    with accelerate.init_empty_weights():
        if _TEXT_ONLY.value:
            config = config.text_config
            model = Gemma4ForCausalLM(config=config)
        else:
            model = Gemma4ForConditionalGeneration(config=config)

    model.load_state_dict(state_tree, assign=True)
    logging.info(
        "Loaded Gemma 4 (%s) in Hugging Face Transformers as a %s instance.",
        variant,
        type(model).__name__,
    )
    model.save_pretrained(output_path, state_dict=state_tree, safe_serialization=True)
    logging.info(
        "Saved Gemma 4 (%s) to SafeTensors in %s using %s",
        variant,
        output_path,
        type(model).__name__,
    )
    del model
    del state_tree

    chat_template = _CHAT_TEMPLATE_LARGE if variant in _LARGE_MODEL_VARIANTS else _CHAT_TEMPLATE
    chat_template_kwargs = {"chat_template": chat_template} if _INCLUDE_CHAT_TEMPLATE.value else {}
    response_schema_kwargs = {"response_schema": _RESPONSE_SCHEMA} if _INCLUDE_RESPONSE_SCHEMA.value else {}

    sentencepiece_extractor = SentencePieceExtractor(_TOKENIZER_PATH.value)
    vocab, _, merges = sentencepiece_extractor.extract()
    tokenizer = GemmaTokenizer(
        vocab=vocab,
        merges=merges,
        add_bos_token=False,
        padding_side="left",
        extra_special_tokens={
            "image_token": "<|image|>",
            "boi_token": "<|image>",
            "eoi_token": "<image|>",
            "audio_token": "<|audio|>",
            "boa_token": "<|audio>",
            "eoa_token": "<audio|>",
            "sot_token": "<|turn>",
            "eot_token": "<turn|>",
            "soc_token": "<|channel>",
            "eoc_token": "<channel|>",
            "think_token": "<|think|>",
            "escape_token": '<|"|>',
            "str_token": "<|tool_response>",
            "etr_token": "<tool_response|>",
            "stc_token": "<|tool_call>",
            "etc_token": "<tool_call|>",
            "std_token": "<|tool>",
            "etd_token": "<tool|>",
        },
        **chat_template_kwargs,
        **response_schema_kwargs,
    )

    # Update config multimodal token IDs from the tokenizer.
    # The Gemma4 SPM (262144 vocab) has native <|image> (255999) and <|audio> (256000)
    # tokens, plus <image|> (258882) and <audio|> (258883) for delimiters.
    # Only <image_soft_token> and <audio_soft_token> are added as new tokens (IDs >= 262144).
    config.image_token_id = tokenizer.image_token_id
    config.boi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
    config.eoi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoi_token)
    config.audio_token_id = tokenizer.audio_token_id
    config.boa_token_id = tokenizer.convert_tokens_to_ids(tokenizer.boa_token)
    config.eoa_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoa_token)
    logging.info(
        "Set multimodal token IDs from tokenizer: image=%d, boi=%d, eoi=%d, audio=%d, boa=%d, eoa=%d",
        config.image_token_id,
        config.boi_token_id,
        config.eoi_token_id,
        config.audio_token_id,
        config.boa_token_id,
        config.eoa_token_id,
    )
    # Re-save the config with correct token IDs
    config.save_pretrained(output_path)

    if _TEXT_ONLY.value:
        tokenizer.save_pretrained(output_path)
        logging.info("Saved GemmaTokenizer for %s to %s", variant, output_path)
    else:
        vision_config = config.vision_config
        feature_extractor = Gemma4AudioFeatureExtractor()
        image_processor = Gemma4ImageProcessor(
            image_seq_length=vision_config.default_output_length,
            do_normalize=False,
            max_soft_tokens=vision_config.default_output_length,
            pooling_kernel_size=3,
        )
        video_processor = Gemma4VideoProcessor()
        processor = Gemma4Processor(
            image_processor=image_processor,
            feature_extractor=feature_extractor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            image_seq_length=vision_config.default_output_length,
            **chat_template_kwargs,
        )
        processor.save_pretrained(output_path)

        logging.info("Saved Gemma4Processor for %s to %s", variant, output_path)
        del feature_extractor, image_processor, processor

    generation_config = GenerationConfig(
        pad_token_id=config.get_text_config().pad_token_id,
        bos_token_id=config.get_text_config().bos_token_id,
        eos_token_id=(
            tokenizer.convert_tokens_to_ids([tokenizer.eos_token, tokenizer.eot_token, tokenizer.str_token])
            if _INCLUDE_CHAT_TEMPLATE.value
            else config.get_text_config().eos_token_id
        ),
        temperature=1.0,
        do_sample=True,
        top_k=64,
        top_p=0.95,
    )
    generation_config.save_pretrained(output_path)


if __name__ == "__main__":
    app.run(main)
