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
    --include_response_format \
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
from safetensors.torch import save_file

from transformers import (
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
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
from transformers.utils.quantization_config import GemmaQuantizationConfig


# ==== Internal Constants and Classes ====

# The correct chat templates were already uploaded to those 2 repos, so download from there
_CHAT_TEMPLATE = pathlib.Path(cached_file("gg-hf-gg/gemma-4-E4B-it", "chat_template.jinja")).read_text()
_CHAT_TEMPLATE_LARGE = pathlib.Path(cached_file("gg-hf-gg/gemma-4-31B-it", "chat_template.jinja")).read_text()

_RESPONSE_FORMAT = {
    "defaults": {"role": "assistant"},
    "fields": {
        "thinking": {
            "open": "<|channel>thought\n",
            "close": "<channel|>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "close": "<tool_call|>",
            "repeats": True,
            "content": "json-lax",
            "content_args": {
                "unquoted_keys": True,
                "string_delims": [['<|"|>', '<|"|>']],
            },
            "assemble": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        "content": {
            "close_pattern": r"(?:<turn\|>|<\|tool_response>|$)",
            "content": "text",
        },
    },
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

_TRANSFORMER_PRE_PROJ_MTP = "transformer/pre_proj"
_TRANSFORMER_POST_PROJ_MTP = "transformer/post_proj"
_TRANSFORMER_NORM_MTP = "transformer/norm"

_VARIANT_GEMMA_4_E2B_ASSISTANT = "gemma-4-e2b-assistant"
_VARIANT_GEMMA_4_E4B_ASSISTANT = "gemma-4-e4b-assistant"
_VARIANT_GEMMA_4_26B_A4B_ASSISTANT = "gemma-4-26b-a4b-assistant"
_VARIANT_GEMMA_4_31B_ASSISTANT = "gemma-4-31b-assistant"

_ASSISTANT_VARIANTS = {
    _VARIANT_GEMMA_4_E2B_ASSISTANT,
    _VARIANT_GEMMA_4_E4B_ASSISTANT,
    _VARIANT_GEMMA_4_26B_A4B_ASSISTANT,
    _VARIANT_GEMMA_4_31B_ASSISTANT,
}

_LARGE_MODEL_VARIANTS = {
    _VARIANT_GEMMA_4_31B,
    _VARIANT_GEMMA_4_26B_A4B,
}

_ASSISTANT_MODEL_COMMON_TEXT_CONFIG_KWARGS = {
    "vocab_size_per_layer_input": 0,
    "hidden_size_per_layer_input": 0,
    "enable_moe_block": False,
    "attention_k_eq_v": False,
    "use_double_wide_mlp": False,
    "num_global_key_value_heads": None,
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


_VARIANTS: Mapping[str, Gemma4Config | Gemma4AssistantConfig] = {
    _VARIANT_GEMMA_4_E2B_ASSISTANT: Gemma4AssistantConfig(
        backbone_hidden_size=1536,
        use_ordered_embeddings=True,
        text_config=Gemma4TextConfig(
            hidden_size=256,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=1,
            num_kv_shared_layers=4,
            head_dim=256,
            global_head_dim=512,
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            **_ASSISTANT_MODEL_COMMON_TEXT_CONFIG_KWARGS,
        ),
    ),
    _VARIANT_GEMMA_4_E4B_ASSISTANT: Gemma4AssistantConfig(
        backbone_hidden_size=2560,
        use_ordered_embeddings=True,
        text_config=Gemma4TextConfig(
            hidden_size=256,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_kv_shared_layers=4,
            head_dim=256,
            global_head_dim=512,
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            **_ASSISTANT_MODEL_COMMON_TEXT_CONFIG_KWARGS,
        ),
    ),
    _VARIANT_GEMMA_4_31B_ASSISTANT: Gemma4AssistantConfig(
        backbone_hidden_size=5376,
        text_config=Gemma4TextConfig(
            hidden_size=1024,
            hidden_size_per_layer_input=0,
            intermediate_size=8192,
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            num_attention_heads=32,
            num_key_value_heads=16,
            num_global_key_value_heads=4,
            attention_k_eq_v=True,
            num_kv_shared_layers=4,
            sliding_window=1024,
            vocab_size_per_layer_input=0,
        ),
    ),
    _VARIANT_GEMMA_4_26B_A4B_ASSISTANT: Gemma4AssistantConfig(
        backbone_hidden_size=2816,
        text_config=Gemma4TextConfig(
            hidden_size=1024,
            hidden_size_per_layer_input=0,
            intermediate_size=8192,
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            num_attention_heads=16,
            num_key_value_heads=8,
            num_global_key_value_heads=2,
            attention_k_eq_v=True,
            num_kv_shared_layers=4,
            sliding_window=1024,
            vocab_size_per_layer_input=0,
        ),
    ),
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

_INCLUDE_RESPONSE_FORMAT = flags.DEFINE_bool(
    name="include_response_format",
    default=False,
    help="If true, will save the default response_format with the tokenizer",
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

_QUANTIZED = flags.DEFINE_bool(
    name="quantized",
    default=False,
    help=(
        "If true, treats the input as a quantized Orbax checkpoint and preserves "
        "quantized weights (INT2/4/8) + scales instead of dequantizing to float. "
        "The output safetensors will contain packed int weights ready for "
        "Gemma4QuantizedLinear. Set quantization_config in the saved config.json."
    ),
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


def convert_quantized_vision_encoder_weights(
    config,
    path: str,
    param: str,
    weights,
) -> Iterable[tuple[str, np.ndarray]]:
    """Convert quantized vision encoder weights from JAX to HuggingFace format.

    This mirrors convert_vision_encoder_weights but handles quantized parameters
    (weight, weight_scale, activation scales) and reshapes 3D attention tensors
    to 2D for compatibility with QuantizedLinear.
    """
    converted_paths = []
    converted_weights = []

    _VISION_ENCODER_TRANSFORMER = "PatchInputVariablePoolingEncoder_0/_model/vit/transformer/stacked_layers/block"

    if not path.startswith(_VISION_ENCODER_TRANSFORMER):
        return zip(converted_paths, converted_weights)

    # Vision tower wraps each `nn.Linear` in `Gemma4ClippableLinear` at `.linear`;
    # the replacer keeps the wrapper and swaps `.linear` with `QuantizedLinear`,
    # so emit at the inner path. `QuantizedLinear` subclasses `nn.Linear` and
    # stores its packed-int storage at `.weight`.
    suffix = None
    is_value = param in ("w_quantized_value", "quantized_value")
    is_scale = param in ("w_quantized_scale", "quantized_scale")
    if is_value:
        suffix = "linear.weight"
    elif is_scale:
        suffix = "linear.weight_scale"
    elif param == "static_quantized_scale_input":
        suffix = "linear.input_activation_scale"
    elif param == "static_quantized_scale_output":
        suffix = "linear.output_activation_scale"

    if suffix is None:
        return zip(converted_paths, converted_weights)

    # All vision transformer layers are stacked in dimension 0
    for i, matrix in enumerate(weights):
        base_path = f"encoder.layers.{i}"

        if path.endswith("q_einsum"):
            # q_proj value: [heads, hidden, head_dim] -> [heads*head_dim, hidden]
            # q_proj scale: [heads, 1, head_dim] -> [heads*head_dim, 1]
            hf_bases = [f"{base_path}.self_attn.q_proj"]
            if is_value:
                matrix = (
                    matrix.transpose(1, 0, 2)
                    .reshape(config.hidden_size, config.num_attention_heads * config.head_dim)
                    .transpose()
                )
            elif is_scale:
                matrix = matrix.transpose(0, 2, 1).reshape(-1, 1)

        elif path.endswith("kv_einsum"):
            # kv_einsum value: [2, heads, hidden, head_dim] -> split k/v, each -> [heads*head_dim, hidden]
            # kv_einsum scale: [2, heads, 1, head_dim] -> split k/v, each -> [heads*head_dim, 1]
            hf_bases = [f"{base_path}.self_attn.k_proj", f"{base_path}.self_attn.v_proj"]
            if is_value:
                k_matrix, v_matrix = matrix
                kv_shape = (config.hidden_size, config.num_key_value_heads * config.head_dim)
                k_reshaped = k_matrix.transpose(1, 0, 2).reshape(kv_shape).transpose()
                v_reshaped = v_matrix.transpose(1, 0, 2).reshape(kv_shape).transpose()
                for hf_base in hf_bases:
                    converted_paths.append(f"{hf_base}.{suffix}")
                converted_weights.append(k_reshaped)
                converted_weights.append(v_reshaped)
                continue
            elif is_scale:
                k_scale, v_scale = matrix
                k_flat = k_scale.transpose(0, 2, 1).reshape(-1, 1)
                v_flat = v_scale.transpose(0, 2, 1).reshape(-1, 1)
                for hf_base in hf_bases:
                    converted_paths.append(f"{hf_base}.{suffix}")
                converted_weights.append(k_flat)
                converted_weights.append(v_flat)
                continue
            else:
                # Scalar params (activation scales, num_bits) are shared for k and v
                for hf_base in hf_bases:
                    converted_paths.append(f"{hf_base}.{suffix}")
                    converted_weights.append(matrix)
                continue

        elif path.endswith("attn_vec_einsum"):
            # o_proj value: [heads, head_dim, hidden] -> [hidden, heads*head_dim]
            # o_proj scale: [1, 1, hidden] -> [hidden, 1]
            hf_bases = [f"{base_path}.self_attn.o_proj"]
            if is_value:
                matrix = matrix.transpose(2, 0, 1).reshape(
                    config.hidden_size, config.num_attention_heads * config.head_dim
                )
            elif is_scale:
                matrix = matrix.reshape(-1, 1)

        elif path.endswith("gating_einsum"):
            # gating_einsum value: [2, intermediate, hidden] -> split gate/up, each stays [inter, hidden]
            # gating_einsum scale: [2, intermediate, 1] -> split gate/up, each stays [inter, 1]
            hf_bases = [f"{base_path}.mlp.gate_proj", f"{base_path}.mlp.up_proj"]
            if is_value or is_scale:
                gate_matrix, up_matrix = matrix
                for hf_base in hf_bases:
                    converted_paths.append(f"{hf_base}.{suffix}")
                converted_weights.append(gate_matrix)
                converted_weights.append(up_matrix)
                continue
            else:
                # Scalar params shared for gate and up
                for hf_base in hf_bases:
                    converted_paths.append(f"{hf_base}.{suffix}")
                    converted_weights.append(matrix)
                continue

        elif path.endswith("linear"):
            # down_proj value: [intermediate, hidden] -> [hidden, intermediate] for nn.Linear
            # down_proj scale: [1, hidden] -> [hidden, 1] to match weight_scale shape
            hf_bases = [f"{base_path}.mlp.down_proj"]
            if is_value:
                matrix = matrix.transpose()
            elif is_scale:
                matrix = matrix.reshape(-1, 1)

        else:
            continue

        for hf_base in hf_bases:
            converted_paths.append(f"{hf_base}.{suffix}")
            converted_weights.append(matrix)

    return zip(converted_paths, converted_weights)


def convert_quantized_audio_encoder_weights(
    config,
    path: str,
    param: str,
    weights,
) -> Iterable[tuple[str, np.ndarray]]:
    """Convert quantized audio encoder weights from JAX to HuggingFace format.

    This handles quantized parameters (weight, weight_scale, activation scales,
    and rqv2_muls) and reshapes attention tensors to 2D for compatibility with
    Gemma4QuantizedLinear.  Weights are packed into uint8 matching their native
    bit width (int2 or int4) so that QuantizedLinear can decode them correctly.
    """
    converted_paths = []
    converted_weights = []

    if not path.startswith(_AUDIO_ENCODER_CONFORMER):
        return zip(converted_paths, converted_weights)

    # Audio tower wraps each `nn.Linear` in `Gemma4ClippableLinear` at `.linear`;
    # the replacer keeps the wrapper and swaps `.linear` with `QuantizedLinear`,
    # so emit at the inner path. `QuantizedLinear` subclasses `nn.Linear` and
    # stores its packed-int storage at `.weight`.
    suffix = None
    is_value = param in ("w_quantized_value", "quantized_value", "kernel_quantized_value")
    is_scale = param in ("w_quantized_scale", "quantized_scale", "kernel_quantized_scale")
    if is_value:
        suffix = "linear.weight"
    elif is_scale:
        suffix = "linear.weight_scale"
    elif param == "static_quantized_scale_input":
        suffix = "linear.input_activation_scale"
    elif param == "static_quantized_scale_output":
        suffix = "linear.output_activation_scale"
    # rqv2_muls is a training artifact not needed for inference dequantization
    # (same as the text model path at line 1709). Drop it silently.

    if suffix is None:
        return zip(converted_paths, converted_weights)

    # Detect int4/int2 source dtype for packing (before np.asarray loses it)
    is_int4 = False
    is_int2 = False
    if is_value:
        jax_dtype = str(getattr(weights, "dtype", ""))
        is_int4 = "int4" in jax_dtype
        is_int2 = "int2" in jax_dtype

    # All audio conformer layers are stacked in dimension 0
    for i, matrix in enumerate(weights):
        base_path = f"layers.{i}"

        if "fflayer_end" in path:
            base = f"{base_path}.feed_forward2"
            if path.endswith("ffn_layer1"):
                hf_bases = [f"{base}.ffw_layer_1"]
                if is_value:
                    matrix = matrix.transpose()
                elif is_scale:
                    matrix = matrix.reshape(-1, 1)
            elif path.endswith("ffn_layer2"):
                hf_bases = [f"{base}.ffw_layer_2"]
                if is_value:
                    matrix = matrix.transpose()
                elif is_scale:
                    matrix = matrix.reshape(-1, 1)
            else:
                continue

        elif "fflayer_start" in path:
            base = f"{base_path}.feed_forward1"
            if path.endswith("ffn_layer1"):
                hf_bases = [f"{base}.ffw_layer_1"]
                if is_value:
                    matrix = matrix.transpose()
                elif is_scale:
                    matrix = matrix.reshape(-1, 1)
            elif path.endswith("ffn_layer2"):
                hf_bases = [f"{base}.ffw_layer_2"]
                if is_value:
                    matrix = matrix.transpose()
                elif is_scale:
                    matrix = matrix.reshape(-1, 1)
            else:
                continue

        elif "lconv" in path:
            base = f"{base_path}.lconv1d"
            if path.endswith("linear_end"):
                hf_bases = [f"{base}.linear_end"]
                if is_value:
                    matrix = matrix.transpose()
                elif is_scale:
                    matrix = matrix.reshape(-1, 1)
            elif path.endswith("linear_start"):
                hf_bases = [f"{base}.linear_start"]
                if is_value:
                    matrix = matrix.transpose()
                elif is_scale:
                    matrix = matrix.reshape(-1, 1)
            else:
                continue

        elif "trans_atten" in path:
            base = f"{base_path}"
            if path.endswith("query_key_value_projection"):
                hf_bases = [f"{base}.self_attn.q_proj", f"{base}.self_attn.k_proj", f"{base}.self_attn.v_proj"]
                if is_value:
                    # Split K/V along first axis: [hidden, 3, heads, head_dim] -> [3, hidden, heads, head_dim]
                    q_matrix, k_matrix, v_matrix = matrix.transpose(1, 0, 2, 3)
                    kv_shape = (config.hidden_size, config.hidden_size)
                    q_reshaped = q_matrix.reshape(kv_shape).transpose()
                    k_reshaped = k_matrix.reshape(kv_shape).transpose()
                    v_reshaped = v_matrix.reshape(kv_shape).transpose()
                    for hf_base in hf_bases:
                        converted_paths.append(f"{hf_base}.{suffix}")
                    converted_weights.extend([q_reshaped, k_reshaped, v_reshaped])
                    continue
                elif is_scale:
                    # Per-channel scale: [1, 3, heads, head_dim] -> split and squeeze
                    q_scale, k_scale, v_scale = matrix[0]
                    q_flat = q_scale.reshape(-1, 1)
                    k_flat = k_scale.reshape(-1, 1)
                    v_flat = v_scale.reshape(-1, 1)
                    for hf_base in hf_bases:
                        converted_paths.append(f"{hf_base}.{suffix}")
                    converted_weights.extend([q_flat, k_flat, v_flat])
                    continue
                else:
                    # Scalar params shared for Q, K, V
                    for hf_base in hf_bases:
                        converted_paths.append(f"{hf_base}.{suffix}")
                        converted_weights.append(matrix)
                    continue

            elif path.endswith("post"):
                hf_bases = [f"{base}.self_attn.post"]
                if is_value:
                    # o_proj: [heads, head_dim, hidden] -> [hidden, heads*head_dim]
                    matrix = matrix.transpose(2, 0, 1).reshape(config.hidden_size, config.hidden_size)
                elif is_scale:
                    matrix = matrix.reshape(-1, 1)
            else:
                continue
        else:
            continue

        for hf_base in hf_bases:
            converted_paths.append(f"{hf_base}.{suffix}")
            converted_weights.append(matrix)

    # Post-processing: pack int4/int2 weights into uint8 (mirrors text model path)
    if is_value:
        if is_int4:
            converted_weights = [pack_int4_to_uint8(np.asarray(w).astype(np.int8)) for w in converted_weights]
        elif is_int2:
            converted_weights = [pack_int2_to_uint8(np.asarray(w).astype(np.int8)) for w in converted_weights]

    return zip(converted_paths, converted_weights)


def convert_transformer_weights(
    config: Gemma4TextConfig | Gemma4AssistantConfig,
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
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
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
            converted_weights.append(matrix.squeeze())
        elif path.endswith("attn/key_norm") and not is_kv_shared_layer:
            converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
            converted_weights.append(matrix.squeeze())
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
            is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
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
                converted_weights.append(matrix.squeeze())
            elif path.endswith("attn/key_norm") and not is_kv_shared_layer:
                converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
                converted_weights.append(matrix.squeeze())
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
    elif path == _TRANSFORMER_NORM_MTP:
        converted_paths.append("final_norm.weight")
        converted_weights.append(weights)
    elif path == _TRANSFORMER_EMBEDDER:
        if param == "input_embedding_ordered" and getattr(config, "use_ordered_embeddings", False):
            converted_paths.append("embed_tokens.weight")
            converted_weights.append(weights)
        elif param == "input_embedding" and not getattr(config, "use_ordered_embeddings", False):
            converted_paths.append("embed_tokens.weight")
            converted_weights.append(weights)
        elif param == "per_layer_embeddings":
            converted_paths.append("embed_tokens_per_layer.weight")
            # JAX uses an einsum, but Transformers uses a Linear, so reshapes are required here and in modeling file.
            vocab_size, num_layers, hidden_dim = weights.shape
            converted_weights.append(weights.reshape(vocab_size, num_layers * hidden_dim))
    elif path.startswith(_TRANSFORMER_EMBEDDER):
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

    restore_args_tree = tree.map_structure(
        lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding, strict=False), target
    )
    restore = obc_args.PyTreeRestore(item=target, restore_args=restore_args_tree)

    checkpointer = obc.PyTreeCheckpointer()
    return checkpointer.restore(checkpoint_path, args=restore)


def convert(checkpoint_path: str, config: Gemma4Config | Gemma4AssistantConfig) -> dict[str, torch.Tensor]:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    ckpt = _restore_checkpoint(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    if _VARIANT.value in _ASSISTANT_VARIANTS:
        flags.FLAGS.text_only = True

    text_path_prefix = "model"
    if not _TEXT_ONLY.value:
        text_path_prefix += ".language_model"

    text_config = config.get_text_config()

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

        # Gemma4MultimodalEmbedder weights
        if path.endswith("audio_input_projection") and not _TEXT_ONLY.value:
            update_tree("model.embed_audio.embedding_projection.weight", value.transpose(), config.audio_config.dtype)
        elif path.endswith("mm_input_projection") and not _TEXT_ONLY.value:
            update_tree(
                "model.embed_vision.embedding_projection.weight", value.transpose(), config.vision_config.dtype
            )
        # Gemma4AssistantForCausalLM weights
        elif param == "centroids":
            update_tree("masked_embedding.centroids.weight", value, text_config.dtype)
        elif param == "token_ordering":
            update_tree("masked_embedding.token_ordering", value, torch.long)
        elif path == _TRANSFORMER_PRE_PROJ_MTP:
            update_tree("pre_projection.weight", value.transpose(), text_config.dtype)
        elif path == _TRANSFORMER_POST_PROJ_MTP:
            update_tree("post_projection.weight", value.transpose(), text_config.dtype)
        # Subordinate model (e.g., language_model, vision_tower, audio_tower) weights
        elif path.startswith(_TRANSFORMER_PARAMETER):
            for hf_path, weights in convert_transformer_weights(text_config, path, param, value):
                update_tree(f"{text_path_prefix}.{hf_path}", weights, text_config.dtype)
        elif path.startswith(_VISION_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
            for hf_path, weights in convert_vision_encoder_weights(config.vision_config, path, param, value):
                update_tree(f"model.vision_tower.{hf_path}", weights, config.vision_config.dtype)
        elif path.startswith(_AUDIO_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
            for hf_path, weights in convert_audio_encoder_weights(config.audio_config, path, param, value):
                update_tree(f"model.audio_tower.{hf_path}", weights, config.audio_config.dtype)

    hf_tree["lm_head.weight"] = hf_tree[f"{text_path_prefix}.embed_tokens.weight"]

    return hf_tree


def _resolve_compression_einsum_path(path: str, param: str) -> tuple[str, str]:
    """Strip `/compression_einsum/{input,output}_activation` suffix from `path`.

    SRQ scales live under nested paths; `param == "static_quantized_scale"` is
    rewritten to `_input` / `_output` to disambiguate.
    """
    for suffix, side in (
        ("/compression_einsum/input_activation", "input"),
        ("/compression_einsum/output_activation", "output"),
    ):
        if suffix in path:
            new_path = path.split("/compression_einsum/")[0]
            new_param = f"static_quantized_scale_{side}" if param == "static_quantized_scale" else param
            return new_path, new_param
    return path, param


def _handle_encoder_quantized_param(
    *,
    path,
    param,
    value,
    transformer_root,
    source_label,
    hf_prefix,
    convert_fn,
    encoder_config,
    store,
    store_int,
    dequant_pending,
):
    """Vision/audio encoder shared branch: try quantized convert, else collect for dequant."""
    if path.startswith(transformer_root):
        effective_path, effective_param = _resolve_compression_einsum_path(path, param)
        results = list(convert_fn(encoder_config, effective_path, effective_param, value))
        for hf_path, tensor in results:
            if "quantized_value" in effective_param or "num_bits" in effective_param:
                store_int(f"{hf_prefix}.{hf_path}", tensor)
            else:
                store(f"{hf_prefix}.{hf_path}", tensor)
        return

    key = path
    if "quantized_value" in param or param == "w_quantized_value":
        dequant_pending.setdefault(key, {})["value"] = value
        dequant_pending[key]["source"] = source_label
    elif "quantized_scale" in param or param == "w_quantized_scale":
        dequant_pending.setdefault(key, {})["scale"] = value
        dequant_pending[key]["source"] = source_label


def convert_quantized(checkpoint_path: str, config: Gemma4Config) -> dict[str, torch.Tensor]:
    """Loads quantized Orbax checkpoint and converts to HF quantized format.

    Unlike `convert()`, this preserves quantized weights as packed integers
    with separate scale tensors, SRQ activation scales, and KV cache scales.
    The output tensors map directly to Gemma4QuantizedLinear buffer names.
    """
    ckpt = _restore_checkpoint(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    text_path_prefix = "model"
    if not _TEXT_ONLY.value:
        text_path_prefix += ".language_model"

    def store(path: str, weights, dtype=None):
        """Store a tensor in the HF tree."""
        weights_np = np.asarray(weights, dtype=np.float32) if dtype is None else np.asarray(weights)
        # .copy() ensures each tensor has its own memory — safetensors drops
        # shared-memory tensors (e.g. duplicated SRQ scales for K/V).
        t = torch.from_numpy(weights_np.copy())
        if dtype is not None:
            t = t.to(dtype)
        hf_tree[path] = t.contiguous()
        if _VERBOSE.value:
            logging.info("%s shape=%s dtype=%s", path, t.shape, t.dtype)

    def store_int(path: str, weights):
        """Store an integer tensor (quantized weights).

        Handles non-standard numpy dtypes (e.g. ml_dtypes.int4) by upcasting
        to int8, since PyTorch only supports standard numpy integer types.
        """
        weights_np = np.asarray(weights)
        # ml_dtypes.int4 / int2 are not supported by torch.from_numpy().
        # Upcast to int8 which preserves the values and is torch-compatible.
        if weights_np.dtype not in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
            logging.info("Upcasting %s from %s to int8", path, weights_np.dtype)
            weights_np = weights_np.astype(np.int8)
        # .copy() ensures each tensor has its own memory — safetensors drops
        # shared-memory tensors.
        t = torch.from_numpy(weights_np.copy())
        hf_tree[path] = t.contiguous()
        if _VERBOSE.value:
            logging.info("%s shape=%s dtype=%s", path, t.shape, t.dtype)

    # Collect all quantized parameters, grouped by their parent path
    # The checkpoint has leaves like:
    #   transformer/layer_0/attn/q_einsum/w_quantized_value
    #   transformer/layer_0/attn/q_einsum/w_quantized_scale
    #   transformer/layer_0/attn/q_einsum/static_quantized_scale
    #   transformer/layer_0/attn/q_einsum/static_num_bits
    # We need to group these by their parent path to emit paired tensors.

    # For layers NOT handled by the quantizer (per_layer_input_gate, per_layer_projection,
    # embedder), we collect value+scale pairs and dequantize them to regular float weights.
    dequant_pending = {}

    for path_tuple, value in tree.flatten_with_path(ckpt):
        param = path_tuple[-1]
        if "params" in path_tuple:
            path_tuple = path_tuple[2:]
        path_tuple = path_tuple[:-1]
        path = "/".join(path_tuple) if len(path_tuple) > 1 else path_tuple[0]

        # For quantized mode, we need to handle both quantized and non-quantized params.
        # Non-quantized params (norms, scalars, embeddings) go through the regular pipeline.
        # Quantized params get preserved as-is.

        # All known quantized param names
        _ALL_QUANTIZED_PARAMS = {
            "w_quantized_value",
            "w_quantized_scale",
            "kernel_quantized_value",
            "kernel_quantized_scale",
            "static_quantized_scale",
            "static_num_bits",
            "rqv2_muls",
            "input_embedding_quantized_value",
            "input_embedding_quantized_scale",
            "per_layer_embeddings_quantized_value",
            "per_layer_embeddings_quantized_scale",
        }

        if param in _ALL_QUANTIZED_PARAMS:
            # This is a quantized parameter. Route based on whether the layer is
            # managed by the quantizer or needs dequantization.

            if path.startswith(_TRANSFORMER_PARAMETER) or path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
                # Transformer layer — try quantized conversion for attn/MLP projections
                if param in ("w_quantized_value", "w_quantized_scale", "static_quantized_scale", "static_num_bits"):
                    # SRQ scales live under nested paths like:
                    #   transformer/layer_0/attn/q_einsum/compression_einsum/input_activation
                    #   transformer/layer_0/attn/q_einsum/compression_einsum/output_activation
                    # Strip the suffix to get the base weight path, and create
                    # a virtual param name that distinguishes input vs output.
                    effective_path = path
                    effective_param = param
                    if "/compression_einsum/input_activation" in path:
                        effective_path = path.split("/compression_einsum/")[0]
                        if param == "static_quantized_scale":
                            effective_param = "static_quantized_scale_input"
                    elif "/compression_einsum/output_activation" in path:
                        effective_path = path.split("/compression_einsum/")[0]
                        if param == "static_quantized_scale":
                            effective_param = "static_quantized_scale_output"

                    results = list(
                        convert_quantized_transformer_weights(
                            config.text_config, effective_path, effective_param, value
                        )
                    )

                    if results:
                        for hf_path, tensor in results:
                            if "quantized_value" in effective_param or "num_bits" in effective_param:
                                store_int(f"{text_path_prefix}.{hf_path}", tensor)
                            else:
                                store(f"{text_path_prefix}.{hf_path}", tensor)
                    else:
                        # Layer not handled by quantizer — collect for dequantization
                        key = effective_path
                        if param == "w_quantized_value":
                            dequant_pending.setdefault(key, {})["value"] = value
                        elif param == "w_quantized_scale":
                            dequant_pending.setdefault(key, {})["scale"] = value
                else:
                    # kernel_quantized_*, rqv2_muls, input_embedding_*, per_layer_embeddings_* in transformer
                    # Collect for dequantization. Use composite key (path:prefix) to avoid
                    # collisions when multiple quantized params share the same path
                    # (e.g. input_embedding_quantized_* and per_layer_embeddings_quantized_*
                    # both live under transformer/embedder).
                    param_prefix = param.rsplit("_quantized_", 1)[0] if "_quantized_" in param else param
                    key = f"{path}:{param_prefix}"
                    if "quantized_value" in param:
                        dequant_pending.setdefault(key, {})["value"] = value
                    elif "quantized_scale" in param:
                        dequant_pending.setdefault(key, {})["scale"] = value
                    # rqv2_muls is not needed for dequantization

            elif path.startswith(_VISION_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
                _handle_encoder_quantized_param(
                    path=path,
                    param=param,
                    value=value,
                    transformer_root=_VISION_ENCODER_TRANSFORMER,
                    source_label="vision",
                    hf_prefix="model.vision_tower",
                    convert_fn=convert_quantized_vision_encoder_weights,
                    encoder_config=config.vision_config,
                    store=store,
                    store_int=store_int,
                    dequant_pending=dequant_pending,
                )

            elif path.startswith(_AUDIO_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
                _handle_encoder_quantized_param(
                    path=path,
                    param=param,
                    value=value,
                    transformer_root=_AUDIO_ENCODER_CONFORMER,
                    source_label="audio",
                    hf_prefix="model.audio_tower",
                    convert_fn=convert_quantized_audio_encoder_weights,
                    encoder_config=config.audio_config,
                    store=store,
                    store_int=store_int,
                    dequant_pending=dequant_pending,
                )

            else:
                # Other quantized params (e.g., top-level embedding)
                key = path
                if "quantized_value" in param:
                    dequant_pending.setdefault(key, {})["value"] = value
                elif "quantized_scale" in param:
                    dequant_pending.setdefault(key, {})["scale"] = value

        elif param in ("fq_static_k_cache", "fq_static_v_cache"):
            # KV cache quantization scales — sub-dicts handled by tree flattening
            pass
        elif path.endswith("fq_static_k_cache") or path.endswith("fq_static_v_cache"):
            # KV cache scales emitted to safetensors but not consumed by modeling.
            # Downstream tooling (e.g. a future quantized cache) reads them; the
            # model-side `_keys_to_ignore_on_load_unexpected` silences the warning.
            cache_type = "k_cache_scale" if "k_cache" in path else "v_cache_scale"
            if path.startswith(_TRANSFORMER_DECODER_BLOCK):
                attention_type_index = int(path[_TRANSFORMER_DECODER_BLOCK_LEN])
                num_stacks = value.shape[0] if hasattr(value, "shape") and value.ndim > 0 else 1
                for stack_i in range(num_stacks):
                    layer_idx = _SLIDING_WINDOW_PATTERN * stack_i + attention_type_index
                    val_slice = value[stack_i] if hasattr(value, "shape") and value.ndim > 0 else value
                    if param == "static_quantized_scale":
                        store(f"{text_path_prefix}.layers.{layer_idx}.self_attn.{cache_type}", val_slice)
            else:
                layer_str = path.split("/")[1]
                layer_idx = int(layer_str.replace("layer_", ""))
                if param == "static_quantized_scale":
                    store(f"{text_path_prefix}.layers.{layer_idx}.self_attn.{cache_type}", value)

        else:
            # Non-quantized params: norms, scalars, embeddings — use regular conversion
            if path.endswith("audio_embedding_norm") and not _TEXT_ONLY.value:
                store("model.embed_audio.hard_embedding_norm.weight", value)
            elif path.endswith("audio_input_projection") and not _TEXT_ONLY.value:
                store("model.embed_audio.embedding_projection.weight", value.transpose())
            elif path.endswith("audio_soft_embedding_norm") and not _TEXT_ONLY.value:
                store("model.embed_audio.soft_embedding_norm.weight", value)
            elif path.endswith("mm_hard_embedding_norm") and not _TEXT_ONLY.value:
                store("model.embed_vision.hard_embedding_norm.weight", value)
            elif path.endswith("mm_input_projection") and not _TEXT_ONLY.value:
                store("model.embed_vision.embedding_projection.weight", value.transpose())
            elif path.endswith("mm_soft_embedding_norm") and not _TEXT_ONLY.value:
                store("model.embed_vision.soft_embedding_norm.weight", value)
            elif path.startswith(_TRANSFORMER_PARAMETER):
                for hf_path, weights in convert_transformer_weights(config.text_config, path, param, value):
                    store(f"{text_path_prefix}.{hf_path}", weights, config.text_config.dtype)
            elif path.startswith(_VISION_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
                for hf_path, weights in convert_vision_encoder_weights(config.vision_config, path, param, value):
                    store(f"model.vision_tower.{hf_path}", weights, config.vision_config.dtype)
            elif path.startswith(_AUDIO_ENCODER_PARAMETER) and not _TEXT_ONLY.value:
                for hf_path, weights in convert_audio_encoder_weights(config.audio_config, path, param, value):
                    store(f"model.audio_tower.{hf_path}", weights)

    # Dequantize collected value+scale pairs from layers not handled by the quantizer
    print(f"DEQUANT-PENDING summary: {len(dequant_pending)} entries", flush=True)
    for dp_key, dp_val in dequant_pending.items():
        print(
            f"  DEQUANT-PENDING: key={dp_key} has_value={'value' in dp_val} has_scale={'scale' in dp_val} source={dp_val.get('source', 'transformer')}",
            flush=True,
        )
    for composite_key, pair in dequant_pending.items():
        if "value" not in pair or "scale" not in pair:
            logging.warning("Incomplete dequant pair for %s (have: %s), skipping", composite_key, list(pair.keys()))
            continue

        # Extract original path and param prefix from composite key
        # Composite keys: "path:param_prefix" (e.g. "transformer/embedder:input_embedding")
        # Non-composite keys: just "path" (e.g. from audio/vision branches)
        if ":" in composite_key:
            orbax_path, param_prefix = composite_key.rsplit(":", 1)
        else:
            orbax_path = composite_key
            param_prefix = "w"  # Default for non-composite keys

        # Dequantize: float_weight = int_value * scale
        value_np = np.asarray(pair["value"])
        scale_np = np.asarray(pair["scale"], dtype=np.float32)
        if value_np.dtype not in (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        ):
            value_np = value_np.astype(np.int8)
        # Squeeze scale to match value dimensionality to avoid broadcasting issues.
        # Scale from Orbax may be (N, 1, 1) but value is (N, M) — squeeze to (N, 1).
        while scale_np.ndim > value_np.ndim:
            scale_np = scale_np.squeeze(-1)
        dequantized = value_np.astype(np.float32) * scale_np
        print(
            f"Dequantized {orbax_path} (prefix={param_prefix}): value_shape={value_np.shape} scale_shape={scale_np.shape} result_shape={dequantized.shape}",
            flush=True,
        )

        # Route through the appropriate converter based on source type
        source = pair.get("source", "transformer")
        if source == "audio" and not _TEXT_ONLY.value:
            for hf_path, weights in convert_audio_encoder_weights(config.audio_config, orbax_path, "w", dequantized):
                store(f"model.audio_tower.{hf_path}", weights)
        elif source == "vision" and not _TEXT_ONLY.value:
            for hf_path, weights in convert_vision_encoder_weights(config.vision_config, orbax_path, "w", dequantized):
                store(f"model.vision_tower.{hf_path}", weights, config.vision_config.dtype)
        elif (
            orbax_path == _TRANSFORMER_EMBEDDER
            or orbax_path.startswith(_TRANSFORMER_EMBEDDER)
            or orbax_path == f"{_TRANSFORMER_POST_TRAINING_PREFIX}{_TRANSFORMER_EMBEDDER}"
            or orbax_path.startswith(f"{_TRANSFORMER_POST_TRAINING_PREFIX}{_TRANSFORMER_EMBEDDER}")
        ):
            # Embedding handling: keep as quantized when possible
            norm_path = orbax_path
            if norm_path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
                norm_path = norm_path[_TRANSFORMER_POST_TRAINING_PREFIX_LEN:]

            if param_prefix in ("per_layer_embeddings", "input_embedding"):
                if norm_path != _TRANSFORMER_EMBEDDER:
                    # Sub-path (e.g. eevee_embedder) — skip, not needed for HF model
                    print(f"SKIP: {orbax_path}:{param_prefix} (sub-path, not main embedder)", flush=True)
                    continue

                # Store embeddings as quantized (packed int + scale) instead of dequantizing
                value_np = np.asarray(pair["value"])
                scale_np = np.asarray(pair["scale"], dtype=np.float32)
                jax_dtype = str(getattr(pair["value"], "dtype", value_np.dtype))

                if param_prefix == "input_embedding":
                    # Main embedding: [262144, 1536], typically int2
                    hf_base = f"{text_path_prefix}.embed_tokens"
                    value_int8 = value_np.astype(np.int8)
                elif param_prefix == "per_layer_embeddings":
                    # Per-layer embedding: [262144, 35, 256], typically int4
                    # Reshape to [262144, 35*256] to match HF Embedding(vocab, num_layers*dim)
                    vocab_size, num_layers, hidden_dim = value_np.shape
                    value_int8 = value_np.reshape(vocab_size, num_layers * hidden_dim).astype(np.int8)
                    # Scale is (vocab_size, 35, 1) — one scale per layer per token.
                    # Keep compact as (vocab_size, 35) — the forward pass uses
                    # scale_block_size to apply block-wise dequantization.
                    scale_np = scale_np.reshape(vocab_size, num_layers)
                    hf_base = f"{text_path_prefix}.embed_tokens_per_layer"

                # Pack based on source dtype
                if "int4" in jax_dtype:
                    packed = pack_int4_to_uint8(value_int8)
                elif "int2" in jax_dtype:
                    packed = pack_int2_to_uint8(value_int8)
                else:
                    packed = value_int8

                store_int(f"{hf_base}.embedding_quantized", packed)
                store(f"{hf_base}.embedding_scale", scale_np)
                print(
                    f"Stored quantized embedding {hf_base}: packed_shape={packed.shape} "
                    f"dtype={packed.dtype} scale_shape={scale_np.shape}",
                    flush=True,
                )
            else:
                # Other embedder sub-paths (per_layer_model_projection, etc.) — dequantize as before
                conv_param = param_prefix
                conv_path = norm_path
                for hf_path, weights in convert_transformer_weights(
                    config.text_config, conv_path, conv_param, dequantized
                ):
                    store(f"{text_path_prefix}.{hf_path}", weights, config.text_config.dtype)
        else:
            # Transformer layers (per_layer_input_gate, per_layer_projection, etc.)
            norm_path = orbax_path
            if norm_path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
                norm_path = norm_path[_TRANSFORMER_POST_TRAINING_PREFIX_LEN:]
            for hf_path, weights in convert_transformer_weights(config.text_config, norm_path, "w", dequantized):
                store(f"{text_path_prefix}.{hf_path}", weights, config.text_config.dtype)

    # Store lm_head as quantized (shares quantized data with embedding).
    # `QuantizedLinear` stores its packed-int storage at `.weight`.
    emb_q_key = f"{text_path_prefix}.embed_tokens.embedding_quantized"
    emb_s_key = f"{text_path_prefix}.embed_tokens.embedding_scale"
    if emb_q_key in hf_tree and emb_s_key in hf_tree:
        hf_tree["lm_head.weight"] = hf_tree[emb_q_key].clone()
        hf_tree["lm_head.weight_scale"] = hf_tree[emb_s_key].clone()
        # No SRQ activation scales for lm_head
        hf_tree["lm_head.input_activation_scale"] = torch.tensor(0.0, dtype=torch.float32)
        hf_tree["lm_head.output_activation_scale"] = torch.tensor(0.0, dtype=torch.float32)
        logging.info(
            "Stored quantized lm_head: weight=%s scale=%s",
            hf_tree["lm_head.weight"].shape,
            hf_tree["lm_head.weight_scale"].shape,
        )
    else:
        # Fallback: non-quantized embedding — copy as float weight
        emb_key = f"{text_path_prefix}.embed_tokens.weight"
        if emb_key in hf_tree:
            hf_tree["lm_head.weight"] = hf_tree[emb_key].clone()

    return hf_tree


def pack_int4_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Pack int8 array (with values in [-8, 7]) into uint8 with 2 values per byte.

    Each pair of consecutive values along the last axis is packed into one byte:
      low nibble  = (val[0] + 8) & 0x0F
      high nibble = (val[1] + 8) << 4

    Args:
        arr: numpy int8 array with values in [-8, 7], last dim must be even.

    Returns:
        numpy uint8 array with last dim halved.
    """
    assert arr.dtype == np.int8, f"Expected int8 array, got {arr.dtype}"
    rows = arr.shape[0]
    cols = arr.shape[1]
    # Pad last dim to even if needed
    if cols % 2 != 0:
        arr = np.pad(arr, ((0, 0), (0, 1)), constant_values=0)
        cols = arr.shape[1]
    # Reshape to (..., cols//2, 2)
    pairs = arr.reshape(rows, cols // 2, 2)
    low = (pairs[..., 0].astype(np.uint8) + 8) & 0x0F
    high = ((pairs[..., 1].astype(np.uint8) + 8) & 0x0F) << 4
    return (low | high).astype(np.uint8)


def pack_int2_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Pack int8 array (with values in [-2, 1]) into uint8 with 4 values per byte.

    Each group of 4 consecutive values along the last axis is packed into one byte:
      bits 0-1 = (val[0] + 2) & 0x03
      bits 2-3 = (val[1] + 2) & 0x03
      bits 4-5 = (val[2] + 2) & 0x03
      bits 6-7 = (val[3] + 2) & 0x03

    Args:
        arr: numpy int8 array with values in [-2, 1], last dim must be divisible by 4.

    Returns:
        numpy uint8 array with last dim quartered.
    """
    assert arr.dtype == np.int8, f"Expected int8 array, got {arr.dtype}"
    orig_shape = arr.shape
    last_dim = orig_shape[-1]
    # Pad last dim to multiple of 4 if needed
    if last_dim % 4 != 0:
        pad_width = [(0, 0)] * (len(orig_shape) - 1) + [(0, 4 - last_dim % 4)]
        arr = np.pad(arr, pad_width, constant_values=0)
        last_dim = arr.shape[-1]
    # Reshape to (..., last_dim//4, 4)
    groups = arr.reshape(*orig_shape[:-1], last_dim // 4, 4)
    v0 = (groups[..., 0].astype(np.uint8) + 2) & 0x03
    v1 = ((groups[..., 1].astype(np.uint8) + 2) & 0x03) << 2
    v2 = ((groups[..., 2].astype(np.uint8) + 2) & 0x03) << 4
    v3 = ((groups[..., 3].astype(np.uint8) + 2) & 0x03) << 6
    return (v0 | v1 | v2 | v3).astype(np.uint8)


def unpack_uint8_to_int4(arr: np.ndarray) -> np.ndarray:
    """Unpack uint8 array to int8 with 2 values per byte (inverse of pack_int4_to_uint8).

    Each byte is split into low nibble and high nibble:
      val[0] = (byte & 0x0F) - 8
      val[1] = (byte >> 4) - 8

    Args:
        arr: numpy uint8 array (packed int4 values).

    Returns:
        numpy int8 array with last dim doubled.
    """
    arr = arr.astype(np.uint8)
    low = (arr & 0x0F).astype(np.int8) - 8
    high = (arr >> 4).astype(np.int8) - 8
    return np.stack([low, high], axis=-1).reshape(*arr.shape[:-1], arr.shape[-1] * 2)


def unpack_uint8_to_int2(arr: np.ndarray) -> np.ndarray:
    """Unpack uint8 array to int8 with 4 values per byte (inverse of pack_int2_to_uint8).

    Each byte is split into 4 crumbs (2-bit values):
      val[0] = (byte & 0x03) - 2
      val[1] = ((byte >> 2) & 0x03) - 2
      val[2] = ((byte >> 4) & 0x03) - 2
      val[3] = ((byte >> 6) & 0x03) - 2

    Args:
        arr: numpy uint8 array (packed int2 values).

    Returns:
        numpy int8 array with last dim quadrupled.
    """
    arr = arr.astype(np.uint8)
    v0 = (arr & 0x03).astype(np.int8) - 2
    v1 = ((arr >> 2) & 0x03).astype(np.int8) - 2
    v2 = ((arr >> 4) & 0x03).astype(np.int8) - 2
    v3 = ((arr >> 6) & 0x03).astype(np.int8) - 2
    return np.stack([v0, v1, v2, v3], axis=-1).reshape(*arr.shape[:-1], arr.shape[-1] * 4)


def convert_quantized_transformer_weights(
    config,
    path: str,
    param: str,
    weights,
) -> Iterable[tuple[str, np.ndarray]]:
    """Map quantized Orbax weights to HF QuantizedLinear storage names.

    For each weight layer like attn/q_einsum:
        w_quantized_value             -> layers.N.self_attn.q_proj.weight
        w_quantized_scale             -> layers.N.self_attn.q_proj.weight_scale
        static_quantized_scale_input  -> layers.N.self_attn.q_proj.input_activation_scale
        static_quantized_scale_output -> layers.N.self_attn.q_proj.output_activation_scale
    """
    if path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
        path = path[_TRANSFORMER_POST_TRAINING_PREFIX_LEN:]

    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    # Handle both unstacked (transformer/layer_X/...) and stacked
    # (transformer/stacked_layers/attention_type_X/...) layer formats.
    is_stacked = False
    if path.startswith(f"{_TRANSFORMER_PARAMETER}/layer_"):
        # Unstacked format: transformer/layer_0/attn/q_einsum
        layer_str = path.split("/")[1]
        layer_idx = int(layer_str.replace("layer_", ""))
        base_path = f"layers.{layer_idx}"
    elif path.startswith(_TRANSFORMER_DECODER_BLOCK):
        # Stacked format: transformer/stacked_layers/attention_type_0/attn/q_einsum
        # Weights have a leading stack dimension [num_stacks, ...] that needs unstacking.
        is_stacked = True
        attention_type_index = int(path[_TRANSFORMER_DECODER_BLOCK_LEN])
        # For stacked, we iterate below — set initial layer_idx to 0 for head_dim
        layer_idx = attention_type_index
        base_path = f"layers.{layer_idx}"  # will be updated per-stack
    else:
        return []

    # For stacked layers, immediately unroll and recurse with per-layer slices.
    # This must happen BEFORE head_dim computation and path-specific reshape logic,
    # because the stacked weights have an extra leading dimension.
    if is_stacked:
        all_paths = []
        all_weights = []
        # Detect int4/int2 from JAX dtype BEFORE converting to numpy (which loses dtype info)
        is_value = param == "w_quantized_value"
        weights_arr = np.asarray(weights)
        # Scalar params (SRQ scales, num_bits) are broadcast to all layers
        if weights_arr.ndim == 0:
            num_stacks = config.num_hidden_layers // _SLIDING_WINDOW_PATTERN
            for stack_i in range(num_stacks):
                stack_layer_idx = _SLIDING_WINDOW_PATTERN * stack_i + attention_type_index
                fake_path = f"{_TRANSFORMER_PARAMETER}/layer_{stack_layer_idx}/{'/'.join(path.split('/')[3:])}"
                for rp, rw in convert_quantized_transformer_weights(config, fake_path, param, weights_arr):
                    all_paths.append(rp)
                    all_weights.append(rw)
        else:
            num_stacks = weights_arr.shape[0]
            for stack_i in range(num_stacks):
                stack_layer_idx = _SLIDING_WINDOW_PATTERN * stack_i + attention_type_index
                fake_path = f"{_TRANSFORMER_PARAMETER}/layer_{stack_layer_idx}/{'/'.join(path.split('/')[3:])}"
                slice_w = weights_arr[stack_i]
                # Note: np.asarray() already converts JAX int4/int2 to int8 with
                # the correct logical shape. No manual unpacking needed here.
                for rp, rw in convert_quantized_transformer_weights(config, fake_path, param, slice_w):
                    all_paths.append(rp)
                    all_weights.append(rw)

        # Note: np.asarray() preserves JAX int4/int2 dtype, so the recursive
        # calls correctly detect int4 and apply packing + num_bits=4 emission.
        # No additional post-processing needed here.
        return zip(all_paths, all_weights)

    head_dim = config.head_dim if config.layer_types[layer_idx] == "sliding_attention" else config.global_head_dim

    # Map the quantized parameter suffix to the HF buffer name suffix.
    # The main routing loop may pass virtual param names with _input/_output
    # suffixes for SRQ scales to distinguish input vs output activation.
    # NOTE: static_num_bits from the checkpoint refers to ACTIVATION precision,
    # not weight bits. The actual weight bit width is carried by the JAX dtype
    # of the value tensor (int4 / int8) and used by the int4 packing post-pass;
    # we don't emit it as a buffer because `QuantizedLinear` reads it from
    # `quantization_config.module_quant_configs` at load time.
    # `QuantizedLinear` subclasses `nn.Linear` and stores its packed-int storage
    # at `.weight`.
    param_suffix_map = {
        "w_quantized_value": "weight",
        "w_quantized_scale": "weight_scale",
        "static_quantized_scale": "input_activation_scale",  # legacy fallback
        "static_quantized_scale_input": "input_activation_scale",
        "static_quantized_scale_output": "output_activation_scale",
    }

    hf_suffix = param_suffix_map.get(param)
    if hf_suffix is None:
        return []

    # Classify param type for reshape decisions:
    # - value: full reshape/transpose matching the non-quantized weight layout
    # - scale: per-channel, just squeeze extra dims to 1D
    # - scalar: pass through unchanged (SRQ activation scale, num_bits)
    is_value = param == "w_quantized_value"
    is_scale = param == "w_quantized_scale"

    # Detect int4/int2 source dtype for packing
    is_int4 = False
    is_int2 = False
    if is_value:
        jax_dtype = str(getattr(weights, "dtype", ""))
        is_int4 = "int4" in jax_dtype
        is_int2 = "int2" in jax_dtype

    def squeeze_scale(w):
        """Reshape per-channel scale to (N, 1) for broadcasting with 2D weight.

        Orbax scales may be (N, 1, 1) or (N,) — normalize to (N, 1) to match
        the buffer shape in Gemma4QuantizedLinear.
        """
        flat = w.reshape(-1)
        return flat.reshape(-1, 1)

    # Determine the HF module path based on the Orbax path
    def reshape_q(w):
        """q_einsum: [hidden, num_heads, head_dim] -> [num_heads*head_dim, hidden]"""
        if w.ndim == 3:
            return w.transpose(1, 0, 2).reshape(config.hidden_size, config.num_attention_heads * head_dim).transpose()
        elif w.ndim == 2:
            return w.transpose()
        return w

    def reshape_kv_single(w):
        """Single K or V: [num_kv_heads, hidden, head_dim] -> [num_kv_heads*head_dim, hidden]"""
        if w.ndim == 3:
            kv_shape = (config.hidden_size, config.num_key_value_heads * head_dim)
            return w.transpose(1, 0, 2).reshape(kv_shape).transpose()
        elif w.ndim == 2:
            return w.transpose()
        return w

    def reshape_o(w):
        """attn_vec_einsum: [num_heads, head_dim, hidden] -> [hidden, num_heads*head_dim]"""
        if w.ndim == 3:
            return w.transpose(2, 0, 1).reshape(config.hidden_size, config.num_attention_heads * head_dim)
        return w

    def reshape_down(w):
        """mlp/linear: [intermediate, hidden] -> [hidden, intermediate]"""
        if w.ndim == 2:
            return w.transpose()
        return w

    if path.endswith("attn/q_einsum"):
        if is_value:
            w = reshape_q(weights)
        elif is_scale:
            w = squeeze_scale(weights)
        else:
            w = weights
        converted_paths.append(f"{base_path}.self_attn.q_proj.{hf_suffix}")
        converted_weights.append(w)
    elif path.endswith("attn/kv_einsum"):
        if is_value:
            # Split K/V along first axis: [2, hidden, num_kv_heads, head_dim]
            k_part = reshape_kv_single(weights[0])
            v_part = reshape_kv_single(weights[1])
            converted_paths.extend(
                [
                    f"{base_path}.self_attn.k_proj.{hf_suffix}",
                    f"{base_path}.self_attn.v_proj.{hf_suffix}",
                ]
            )
            converted_weights.extend([k_part, v_part])
        elif is_scale:
            # Per-channel scale: split and squeeze
            k_part = squeeze_scale(weights[0])
            v_part = squeeze_scale(weights[1])
            converted_paths.extend(
                [
                    f"{base_path}.self_attn.k_proj.{hf_suffix}",
                    f"{base_path}.self_attn.v_proj.{hf_suffix}",
                ]
            )
            converted_weights.extend([k_part, v_part])
        else:
            # Scalar SRQ scales: duplicate for K and V
            converted_paths.extend(
                [
                    f"{base_path}.self_attn.k_proj.{hf_suffix}",
                    f"{base_path}.self_attn.v_proj.{hf_suffix}",
                ]
            )
            converted_weights.extend([weights, weights])
    elif path.endswith("attn/k_einsum"):

        def reshape_k(w):
            if w.ndim == 3:
                return (
                    w.transpose(1, 0, 2)
                    .reshape(config.hidden_size, config.num_global_key_value_heads * head_dim)
                    .transpose()
                )
            elif w.ndim == 2:
                return w.transpose()
            return w

        if is_value:
            w = reshape_k(weights)
        elif is_scale:
            w = squeeze_scale(weights)
        else:
            w = weights
        converted_paths.append(f"{base_path}.self_attn.k_proj.{hf_suffix}")
        converted_weights.append(w)
    elif path.endswith("attn/attn_vec_einsum"):
        if is_value:
            w = reshape_o(weights)
        elif is_scale:
            w = squeeze_scale(weights)
        else:
            w = weights
        converted_paths.append(f"{base_path}.self_attn.o_proj.{hf_suffix}")
        converted_weights.append(w)
    elif path.endswith("mlp/gating_einsum"):
        if is_value:
            # Split gate/up: [2, intermediate, hidden]
            gate_part, up_part = weights[0], weights[1]
            converted_paths.extend(
                [
                    f"{base_path}.mlp.gate_proj.{hf_suffix}",
                    f"{base_path}.mlp.up_proj.{hf_suffix}",
                ]
            )
            converted_weights.extend([gate_part, up_part])
        elif is_scale:
            # Per-channel scale: split and squeeze
            gate_part = squeeze_scale(weights[0])
            up_part = squeeze_scale(weights[1])
            converted_paths.extend(
                [
                    f"{base_path}.mlp.gate_proj.{hf_suffix}",
                    f"{base_path}.mlp.up_proj.{hf_suffix}",
                ]
            )
            converted_weights.extend([gate_part, up_part])
        else:
            # Scalar SRQ params: duplicate for gate and up
            converted_paths.extend(
                [
                    f"{base_path}.mlp.gate_proj.{hf_suffix}",
                    f"{base_path}.mlp.up_proj.{hf_suffix}",
                ]
            )
            converted_weights.extend([weights, weights])
    elif path.endswith("mlp/linear"):
        if is_value:
            w = reshape_down(weights)
        elif is_scale:
            w = squeeze_scale(weights)
        else:
            w = weights
        converted_paths.append(f"{base_path}.mlp.down_proj.{hf_suffix}")
        converted_weights.append(w)
    elif path.endswith("per_layer_input_gate"):
        # per_layer_input_gate: JAX [hidden, per_layer_dim] → HF [per_layer_dim, hidden]
        if is_value:
            w = weights.transpose() if weights.ndim == 2 else weights
        elif is_scale:
            w = squeeze_scale(weights)
        else:
            w = weights
        converted_paths.append(f"{base_path}.per_layer_input_gate.{hf_suffix}")
        converted_weights.append(w)
    elif path.endswith("per_layer_projection"):
        # per_layer_projection: JAX [per_layer_dim, hidden] → HF [hidden, per_layer_dim]
        if is_value:
            w = weights.transpose() if weights.ndim == 2 else weights
        elif is_scale:
            w = squeeze_scale(weights)
        else:
            w = weights
        converted_paths.append(f"{base_path}.per_layer_projection.{hf_suffix}")
        converted_weights.append(w)
    elif path.endswith("fq_static_k_cache") or path.endswith("fq_static_v_cache"):
        cache_type = "k_cache_scale" if "k_cache" in path else "v_cache_scale"
        if param in ("static_quantized_scale", "static_quantized_scale_input"):
            converted_paths.append(f"{base_path}.self_attn.{cache_type}")
            converted_weights.append(weights)
    else:
        return []

    # Post-processing: pack int4/int2 weights and emit num_bits metadata
    if is_value:
        if is_int4:
            # Convert JAX int4 to numpy int8, then pack into uint8.
            converted_weights = [pack_int4_to_uint8(np.asarray(w).astype(np.int8)) for w in converted_weights]
        elif is_int2:
            # Convert JAX int2 to numpy int8, then pack into uint8.
            converted_weights = [pack_int2_to_uint8(np.asarray(w).astype(np.int8)) for w in converted_weights]

    return zip(converted_paths, converted_weights)


def _verify_quant_bits(state_tree, module_quant_configs, config):
    """Cross-check `module_quant_configs` against the converted state_tree.

    Each quantized weight's storage layout (uint8 packed at half/quarter width
    or int8 full width) determines how `QuantizedLinear` decodes the bytes at
    load time. That implied bit width has to match the regex map; if it
    doesn't, the loader silently mis-decodes and inference produces garbage.
    """
    import re

    with torch.device("meta"):
        model = (
            Gemma4ForCausalLM(config=config.text_config)
            if _TEXT_ONLY.value
            else Gemma4ForConditionalGeneration(config=config)
        )
    linears = {n: m.in_features for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    embeds = {n: m.embedding_dim for n, m in model.named_modules() if isinstance(m, torch.nn.Embedding)}

    def storage_to_bits(t, in_features):
        _, packed = t.shape
        if t.dtype == torch.int8 and packed == in_features:
            return 8
        if t.dtype == torch.uint8 and packed == (in_features + 1) // 2:
            return 4
        if t.dtype == torch.uint8 and packed == (in_features + 3) // 4:
            return 2
        return None

    def resolve_bits(name):
        for pattern, opts in module_quant_configs.items():
            if re.search(pattern, name):
                return int(opts.get("num_bits", 4))
        return 4

    mismatches = []
    checked = 0
    for k, t in state_tree.items():
        if t.dtype not in (torch.uint8, torch.int8) or t.ndim != 2:
            continue
        if k.endswith(".weight"):
            module_name = k.removesuffix(".weight")
            in_w = linears.get(module_name)
            cfg_lookup = module_name.removesuffix(".linear")
        elif k.endswith(".embedding_quantized"):
            module_name = k.removesuffix(".embedding_quantized")
            in_w = embeds.get(module_name)
            cfg_lookup = module_name
        else:
            continue
        if in_w is None:
            continue
        checked += 1
        bits_storage = storage_to_bits(t, in_w)
        bits_config = resolve_bits(cfg_lookup.removeprefix("model."))
        if bits_storage != bits_config:
            mismatches.append((k, bits_storage, bits_config))

    if mismatches:
        for k, s, c in mismatches[:20]:
            logging.error("num_bits mismatch: %s storage=%s config=%s", k, s, c)
        raise SystemExit(
            f"module_quant_configs disagrees with the converted checkpoint on "
            f"{len(mismatches)} tensor(s). See logs above."
        )
    logging.info("Verified num_bits on %d quantized tensors against the checkpoint storage.", checked)


def main(*args):
    del args

    output_path = _OUTPUT_PATH.value
    variant = _VARIANT.value

    config = _VARIANTS[variant]
    is_drafter = variant in _ASSISTANT_VARIANTS

    logging.info("Converting Gemma 4 variant: %s", variant)

    config.get_text_config().dtype = getattr(torch, _TEXT_DTYPE.value)

    if not is_drafter:
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
    logging.info("Converted Gemma 4 (%s) state tree from Orbax to Hugging Face.", variant)

    if _QUANTIZED.value:
        state_tree = convert_quantized(_CHECKPOINT_PATH.value, config)
        # Ensure every quantized weight has companion SRQ scales (uncalibrated
        # layers store 0.0 placeholders). Identify quantized layers by the
        # int8/uint8 dtype of their `.weight`.
        for k in list(state_tree.keys()):
            if not k.endswith(".weight") or state_tree[k].dtype not in (torch.uint8, torch.int8):
                continue
            base_path = k.removesuffix(".weight")
            for scale_suffix in (".input_activation_scale", ".output_activation_scale"):
                state_tree.setdefault(f"{base_path}{scale_suffix}", torch.tensor(0.0, dtype=torch.float32))

        # Per-module bit widths and the modules that the orbax checkpoint stores
        # un-quantized. Patterns are regexes matched with `re.search`. The dict
        # is serialized with `sort_keys=True` by `PretrainedConfig`, so insertion
        # order is lost — anchor patterns end-of-name (`$`) or with `\.` so
        # ordering doesn't matter.
        if "e4b" in variant:
            module_quant_configs = {
                # Embeddings — int2 across the board for E4B.
                r"language_model\.embed_tokens_per_layer$": {"num_bits": 2},
                r"language_model\.embed_tokens$": {"num_bits": 2},
                r"^lm_head$": {"num_bits": 2},
                # PLE Linear Layers — int8.
                r"language_model\.layers\.\d+\.per_layer_input_gate$": {"num_bits": 8},
                r"language_model\.layers\.\d+\.per_layer_projection$": {"num_bits": 8},
                # Vision tower — int8.
                r"vision_tower": {"num_bits": 8},
                # Audio tower — int2 for most layers, int4 for lconv/linear_start.
                # The negative lookahead sorts before the specific pattern
                # (since '(' < '\') ensuring correct resolve order after
                # JSON sort_keys=True serialization.
                r"audio_tower(?!.*lconv1d\.linear_start)": {"num_bits": 2},
                r"audio_tower\.layers\.\d+\.lconv1d\.linear_start\.": {"num_bits": 4},
                # Language model attention + MLP — int4 across all layers.
                r"language_model\.layers\.\d+\.self_attn\.": {"num_bits": 4},
                r"language_model\.layers\.\d+\.mlp\.": {"num_bits": 4},
            }
        else:  # E2B
            module_quant_configs = {
                r"language_model\.embed_tokens_per_layer$": {"num_bits": 4},
                r"language_model\.embed_tokens$": {"num_bits": 2},
                r"^lm_head$": {"num_bits": 2},
                r"language_model\.layers\.\d+\.per_layer_input_gate$": {"num_bits": 8},
                r"language_model\.layers\.\d+\.per_layer_projection$": {"num_bits": 8},
                r"vision_tower": {"num_bits": 8},
                # Audio tower — int2 for most layers, int4 for lconv/linear_start.
                r"audio_tower(?!.*lconv1d\.linear_start)": {"num_bits": 2},
                r"audio_tower\.layers\.\d+\.lconv1d\.linear_start\.": {"num_bits": 4},
                r"language_model\.layers\.\d+\.self_attn\.": {"num_bits": 4},
                # MLP: int4 for layers 0-14, int2 for layers 15+. The layer-range
                # pattern alphabetizes before `\d+` (the `(` sorts before `\`),
                # so the more specific match wins after the JSON sort.
                r"language_model\.layers\.(\d|1[0-4])\.mlp\.": {"num_bits": 4},
                r"language_model\.layers\.\d+\.mlp\.": {"num_bits": 2},
            }
        modules_to_not_convert = [
            "model.vision_tower.patch_embedder",
            "model.audio_tower.subsample_conv_projection",
            "model.audio_tower.output_proj",
            "relative_k_proj",
            "model.embed_audio",
            "model.embed_vision",
            "per_layer_model_projection",
        ]
        # Sanity check: every quantized tensor's storage layout (dtype + shape)
        # must agree with the bit width `module_quant_configs` resolves to.
        # Storage is what `QuantizedLinear` decodes at load time; this catches
        # config drift before users see gibberish.
        _verify_quant_bits(state_tree, module_quant_configs, config)

        # Save tensors directly to safetensors and write config.json manually.
        os.makedirs(output_path, exist_ok=True)
        save_file(state_tree, os.path.join(output_path, "model.safetensors"))
        logging.info("Saved %d quantized tensors to safetensors", len(state_tree))

        quant_config = GemmaQuantizationConfig(
            quantize_embeddings=True,
            module_quant_configs=module_quant_configs,
            modules_to_not_convert=modules_to_not_convert,
        )
        config.quantization_config = quant_config.to_dict()
        config.tie_word_embeddings = False
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.tie_word_embeddings = False
        # The quantized checkpoint has no calibrated clip bounds, so the clip
        # buffers in `Gemma4ClippableLinear` would be ±inf no-ops at runtime.
        # Disable them so the load report doesn't spam MISSING for them and the
        # forward path skips two extra clamps per call.
        if hasattr(config, "audio_config") and config.audio_config is not None:
            config.audio_config.use_clipped_linears = False
        if hasattr(config, "vision_config") and config.vision_config is not None:
            config.vision_config.use_clipped_linears = False
        config.save_pretrained(output_path)
        logging.info("Saved config with quantization_config to %s", output_path)

        del state_tree
    else:
        state_tree = convert(_CHECKPOINT_PATH.value, config)

        with accelerate.init_empty_weights():
            if is_drafter:
                model = Gemma4AssistantForCausalLM(config=config)
            elif _TEXT_ONLY.value:
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
        model.save_pretrained(output_path, safe_serialization=True)
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
    response_format_kwargs = {"response_format": _RESPONSE_FORMAT} if _INCLUDE_RESPONSE_FORMAT.value else {}

    # Load the tokenizer from either a SentencePiece `.model` file (when the
    # path exists locally) or directly from a HF repo (e.g. `google/gemma-4-E2B-it`).
    # The HF-repo path lets us re-convert the weights without needing the
    # original SentencePiece file.
    if os.path.isfile(_TOKENIZER_PATH.value):
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
            **response_format_kwargs,
        )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            _TOKENIZER_PATH.value, **chat_template_kwargs, **response_schema_kwargs
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
        # Save the audio feature extractor separately
        feature_extractor.save_pretrained(output_path)

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
