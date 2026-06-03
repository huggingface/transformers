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

r"""Utility to convert Gemma4 Unified models from Orbax to HF Transformers checkpoint.

This script handles the unified architecture which has NO vision tower and NO
audio tower. Instead, vision inputs use a direct patch embedding pipeline
(LN → Dense → LN → +posemb → LN → RMSNorm → Linear) and audio inputs project
raw waveform frames directly through a multimodal embedder (RMSNorm → Linear).

The text/transformer weight mapping is identical to the standard Gemma4 conversion
(minus MoE and PLE).

Usage:
    python src/transformers/models/gemma4_unified/convert_gemma4_unified_weights.py \
        --variant='gemma-4-12b' \
        --include_chat_template \
        --include_response_schema \
        --tokenizer_path="$HOME/tokenizers/gemma4/gemma4_cleaned_262144.model" \
        --checkpoint_path="$HOME/gemma4/checkpoints/gemma4_12b_orbax" \
        --output_path="$HOME/gemma4/checkpoints/gemma4_12b_safetensors"
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
    Gemma4UnifiedAudioConfig,
    Gemma4UnifiedAudioFeatureExtractor,
    Gemma4UnifiedConfig,
    Gemma4UnifiedForCausalLM,
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedImageProcessor,
    Gemma4UnifiedProcessor,
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVideoProcessor,
    Gemma4UnifiedVisionConfig,
    GemmaTokenizer,
    GenerationConfig,
    RopeParameters,
)
from transformers.models.gemma4_unified_assistant.configuration_gemma4_unified_assistant import (
    Gemma4UnifiedAssistantConfig,
)
from transformers.models.gemma4_unified_assistant.modeling_gemma4_unified_assistant import (
    Gemma4UnifiedAssistantForCausalLM,
)
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor
from transformers.utils.hub import cached_file


# ==== Internal Constants and Classes ====

# The correct chat templates were already uploaded to those 2 repos, so download from there
_CHAT_TEMPLATE = pathlib.Path(cached_file("gg-hf-gg/gemma-4-31B-it", "chat_template.jinja")).read_text()

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

_TRANSFORMER_PARAMETER = "transformer"
_TRANSFORMER_DECODER_BLOCK = f"{_TRANSFORMER_PARAMETER}/stacked_layers/attention_type_"
_TRANSFORMER_DECODER_BLOCK_LEN = len(_TRANSFORMER_DECODER_BLOCK)
_TRANSFORMER_EMBEDDER = f"{_TRANSFORMER_PARAMETER}/embedder"
_TRANSFORMER_FINAL_NORM = "transformer/final_norm"
_TRANSFORMER_POST_TRAINING_PREFIX = "rlx_networks/policy_network/"
_TRANSFORMER_POST_TRAINING_PREFIX_LEN = len(_TRANSFORMER_POST_TRAINING_PREFIX)

_VARIANT_GEMMA_4_12B = "gemma-4-12b"
_VARIANT_GEMMA_4_12B_ASSISTANT = "gemma-4-12b-assistant"

_TRANSFORMER_PRE_PROJ_MTP = "transformer/pre_proj"
_TRANSFORMER_POST_PROJ_MTP = "transformer/post_proj"
_TRANSFORMER_NORM_MTP = "transformer/norm"

_ASSISTANT_VARIANTS = {
    _VARIANT_GEMMA_4_12B_ASSISTANT,
}

_DEFAULT_AUDIO_CONFIG = Gemma4UnifiedAudioConfig()

_DEFAULT_VISION_CONFIG = Gemma4UnifiedVisionConfig()

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

_GEMMA_4_12B_LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]


_VARIANTS: Mapping[str, Gemma4UnifiedConfig | Gemma4UnifiedAssistantConfig] = {
    _VARIANT_GEMMA_4_12B_ASSISTANT: Gemma4UnifiedAssistantConfig(
        backbone_hidden_size=3840,
        use_ordered_embeddings=False,
        text_config=Gemma4UnifiedTextConfig(
            hidden_size=1024,
            intermediate_size=8192,
            num_hidden_layers=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            num_attention_heads=16,
            num_key_value_heads=8,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            num_kv_shared_layers=4,
            sliding_window=1024,
            rope_parameters=_ROPE_PARAMS,
            use_double_wide_mlp=False,
            # BC: not used by the model
            vocab_size_per_layer_input=0,
            hidden_size_per_layer_input=0,
            enable_moe_block=False,
            num_experts=None,
            top_k_experts=None,
            moe_intermediate_size=None,
        ),
    ),
    _VARIANT_GEMMA_4_12B: Gemma4UnifiedConfig(
        text_config=Gemma4UnifiedTextConfig(
            hidden_size=3840,
            intermediate_size=4 * 3840,
            num_hidden_layers=48,
            layer_types=_GEMMA_4_12B_LAYER_TYPES * 8,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            final_logit_softcapping=30.0,
            rope_parameters=_ROPE_PARAMS,
            # BC: not used by the model
            vocab_size_per_layer_input=0,
            hidden_size_per_layer_input=0,
            enable_moe_block=False,
            num_experts=None,
            top_k_experts=None,
            moe_intermediate_size=None,
        ),
        vision_config=_DEFAULT_VISION_CONFIG,
        audio_config=_DEFAULT_AUDIO_CONFIG,
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
    help="If True, saves a Gemma4UnifiedForCasualLM model instead of a Gemma4UnifiedForConditionalGeneration model.",
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


def convert_embedder_weights(
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray, str]]:
    """Convert unified embedder weights (vision patch embedding, positional embedding, etc.).

    The unified model embeds vision and audio directly through the transformer embedder
    rather than separate encoder towers. This function maps the embedder sub-paths to the
    corresponding HuggingFace module paths.

    Returns:
        Iterable of (hf_path, converted_weights, dtype_group) tuples, where dtype_group is
        one of 'vision', 'audio', or 'text' to control output dtype.
    """
    converted: list[tuple[str, np.ndarray, str]] = []

    if path.endswith("mm_patch_embed_ln1"):
        if param == "scale":
            converted.append(("model.vision_embedder.patch_ln1.weight", weights, "vision"))
        elif param == "bias":
            converted.append(("model.vision_embedder.patch_ln1.bias", weights, "vision"))

    elif path.endswith("mm_patch_embed_dense"):
        if param == "kernel":
            converted.append(("model.vision_embedder.patch_dense.weight", weights.transpose(), "vision"))
        elif param == "bias":
            converted.append(("model.vision_embedder.patch_dense.bias", weights, "vision"))

    elif path.endswith("mm_patch_embed_ln2"):
        if param == "scale":
            converted.append(("model.vision_embedder.patch_ln2.weight", weights, "vision"))
        elif param == "bias":
            converted.append(("model.vision_embedder.patch_ln2.bias", weights, "vision"))

    elif path.endswith("mm_encoder_norm"):
        # This is the pos_norm (LayerNorm after adding positional embeddings)
        if param == "scale":
            converted.append(("model.vision_embedder.pos_norm.weight", weights, "vision"))
        elif param == "bias":
            converted.append(("model.vision_embedder.pos_norm.bias", weights, "vision"))

    elif path == _TRANSFORMER_EMBEDDER and param == "mm_pos_embedding":
        # Factorized 2D positional embedding: (mm_posemb_size, 2, mm_embed_dim)
        converted.append(("model.vision_embedder.pos_embedding", weights, "vision"))

    elif path.endswith("mm_input_projection"):
        # Vision multimodal embedder projection: RMSNorm → Linear
        if param == "w":
            converted.append(("model.embed_vision.embedding_projection.weight", weights.transpose(), "vision"))

    elif path.endswith("audio_input_projection"):
        # Audio multimodal embedder projection: RMSNorm → Linear
        if param == "w":
            converted.append(("model.embed_audio.embedding_projection.weight", weights.transpose(), "audio"))

    # NOTE: mm_input_embedding_extra and audio_input_embedding_extra are captured
    # in the main convert() loop (not here) for EOA/EOI embedding unification.
    # See the post-loop embedding table patching section in convert().

    return converted


def convert_transformer_weights(
    config: Gemma4UnifiedTextConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    if path.startswith(_TRANSFORMER_POST_TRAINING_PREFIX):
        path = path[_TRANSFORMER_POST_TRAINING_PREFIX_LEN:]

    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    # Handle new checkpoint format: transformer/layer_N/...
    if path.startswith(f"{_TRANSFORMER_PARAMETER}/layer_"):
        # Extract layer number from path like "transformer/layer_0/attn/q_einsum"
        layer_str = path.split("/")[1]  # "layer_0"
        layer_idx = int(layer_str.replace("layer_", ""))  # 0
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
        base_path = f"layers.{layer_idx}"

        # Determine head_dim from actual checkpoint weight dimensions
        if path.endswith("attn/key_norm") or path.endswith("attn/query_norm"):
            head_dim = weights.shape[0]
        elif path.endswith("attn/q_einsum"):
            head_dim = weights.shape[-1]
        else:
            head_dim = (
                config.global_head_dim
                if config.layer_types[layer_idx] == "full_attention" and config.global_head_dim
                else config.head_dim
            )

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
            elif path.endswith("attn/kv_einsum"):
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
            elif path.endswith("attn/k_einsum"):
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
            elif path.endswith("attn/key_norm"):
                converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
                converted_weights.append(matrix)
            elif path.endswith("mlp/gating_einsum"):
                converted_paths.extend([f"{base_path}.mlp.gate_proj.weight", f"{base_path}.mlp.up_proj.weight"])
                gate_proj_weight, up_proj_weight = matrix
                converted_weights.extend([gate_proj_weight, up_proj_weight])
            elif path.endswith("mlp/linear"):
                converted_paths.append(f"{base_path}.mlp.down_proj.weight")
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
    elif path == _TRANSFORMER_EMBEDDER:
        if param == "input_embedding":
            converted_paths.append("embed_tokens.weight")
            converted_weights.append(weights)
    elif path == _TRANSFORMER_FINAL_NORM:
        converted_paths = ["norm.weight"]
        converted_weights = [weights]
    elif path == _TRANSFORMER_NORM_MTP:
        converted_paths.append("final_norm.weight")
        converted_weights.append(weights)

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


def convert(
    checkpoint_path: str, config: Gemma4UnifiedConfig | Gemma4UnifiedAssistantConfig
) -> dict[str, torch.Tensor]:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    ckpt = _restore_checkpoint(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    is_drafter = _VARIANT.value in _ASSISTANT_VARIANTS
    if is_drafter:
        flags.FLAGS.text_only = True

    text_path_prefix = "model"
    if not _TEXT_ONLY.value:
        text_path_prefix += ".language_model"

    text_config = config.get_text_config()

    # Collect extra embedding tables and projections for EOA/EOI unification.
    # These are projected and inserted into the main embedding table after the
    # main conversion loop.
    audio_extra_embedding: np.ndarray | None = None
    audio_projection_w: np.ndarray | None = None
    mm_extra_embedding: np.ndarray | None = None
    mm_projection_w: np.ndarray | None = None

    def update_tree(path: str, weights: np.ndarray, target_dtype: torch.dtype) -> None:
        # Convert directly to float32 in a single step to avoid an extra intermediate copy.
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

    vision_dtype = getattr(torch, _VISION_DTYPE.value)
    audio_dtype = getattr(torch, _AUDIO_DTYPE.value)
    text_dtype = text_config.dtype

    dtype_map = {
        "vision": vision_dtype,
        "audio": audio_dtype,
        "text": text_dtype,
    }

    for path_tuple, value in tree.flatten_with_path(ckpt):
        param = path_tuple[-1]
        if "params" in path_tuple:
            path_tuple = path_tuple[2:]
        path_tuple = path_tuple[:-1]
        path = "/".join(path_tuple) if len(path_tuple) > 1 else path_tuple[0]

        # Intercept extra embedding tables before the embedder converter.
        if path == _TRANSFORMER_EMBEDDER and param == "audio_input_embedding_extra":
            audio_extra_embedding = np.asarray(value, dtype=np.float32)
            logging.info("Collected audio_input_embedding_extra: shape=%s", audio_extra_embedding.shape)
            continue
        if path == _TRANSFORMER_EMBEDDER and param == "mm_input_embedding_extra":
            mm_extra_embedding = np.asarray(value, dtype=np.float32)
            logging.info("Collected mm_input_embedding_extra: shape=%s", mm_extra_embedding.shape)
            continue
        # Also capture raw (un-transposed) projection weights for EOA/EOI patching.
        if path.endswith("audio_input_projection") and param == "w":
            audio_projection_w = np.asarray(value, dtype=np.float32)
            logging.info("Collected audio_input_projection/w: shape=%s", audio_projection_w.shape)
        if path.endswith("mm_input_projection") and param == "w":
            mm_projection_w = np.asarray(value, dtype=np.float32)
            logging.info("Collected mm_input_projection/w: shape=%s", mm_projection_w.shape)

        # Handle embedder weights (vision/audio multimodal components)
        if path.startswith(_TRANSFORMER_EMBEDDER) and not _TEXT_ONLY.value:
            # Try the embedder converter first for multimodal-specific weights
            embedder_results = list(convert_embedder_weights(path, param, value))
            if embedder_results:
                for hf_path, weights, dtype_group in embedder_results:
                    update_tree(hf_path, weights, dtype_map[dtype_group])
                continue

        # Gemma4UnifiedAssistantForCausalLM weights
        if param == "centroids":
            update_tree("masked_embedding.centroids.weight", value, text_dtype)
        elif param == "token_ordering":
            update_tree("masked_embedding.token_ordering", value, torch.long)
        elif path == _TRANSFORMER_PRE_PROJ_MTP:
            update_tree("pre_projection.weight", value.transpose(), text_dtype)
        elif path == _TRANSFORMER_POST_PROJ_MTP:
            update_tree("post_projection.weight", value.transpose(), text_dtype)
        # Handle transformer weights (text model)
        elif path.startswith(_TRANSFORMER_PARAMETER):
            for hf_path, weights in convert_transformer_weights(text_config, path, param, value):
                update_tree(f"{text_path_prefix}.{hf_path}", weights, text_dtype)

    # Unify extra embeddings into the main embedding table.
    # The extra embeddings are scaled by sqrt(d) (matching the Gemma convention
    # for embedding tables), projected through their projection matrices, and
    # then divided by sqrt(hidden_size) to compensate for the
    # ScaledWordEmbedding which multiplies by sqrt(hidden_size) at forward time.
    #
    #   stored = (extra_raw * sqrt(d)) @ proj_jax / sqrt(hidden_size)
    #
    embed_key = f"{text_path_prefix}.embed_tokens.weight"
    text_dtype = getattr(torch, _TEXT_DTYPE.value)
    eoa_token_id = getattr(config, "eoa_token_index", None)
    eoi_token_id = getattr(config, "eoi_token_id", None)
    hidden_size = text_config.hidden_size

    if embed_key in hf_tree:
        embed_table = hf_tree[embed_key].float()

        if audio_extra_embedding is not None and audio_projection_w is not None and eoa_token_id is not None:
            eoa_raw = audio_extra_embedding[0]  # First row is the EOA embedding

            # Scale by sqrt(d), project, then divide by sqrt(hidden_size).
            d = eoa_raw.shape[0]
            eoa_scaled = eoa_raw * np.sqrt(d).astype(eoa_raw.dtype)
            projected = np.dot(eoa_scaled.reshape(1, -1), audio_projection_w)  # (1, hidden_dim)
            projected = projected / np.sqrt(hidden_size).astype(projected.dtype)
            embed_table[eoa_token_id] = torch.from_numpy(projected[0])
            logging.info(
                "Unified EOA embedding: sqrt(%d)-scaled + projected audio_input_embedding_extra[0] (%d,) "
                "through audio_input_projection (%s), divided by sqrt(%d), "
                "and inserted at embed_tokens[%d]",
                d,
                eoa_raw.shape[0],
                audio_projection_w.shape,
                hidden_size,
                eoa_token_id,
            )

        if mm_extra_embedding is not None and mm_projection_w is not None and eoi_token_id is not None:
            eoi_raw = mm_extra_embedding[0]  # First row is the EOI embedding

            # Scale by sqrt(d), project, then divide by sqrt(hidden_size).
            d = eoi_raw.shape[0]
            eoi_scaled = eoi_raw * np.sqrt(d).astype(eoi_raw.dtype)
            projected = np.dot(eoi_scaled.reshape(1, -1), mm_projection_w)  # (1, hidden_dim)
            projected = projected / np.sqrt(hidden_size).astype(projected.dtype)
            embed_table[eoi_token_id] = torch.from_numpy(projected[0])
            logging.info(
                "Unified EOI embedding: sqrt(%d)-scaled + projected mm_input_embedding_extra[0] (%d,) "
                "through mm_input_projection (%s), divided by sqrt(%d), "
                "and inserted at embed_tokens[%d]",
                d,
                eoi_raw.shape[0],
                mm_projection_w.shape,
                hidden_size,
                eoi_token_id,
            )

        hf_tree[embed_key] = embed_table.to(text_dtype)

    # Keep input_embeddings and lm_head tied — both point to the same
    # (now-patched) embedding table.  Suppress EOA/EOI generation via
    # suppress_tokens in GenerationConfig instead of untying.
    hf_tree["lm_head.weight"] = hf_tree[f"{text_path_prefix}.embed_tokens.weight"]

    return hf_tree


def main(*args):
    del args

    output_path = _OUTPUT_PATH.value
    variant = _VARIANT.value
    is_drafter = variant in _ASSISTANT_VARIANTS

    config = _VARIANTS[variant]
    config.get_text_config().dtype = getattr(torch, _TEXT_DTYPE.value)

    if not is_drafter:
        if config.vision_config is not None:
            config.vision_config.dtype = getattr(torch, _VISION_DTYPE.value)
        if (audio_config := config.audio_config) is not None:
            audio_config.dtype = getattr(torch, _AUDIO_DTYPE.value)

    if _INCLUDE_CHAT_TEMPLATE.value:
        # Chat template is included for instruction tuned models, which treat
        # both "<eos>" and "<end_of_turn>" as generation stoppers.
        config.eos_token_id = [1, 106]

    logging.info(
        "Converting Gemma 4 Unified (%s) @ %s (language) and %s (vision)",
        variant,
        _TEXT_DTYPE.value,
        _VISION_DTYPE.value,
    )
    state_tree = convert(_CHECKPOINT_PATH.value, config)
    logging.info("Converted Gemma 4 Unified (%s) state tree from Orbax to Hugging Face.", variant)

    with accelerate.init_empty_weights():
        if is_drafter:
            model = Gemma4UnifiedAssistantForCausalLM(config=config)
        elif _TEXT_ONLY.value:
            config = config.text_config
            model = Gemma4UnifiedForCausalLM(config=config)
        else:
            model = Gemma4UnifiedForConditionalGeneration(config=config)

    model.load_state_dict(state_tree, assign=True, strict=False)
    logging.info(
        "Loaded Gemma 4 Unified (%s) in Hugging Face Transformers as a %s instance.",
        variant,
        type(model).__name__,
    )
    model.save_pretrained(output_path, state_dict=state_tree, safe_serialization=True)
    logging.info(
        "Saved Gemma 4 Unified (%s) to SafeTensors in %s using %s",
        variant,
        output_path,
        type(model).__name__,
    )
    del model
    del state_tree

    chat_template_kwargs = {"chat_template": _CHAT_TEMPLATE} if _INCLUDE_CHAT_TEMPLATE.value else {}
    response_schema_kwargs = {"response_schema": _RESPONSE_SCHEMA} if _INCLUDE_RESPONSE_SCHEMA.value else {}

    # Add <bos> for PT models.
    add_bos_token = not _INCLUDE_CHAT_TEMPLATE.value

    sentencepiece_extractor = SentencePieceExtractor(_TOKENIZER_PATH.value)
    vocab, _, merges = sentencepiece_extractor.extract()
    tokenizer = GemmaTokenizer(
        vocab=vocab,
        merges=merges,
        add_bos_token=add_bos_token,
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
    config.image_token_id = tokenizer.image_token_id
    config.boi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
    config.eoi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoi_token)
    config.audio_token_id = tokenizer.audio_token_id
    config.boa_token_id = tokenizer.convert_tokens_to_ids(tokenizer.boa_token)
    config.eoa_token_index = tokenizer.convert_tokens_to_ids(tokenizer.eoa_token)
    logging.info(
        "Set multimodal token IDs from tokenizer: image=%d, boi=%d, eoi=%d, audio=%d, boa=%d, eoa=%d",
        config.image_token_id,
        config.boi_token_id,
        config.eoi_token_id,
        config.audio_token_id,
        config.boa_token_id,
        config.eoa_token_index,
    )
    # Re-save the config with correct token IDs
    config.save_pretrained(output_path)

    if _TEXT_ONLY.value:
        tokenizer.save_pretrained(output_path)
        logging.info("Saved GemmaTokenizer for %s to %s", variant, output_path)
    else:
        vision_config = config.vision_config
        feature_extractor = Gemma4UnifiedAudioFeatureExtractor()
        image_processor = Gemma4UnifiedImageProcessor(
            max_soft_tokens=vision_config.num_soft_tokens,
            do_normalize=False,
            pooling_kernel_size=vision_config.pooling_kernel_size,
        )
        video_processor = Gemma4UnifiedVideoProcessor()
        processor = Gemma4UnifiedProcessor(
            image_processor=image_processor,
            feature_extractor=feature_extractor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            image_seq_length=vision_config.num_soft_tokens,
            **chat_template_kwargs,
        )
        processor.save_pretrained(output_path)

        logging.info("Saved Gemma4UnifiedProcessor for %s to %s", variant, output_path)
        del feature_extractor, image_processor, processor

    # Build suppress_tokens list from EOA and EOI token IDs to prevent
    # the model from generating these control tokens during decoding.
    suppress_tokens = []
    if config.eoa_token_index is not None:
        suppress_tokens.append(config.eoa_token_index)
    if config.eoi_token_id is not None:
        suppress_tokens.append(config.eoi_token_id)

    generation_config = GenerationConfig(
        pad_token_id=config.get_text_config().pad_token_id,
        bos_token_id=config.get_text_config().bos_token_id,
        eos_token_id=(
            tokenizer.convert_tokens_to_ids([tokenizer.eos_token, tokenizer.eot_token, tokenizer.str_token])
            if _INCLUDE_CHAT_TEMPLATE.value
            else config.get_text_config().eos_token_id
        ),
        suppress_tokens=suppress_tokens if suppress_tokens else None,
        temperature=1.0,
        do_sample=True,
        top_k=64,
        top_p=0.95,
    )
    generation_config.save_pretrained(output_path)


if __name__ == "__main__":
    app.run(main)
