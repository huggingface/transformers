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

r"""Convert an EmbeddingGemma 2 Orbax checkpoint to a HF Transformers checkpoint.

The output directory is a SentenceTransformers model: alongside the usual
`config.json` / `model.safetensors` / tokenizer and processor files it contains
`modules.json` and `config_sentence_transformers.json`, so it can be loaded with
either `EmbeddingGemma2Model.from_pretrained` or `SentenceTransformer`.

python src/transformers/models/embedding_gemma2/convert_embedding_gemma2_weights.py \
    --tokenizer_path="/path/to/tokenizer.model" \
    --checkpoint_path="/path/to/orbax_checkpoint" \
    --output_path="/path/to/output_dir"
"""

import ast
import json
import os
from collections.abc import Iterable
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
    EmbeddingGemma2Config,
    EmbeddingGemma2Model,
    EmbeddingGemma2Processor,
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2VideoProcessor,
    Gemma4AudioConfig,
    Gemma4AudioFeatureExtractor,
    Gemma4ImageProcessor,
    Gemma4VisionConfig,
    GemmaTokenizer,
)
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor


# ==== Internal Constants and Classes ====

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

_VISION_ENCODER_PARAMETER = "PatchInputVariablePoolingEncoder_0"
_VISION_ENCODER_VIT_PARAMETER = f"{_VISION_ENCODER_PARAMETER}/_model/vit"
_VISION_ENCODER_ENTRY = f"{_VISION_ENCODER_VIT_PARAMETER}/entry"
_VISION_ENCODER_TRANSFORMER = f"{_VISION_ENCODER_VIT_PARAMETER}/transformer/stacked_layers/block"

# The embedding head, stored alongside the rest of the transformer tree.
_EMBEDDING_PROJECTION = "transformer/default_projection/linear"

# Boundary-token embeddings live outside the main table in the Orbax checkpoint and are fused into
# it by `_fuse_boundary_token_embeddings`.
_AUDIO_INPUT_EMBEDDING_EXTRA = "audio_input_embedding_extra"
_MM_INPUT_EMBEDDING_EXTRA = "mm_input_embedding_extra"

# An embedding-only chat template: it emits no turn markers or role prefixes. System messages
# (e.g. Sentence Transformers task prompts) are emitted first right after `<bos>`, matching the
# training distribution. Non-system content entries are rendered in the order the caller supplied
# them (Sentence Transformers preserves the input dict's key order since v6.1.0). Each non-text
# part becomes a bare `<|image|>` / `<|video|>` / `<|audio|>` placeholder unless the message's text
# already specifies manual placeholders, in which case automatic insertion is disabled for that
# message and the processor validates that placeholder counts match the passed multimodal inputs.
_EMBEDDING_CHAT_TEMPLATE = (
    "{%- for msg in messages if msg.get('role') == 'system' -%}"
    "{%- if msg.get('content') is string -%}"
    "{{ msg['content'] }}"
    "{%- else -%}"
    "{%- for item in msg['content'] if item.get('type') == 'text' -%}"
    "{{ item['text'] }}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- for msg in messages if msg.get('role') != 'system' -%}"
    "{%- if msg.get('content') is string -%}"
    "{{ msg['content'] }}"
    "{%- else -%}"
    "{%- set existing_text = msg['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | join -%}"
    "{%- set has_manual_placeholders = ('<|image|>' in existing_text) or ('<|video|>' in existing_text) or ('<|audio|>' in existing_text) -%}"
    "{%- for item in msg['content'] -%}"
    "{%- if item.get('type') == 'text' -%}"
    "{{ item['text'] }}"
    "{%- elif not has_manual_placeholders and item.get('type') == 'image' -%}"
    "<|image|>"
    "{%- elif not has_manual_placeholders and item.get('type') == 'video' -%}"
    "<|video|>"
    "{%- elif not has_manual_placeholders and item.get('type') == 'audio' -%}"
    "<|audio|>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
)

_EXTRA_SPECIAL_TOKENS = {
    "image_token": "<|image|>",
    "video_token": "<|video|>",
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
}

# Prompt prefixes exposed through `SentenceTransformer(prompts=...)`. These are the MTEB-facing
# surface of the model; the keys are task names understood by the MTEB harness.
_TASK_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "BitextMining": "task: search result | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Clustering": "task: clustering | query: ",
    "Classification": "task: classification | query: ",
    "Document": "title: none | text: ",
    "FactChecking": "task: fact checking | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "SearchQuery": "task: search result | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
}

# Declared in `config_sentence_transformers.json` and verified by SentenceTransformer before any module
# is loaded (sentence-transformers >= 6.0.0). SentenceTransformers >= 6.1.0 is required for content-order
# preservation and multiple items per modality in dict inputs.
_ST_REQUIREMENTS = {
    "sentence-transformers": {
        "specifier": ">=6.1.0",
        "reason": (
            "Older versions ignore the order of multimodal dict inputs and cannot put several images "
            "or texts into a single embedding."
        ),
    },
}

_VISION_CONFIG = Gemma4VisionConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=16,
    num_attention_heads=12,
    num_key_value_heads=12,
    head_dim=64,
    global_head_dim=64,
    default_output_length=280,
    pooling_kernel_size=3,
    position_embedding_size=10240,
    use_clipped_linears=False,
)

_AUDIO_CONFIG = Gemma4AudioConfig()

_NUM_HIDDEN_LAYERS = 24

_CONFIG = EmbeddingGemma2Config(
    text_config=EmbeddingGemma2TextConfig(
        vocab_size=262_144,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=_NUM_HIDDEN_LAYERS,
        num_attention_heads=4,
        # Sliding layers: 2 KV heads at head_dim 256; global layers: 1 KV head at head_dim 512.
        # The config expands the `global_*` values into `per_layer_config` overrides.
        num_key_value_heads=2,
        head_dim=256,
        num_global_key_value_heads=1,
        global_head_dim=512,
        max_position_embeddings=262_144,
        # Inclusive radius of the symmetric window; the JAX config states it as a 1024-wide window.
        sliding_window=512,
        hidden_size_per_layer_input=512,
        rope_parameters=None,
        embedding_dim=768,
    ),
    vision_config=_VISION_CONFIG,
    audio_config=_AUDIO_CONFIG,
    vision_soft_tokens_per_image=280,
    boi_token_id=255999,
    eoi_token_id=258882,
    image_token_id=258880,
    video_token_id=258884,
    boa_token_id=256000,
    eoa_token_index=258883,
    audio_token_id=258881,
)

_IMAGE_SEQ_LENGTH = 280
_AUDIO_SEQ_LENGTH = 280
_VIDEO_MAX_SOFT_TOKENS = 140
# Frame sampling defaults: one frame per second, capped at 32 frames by uniform (linspace) re-sampling.
_VIDEO_FPS = 1
_VIDEO_MAX_FRAMES = 32
_VIDEO_OVERFLOW_STRATEGY = "uniform"


# ==== Flags ====

_AUDIO_DTYPE = flags.DEFINE_enum(
    name="audio_dtype",
    default="bfloat16",
    help="The floating point precision (aka dtype) of the audio tower.",
    enum_values=_DTYPES,
)

_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path",
    default=None,
    help="Path to the Orbax checkpoint.",
    required=True,
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
    help="The floating point precision (aka dtype) of the text backbone.",
    enum_values=_DTYPES,
)

_TOKENIZER_PATH = flags.DEFINE_string(
    name="tokenizer_path",
    default=None,
    help="Path to the SentencePiece model file.",
    required=True,
)

_VERBOSE = flags.DEFINE_bool(
    name="verbose",
    default=False,
    help="If true, log the path, shape and dtype of every converted layer.",
)

_VISION_DTYPE = flags.DEFINE_enum(
    name="vision_dtype",
    default="bfloat16",
    help="The floating point precision (aka dtype) of the vision tower.",
    enum_values=_DTYPES,
)


# ==== Weight conversion ====


def convert_audio_encoder_weights(
    config: Gemma4AudioConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    converted_paths: list[str] = []
    converted_weights: list[Any] = []

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
    config: Gemma4VisionConfig,
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

    # Patch Embedder - Entry. Both keys are consumed by `Gemma4VisionPatchEmbedder`.
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

    # Transformer Layers (stacked format)
    elif path.startswith(_VISION_ENCODER_TRANSFORMER):
        # All vision transformer layers are stacked in dimension 0
        num_layers = weights.shape[0]
        assert num_layers == config.num_hidden_layers, f"Expected {config.num_hidden_layers} layers, got {num_layers}"

        for i, matrix in enumerate(weights):
            base_path = f"encoder.layers.{i}"

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
    config: EmbeddingGemma2TextConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    # Text transformer layers are stacked by attention type: transformer/stacked_layers/attention_type_N/...
    if path.startswith(_TRANSFORMER_DECODER_BLOCK):
        attention_type_index = int(path[_TRANSFORMER_DECODER_BLOCK_LEN])
        expected_layers_per_group = config.num_hidden_layers / _SLIDING_WINDOW_PATTERN
        observed_layers_per_group = weights.shape[0]
        assert observed_layers_per_group == expected_layers_per_group, (
            f"Expected {observed_layers_per_group=} to be {expected_layers_per_group=}"
        )

        for i, matrix in enumerate(weights):
            layer_idx = _SLIDING_WINDOW_PATTERN * i + attention_type_index
            base_path = f"layers.{layer_idx}"
            head_dim = config.per_layer_config[layer_idx].head_dim
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
                num_kv_heads = config.per_layer_config[layer_idx].num_key_value_heads
                kv_proj_shape = (config.hidden_size, num_kv_heads * head_dim)
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
                    .reshape(config.hidden_size, config.num_attention_heads * head_dim)
                    .transpose()
                )
            elif path.endswith("attn/query_norm"):
                converted_paths.append(f"{base_path}.self_attn.q_norm.weight")
                converted_weights.append(matrix.squeeze())
            elif path.endswith("attn/key_norm"):
                converted_paths.append(f"{base_path}.self_attn.k_norm.weight")
                converted_weights.append(matrix.squeeze())
            elif path.endswith("mlp/gating_einsum"):
                # Dense MLP: matrix shape [2, intermediate_size, hidden_size]
                gate_proj_weight, up_proj_weight = matrix
                converted_paths.extend([f"{base_path}.mlp.gate_proj.weight", f"{base_path}.mlp.up_proj.weight"])
                converted_weights.extend([gate_proj_weight, up_proj_weight])
            elif path.endswith("mlp/linear"):
                converted_paths.append(f"{base_path}.mlp.down_proj.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("per_layer_input_gate"):
                converted_paths.append(f"{base_path}.ple_block.per_layer_input_gate.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("per_layer_projection"):
                converted_paths.append(f"{base_path}.ple_block.per_layer_projection.weight")
                converted_weights.append(matrix.transpose())
            elif path.endswith("post_attention_norm"):
                converted_paths.append(f"{base_path}.post_attention_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_ffw_norm"):
                converted_paths.append(f"{base_path}.post_feedforward_layernorm.weight")
                converted_weights.append(matrix)
            elif path.endswith("post_per_layer_input_norm"):
                converted_paths.append(f"{base_path}.ple_block.post_per_layer_input_norm.weight")
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
    elif path.startswith(_TRANSFORMER_EMBEDDER):
        if path.endswith("per_layer_model_projection"):
            converted_paths.append("ple.per_layer_model_projection.weight")
            converted_weights.append(
                weights.reshape(
                    config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input
                ).transpose()
            )
        elif path.endswith("per_layer_projection_norm"):
            converted_paths.append("ple.per_layer_projection_norm.weight")
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


def _round_to_dtype(value: float, dtype: torch.dtype) -> float:
    """Round a scalar to `dtype`'s precision, returned as a Python float."""
    return torch.tensor(value, dtype=torch.float64).to(dtype).item()


def _fuse_boundary_token_embeddings(
    hf_tree: dict[str, torch.Tensor],
    config: EmbeddingGemma2Config,
    extras: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
) -> None:
    """Fuse the end-of-audio and end-of-image embeddings into the main embedding table.

    The Orbax checkpoint keeps these two embeddings *outside* `embed_tokens`, in
    `audio_input_embedding_extra` and `mm_input_embedding_extra` under the transformer embedder,
    expressed in their own modality's space. Row 0 of each is the boundary token; it is projected
    into text space with that modality's input projection and written into the table:

        embed_tokens[t] = (extra[0] * sqrt(d)) @ W_proj / sqrt(hidden_size)

    The `sqrt(d)` factor matches the Gemma embedding-table convention, and the
    `1 / sqrt(hidden_size)` divisor pre-compensates for `EmbeddingGemma2TextScaledWordEmbedding`
    multiplying by `sqrt(hidden_size)` at forward time. Both constants are rounded to the
    precision they are applied in at runtime -- see the comment on the multiplication below.

    Only row 0 of each extra table is used -- the begin-of-audio and begin-of-image tokens are
    *not* fused, because they are plain learned tokens already present in `embed_tokens`.
    """
    embed_key = "language_model.embed_tokens.weight"
    if embed_key not in hf_tree:
        raise ValueError(
            f"Cannot fuse boundary-token embeddings: {embed_key!r} is missing from the converted "
            f"tree. Got {len(hf_tree)} tensors; is the checkpoint layout as expected?"
        )

    hidden_size = config.text_config.hidden_size
    embed_table = hf_tree[embed_key].float()

    targets = (
        ("EOA", _AUDIO_INPUT_EMBEDDING_EXTRA, "audio_input_projection", config.eoa_token_index),
        ("EOI", _MM_INPUT_EMBEDDING_EXTRA, "mm_input_projection", config.eoi_token_id),
    )

    for label, extra_key, projection_key, token_id in targets:
        extra = extras.get(extra_key)
        projection = projections.get(projection_key)
        # Hard failure: a renamed Orbax key would otherwise yield silently wrong embeddings.
        if extra is None:
            raise ValueError(f"{label}: {extra_key!r} not found in the Orbax checkpoint.")
        if projection is None:
            raise ValueError(f"{label}: {projection_key!r} not found in the Orbax checkpoint.")
        if token_id is None:
            raise ValueError(f"{label}: the target token id is None on the config.")

        raw = extra[0]
        modality_dim = raw.shape[0]
        # Both scales are rounded to `bfloat16` because that is the precision they are applied in
        # at runtime: the reference embedder scales the extra table in bfloat16 activations, and
        # `EmbeddingGemma2TextScaledWordEmbedding` casts `embed_scale` to the weight dtype before
        # multiplying. Using full-precision constants here instead leaves the two fused rows ~0.15%
        # short of the reference (sqrt(1536) is 39.25 in bfloat16, not 39.1918).
        scaled = raw * _round_to_dtype(np.sqrt(modality_dim), torch.bfloat16)
        projected = np.dot(scaled.reshape(1, -1), projection)
        projected = projected / _round_to_dtype(np.sqrt(hidden_size), config.text_config.dtype)
        embed_table[token_id] = torch.from_numpy(projected[0])

        logging.info(
            "Fused %s embedding into embed_tokens[%d]: %s[0] (%d,) scaled by sqrt(%d), projected "
            "through %s %s, divided by sqrt(%d). %d unused row(s) remain in %s.",
            label,
            token_id,
            extra_key,
            modality_dim,
            modality_dim,
            projection_key,
            projection.shape,
            hidden_size,
            extra.shape[0] - 1,
            extra_key,
        )

    hf_tree[embed_key] = embed_table.to(config.text_config.dtype)


def convert(checkpoint_path: str, config: EmbeddingGemma2Config) -> dict[str, torch.Tensor]:
    """Loads the Orbax checkpoint from `checkpoint_path` and converts it to an HF state tree."""
    ckpt = _restore_checkpoint(checkpoint_path)
    hf_tree: dict[str, torch.Tensor] = {}

    text_config = config.text_config

    # `EmbeddingGemma2Model` holds its submodules at the top level (no `model.` prefix).
    text_path_prefix = "language_model"

    def update_tree(path: str, weights: np.ndarray, target_dtype: torch.dtype) -> None:
        # Convert directly to float32 in a single step to avoid an extra intermediate copy.
        weights_f32 = np.asarray(weights, dtype=np.float32)
        del weights  # allow GC of the input (JAX array or numpy view)
        t = torch.from_numpy(weights_f32)  # shares memory with weights_f32
        if t.dtype != target_dtype:
            hf_tree[path] = t.to(target_dtype)
            del t, weights_f32  # free the float32 intermediate
        else:
            # Some source paths (e.g. the audio attention clip bounds) intentionally emit the same
            # numpy array for several targets; `safetensors` rejects the resulting shared storage.
            hf_tree[path] = t.clone()
        if _VERBOSE.value:
            logging.info("%s converted shape=%s with dtype=%s", path, hf_tree[path].shape, target_dtype)

    # Collected during the walk (order is arbitrary) and consumed by `_fuse_boundary_token_embeddings`.
    extras: dict[str, np.ndarray] = {}
    projections: dict[str, np.ndarray] = {}

    for path_tuple, value in tree.flatten_with_path(ckpt):
        param = path_tuple[-1]
        if "params" in path_tuple:
            path_tuple = path_tuple[2:]
        path_tuple = path_tuple[:-1]
        path = "/".join(path_tuple) if len(path_tuple) > 1 else path_tuple[0]

        if path == _EMBEDDING_PROJECTION:
            # The embedding head. JAX stores (in_features, out_features); torch wants the transpose.
            update_tree(f"{text_path_prefix}.embedding_projection.weight", value.transpose(), text_config.dtype)
        elif path == _TRANSFORMER_EMBEDDER and param in (_AUDIO_INPUT_EMBEDDING_EXTRA, _MM_INPUT_EMBEDDING_EXTRA):
            extras[param] = np.asarray(value, dtype=np.float32)
            logging.info("Collected %s: shape=%s", param, extras[param].shape)
        # EmbeddingGemma2MultimodalEmbedder weights
        elif path.endswith("audio_input_projection"):
            projections["audio_input_projection"] = np.asarray(value, dtype=np.float32)
            update_tree("embed_audio.embedding_projection.weight", value.transpose(), config.audio_config.dtype)
        elif path.endswith("mm_input_projection"):
            projections["mm_input_projection"] = np.asarray(value, dtype=np.float32)
            update_tree("embed_vision.embedding_projection.weight", value.transpose(), config.vision_config.dtype)
        # Subordinate model (language_model, vision_tower, audio_tower) weights
        elif path.startswith(_TRANSFORMER_PARAMETER):
            for hf_path, weights in convert_transformer_weights(text_config, path, param, value):
                update_tree(f"{text_path_prefix}.{hf_path}", weights, text_config.dtype)
        elif path.startswith(_VISION_ENCODER_PARAMETER):
            for hf_path, weights in convert_vision_encoder_weights(config.vision_config, path, param, value):
                update_tree(f"vision_tower.{hf_path}", weights, config.vision_config.dtype)
        elif path.startswith(_AUDIO_ENCODER_PARAMETER):
            for hf_path, weights in convert_audio_encoder_weights(config.audio_config, path, param, value):
                update_tree(f"audio_tower.{hf_path}", weights, config.audio_config.dtype)

    _fuse_boundary_token_embeddings(hf_tree, config, extras, projections)

    # No `lm_head.weight`: EmbeddingGemma 2 has no language modeling head.
    return hf_tree


def _build_tokenizer(tokenizer_path: str) -> GemmaTokenizer:
    vocab, _, merges = SentencePieceExtractor(tokenizer_path).extract()
    return GemmaTokenizer(
        vocab=vocab,
        merges=merges,
        add_bos_token=True,
        add_eos_token=True,
        padding_side="right",
        extra_special_tokens=_EXTRA_SPECIAL_TOKENS,
        chat_template=_EMBEDDING_CHAT_TEMPLATE,
    )


def _sync_token_ids(config: EmbeddingGemma2Config, tokenizer: GemmaTokenizer) -> None:
    """Take the multimodal token ids from the tokenizer, asserting they match the config.

    The boundary-token fusion writes into `embed_tokens` at `config.eoa_token_index` and
    `config.eoi_token_id`. If the hardcoded config ids and the tokenizer's ids ever diverge, the
    fusion would silently target the wrong rows, so the mismatch is raised rather than patched over.
    """
    from_tokenizer = {
        "image_token_id": tokenizer.image_token_id,
        "video_token_id": tokenizer.video_token_id,
        "audio_token_id": tokenizer.audio_token_id,
        "boi_token_id": tokenizer.convert_tokens_to_ids(tokenizer.boi_token),
        "eoi_token_id": tokenizer.convert_tokens_to_ids(tokenizer.eoi_token),
        "boa_token_id": tokenizer.convert_tokens_to_ids(tokenizer.boa_token),
        "eoa_token_index": tokenizer.convert_tokens_to_ids(tokenizer.eoa_token),
    }

    mismatches = {
        name: (getattr(config, name), token_id)
        for name, token_id in from_tokenizer.items()
        if getattr(config, name) != token_id
    }
    if mismatches:
        raise ValueError(
            "Tokenizer and config disagree on multimodal token ids "
            f"(config, tokenizer): {mismatches}. The boundary-token embedding fusion writes into "
            "`embed_tokens` by id, so this must be resolved rather than silently overridden."
        )

    for name, token_id in from_tokenizer.items():
        setattr(config, name, token_id)


def _build_processor(tokenizer: GemmaTokenizer) -> EmbeddingGemma2Processor:
    image_processor = Gemma4ImageProcessor(
        image_seq_length=_IMAGE_SEQ_LENGTH,
        do_normalize=False,
        max_soft_tokens=_IMAGE_SEQ_LENGTH,
        pooling_kernel_size=3,
    )
    video_processor = EmbeddingGemma2VideoProcessor(
        max_soft_tokens=_VIDEO_MAX_SOFT_TOKENS,
        do_normalize=False,
        # All four are already the EmbeddingGemma 2 class defaults; passed explicitly so the exported
        # config records the sampling contract rather than relying on the class definition.
        # `num_frames` is deliberately absent: `EmbeddingGemma2VideoProcessor.sample_frames` rejects it
        # (it would otherwise be forwarded by `preprocess` and raise on every call).
        fps=_VIDEO_FPS,
        max_frames=_VIDEO_MAX_FRAMES,
        overflow_strategy=_VIDEO_OVERFLOW_STRATEGY,
        add_timestamps=False,
    )
    return EmbeddingGemma2Processor(
        image_processor=image_processor,
        feature_extractor=Gemma4AudioFeatureExtractor(),
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=_EMBEDDING_CHAT_TEMPLATE,
        image_seq_length=_IMAGE_SEQ_LENGTH,
        audio_seq_length=_AUDIO_SEQ_LENGTH,
    )


def main(*args):
    del args

    output_path = _OUTPUT_PATH.value
    config = _CONFIG

    config.text_config.dtype = getattr(torch, _TEXT_DTYPE.value)
    config.vision_config.dtype = getattr(torch, _VISION_DTYPE.value)
    config.audio_config.dtype = getattr(torch, _AUDIO_DTYPE.value)

    logging.info(
        "Converting EmbeddingGemma 2 @ %s (text), %s (vision), %s (audio)",
        _TEXT_DTYPE.value,
        _VISION_DTYPE.value,
        _AUDIO_DTYPE.value,
    )

    # Built before the weight conversion: `_sync_token_ids` validates the ids that the
    # boundary-token fusion writes to.
    tokenizer = _build_tokenizer(_TOKENIZER_PATH.value)
    _sync_token_ids(config, tokenizer)

    state_tree = convert(_CHECKPOINT_PATH.value, config)

    with accelerate.init_empty_weights():
        model = EmbeddingGemma2Model(config=config)

    model.load_state_dict(state_tree, assign=True)
    logging.info("Loaded EmbeddingGemma 2 as a %s instance.", type(model).__name__)

    model.save_pretrained(output_path, safe_serialization=True)
    logging.info("Saved EmbeddingGemma 2 to SafeTensors in %s", output_path)
    del model
    del state_tree

    tokenizer.save_pretrained(output_path)
    processor = _build_processor(tokenizer)
    processor.save_pretrained(output_path)
    processor.feature_extractor.save_pretrained(output_path)
    logging.info("Saved tokenizer, %s and feature extractor to %s", type(processor).__name__, output_path)

    # `models.Transformer` reads the model, config and tokenizer back off disk, so everything above
    # has to be saved first.
    from sentence_transformers import SentenceTransformer, models

    st_model = SentenceTransformer(
        modules=[
            models.Transformer(output_path),
            models.Pooling(config.text_config.embedding_dim, pooling_mode="mean"),
            models.Normalize(),
        ],
        prompts=_TASK_PROMPTS,
    )
    st_model = st_model.to(getattr(torch, _TEXT_DTYPE.value))
    st_model.save_pretrained(output_path)
    logging.info("Saved SentenceTransformer to %s", output_path)
    del st_model

    # `requirements` is read but never written by SentenceTransformers, so it is patched in afterwards.
    st_config_path = os.path.join(output_path, "config_sentence_transformers.json")
    with open(st_config_path, encoding="utf-8") as f:
        st_config = json.load(f)
    st_config["requirements"] = _ST_REQUIREMENTS
    with open(st_config_path, "w", encoding="utf-8") as f:
        json.dump(st_config, f, indent=2, sort_keys=True)
        f.write("\n")
    logging.info("Declared SentenceTransformers requirements: %s", _ST_REQUIREMENTS)


if __name__ == "__main__":
    app.run(main)
