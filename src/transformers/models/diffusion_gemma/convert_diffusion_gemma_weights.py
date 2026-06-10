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

r"""Utility to convert DiffusionGemma models from Orbax to HF Transformers checkpoint.

python src/transformers/models/diffusion_gemma/convert_diffusion_gemma_weights.py \
    --include_chat_template \
    --checkpoint_path="$HOME/diffusion_gemma/checkpoints/diffusion_gemma_orbax" \
    --output_path="$HOME/diffusion_gemma/checkpoints/diffusion_gemma_safetensors"
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

# version used: orbax-checkpoint==0.11.33 (0.11.34-0.11.36 fail)
from orbax import checkpoint as obc
from orbax.checkpoint import args as obc_args
from orbax.checkpoint import type_handlers

from transformers import (
    AutoProcessor,
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaTextConfig,
    EntropyBoundSamplerConfig,
    Gemma4VisionConfig,
    RopeParameters,
)


_DTYPES = {"float32", "bfloat16", "float16"}

_TRANSFORMER_PARAMETER = "transformer"
_TRANSFORMER_EMBEDDER = f"{_TRANSFORMER_PARAMETER}/embedder"
_TRANSFORMER_FINAL_NORM = f"{_TRANSFORMER_PARAMETER}/final_norm"

_VISION_ENCODER_PARAMETER = f"{_TRANSFORMER_PARAMETER}/vision_encoder"
_VISION_ENCODER_ENTRY = f"{_VISION_ENCODER_PARAMETER}/entry"
_VISION_ENCODER_STANDARDIZE = f"{_VISION_ENCODER_PARAMETER}/standardize"
_VISION_ENCODER_TRANSFORMER = f"{_VISION_ENCODER_PARAMETER}/transformer/stacked_layers/block"


_VISION_CONFIG = Gemma4VisionConfig(
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

_MODEL_CONFIG = DiffusionGemmaConfig(
    canvas_length=256,
    text_config=DiffusionGemmaTextConfig(
        hidden_size=2816,
        intermediate_size=2112,  # Shared expert FFW
        num_hidden_layers=30,
        layer_types=_DEFAULT_LAYER_TYPES * 5,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        use_bidirectional_attention="vision",
        num_experts=128,
        moe_intermediate_size=704,
        top_k_experts=8,
        sliding_window=1024,
        final_logit_softcapping=30.0,
        rope_parameters=_ROPE_PARAMS,
        max_position_embeddings=262_144,
    ),
    vision_config=_VISION_CONFIG,
    vision_soft_tokens_per_image=280,
)

_CHECKPOINT_PATH = flags.DEFINE_string(
    name="checkpoint_path",
    default=None,
    help="Path to the Orbax checkpoint.",
    required=True,
)

_INCLUDE_CHAT_TEMPLATE = flags.DEFINE_bool(
    name="include_chat_template",
    default=False,
    help="If true, will save the default chat template with the tokenizer",
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
            elif path.endswith("kv_einsum/ClippedEinsum_0"):
                # NOTE: In JAX reference implementations of Gemma, k_proj and v_proj are performed with a single einsum
                # operation. We split this into two operations in Transformers, but they are passed the same input and
                # share the same activation bounds for clipping, thus we re-use the same matrix for both.
                converted_paths.append(f"{base_path}.self_attn.k_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
                converted_paths.append(f"{base_path}.self_attn.v_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
            elif path.endswith("q_einsum/ClippedEinsum_0"):
                converted_paths.append(f"{base_path}.self_attn.q_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
            elif path.endswith("gating_einsum/ClippedEinsum_0"):
                # NOTE: In JAX reference implementations of Gemma, gate_proj and up_proj are performed with a single
                # einsum operation. We split this into two operations in Transformers, but they are passed the same
                # input and share the same activation bounds for clipping, thus we re-use the same matrix for both.
                converted_paths.append(f"{base_path}.mlp.gate_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
                converted_paths.append(f"{base_path}.mlp.up_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)
            elif path.endswith("linear/ClippedEinsum_0"):
                converted_paths.append(f"{base_path}.mlp.down_proj.{param.removeprefix('clip_')}")
                converted_weights.append(matrix)

            # Handle clipped einsum states (`compression_einsum` target paths).
            # The target path specifies the activation direction (`input` or `output`),
            # and the parameter holds `clip_min` or `clip_max`.
            elif "/compression_einsum/" in path:
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
                else:
                    logging.warning(f"Possibly unused path in vision encoder: {path}. (param: {param})")

            elif path.endswith("attn/attn_vec_einsum"):
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
            else:
                logging.warning(f"Possibly unused path in vision encoder: {path}. (param: {param})")
    else:
        logging.warning(f"Possibly unused path in vision encoder: {path}. (param: {param})")

    if (cpl := len(converted_paths)) != (cwl := len(converted_weights)):
        raise ValueError(
            "The `converted_paths` and `converted_weights` should be the same "
            f"length. Got {cpl} and {cwl}, respectively, for {path}."
        )

    return zip(converted_paths, converted_weights)


def convert_self_conditioner_weights(
    config: DiffusionGemmaConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    converted_paths: list[str] = []
    converted_weights: list[Any] = []
    matrix = weights

    if path.endswith("gating_einsum"):
        # Dense MLP: matrix shape [2, intermediate_size, hidden_size]
        converted_paths.extend(
            ["model.decoder.self_conditioning.gate_proj.weight", "model.decoder.self_conditioning.up_proj.weight"]
        )
        gate_proj_weight, up_proj_weight = matrix
        converted_weights.extend([gate_proj_weight, up_proj_weight])
    elif path.endswith("linear"):
        converted_paths.append("model.decoder.self_conditioning.down_proj.weight")
        converted_weights.append(matrix.transpose())
    elif path.endswith("pre_norm"):
        converted_paths.append("model.decoder.self_conditioning.pre_norm.weight")
        converted_weights.append(matrix)
    else:
        logging.warning(f"Possibly unused path in self-conditioning: {path}. (param: {param})")

    if (cpl := len(converted_paths)) != (cwl := len(converted_weights)):
        raise ValueError(
            "The `converted_paths` and `converted_weights` should be the same "
            f"length. Got {cpl} and {cwl}, respectively, for {path}."
        )

    return zip(converted_paths, converted_weights)


def convert_transformer_weights(
    config: DiffusionGemmaTextConfig,
    path: str,
    param: str,
    weights: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    converted_paths: list[str] = []
    converted_weights: list[Any] = []

    # Handle new checkpoint format: transformer/layer_N/...
    # TODO(philculliton):Direct handling for unstacked checkpoint type, needs to be merged to allow for unified tensor handling
    if path.startswith(f"{_TRANSFORMER_PARAMETER}/layer_"):
        # Extract layer number from path like "transformer/layer_0/attn/q_einsum"
        layer_str = path.split("/")[1]  # "layer_0"
        layer_idx = int(layer_str.replace("layer_", ""))  # 0
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
            # NOTE: The JAX implementations changes the type of the primary `mlp` for MOE models and adds a new
            # `mlp2` that operates _before_ `mlp`. In Hugging Face Transformers we keep the type of `mlp` constant
            # and add an `experts` that operates after `mlp`, so we need to invert this assignment when using MOE arch.

            # MoE expert weights: matrix shape [num_experts, 2, moe_intermediate_size, hidden_size]
            # -> experts.gate_up_proj (nn.Parameter, shape [E, 2*moe_inter, hidden])
            num_experts, _, expert_inter, hidden_size = matrix.shape
            gate_up_proj_weight = matrix.reshape(num_experts, 2 * expert_inter, hidden_size)
            converted_paths.append(f"{base_path}.experts.gate_up_proj")
            converted_weights.append(gate_up_proj_weight)
        elif path.endswith("mlp/linear"):
            # NOTE: The JAX implementations changes the type of the primary `mlp` for MOE models and adds a new
            # `mlp2` that operates _before_ `mlp`. In Hugging Face Transformers we keep the type of `mlp` constant
            # and add an `experts` that operates after `mlp`, so we need to invert this assignment when using MOE arch.

            # MoE expert down_proj: matrix shape [num_experts, moe_inter, hidden]
            # -> experts.down_proj (nn.Parameter, shape [E, hidden, moe_inter])
            converted_paths.append(f"{base_path}.experts.down_proj")
            converted_weights.append(matrix.transpose(0, 2, 1))
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
            # NOTE: The JAX implementations changes the type of the primary `mlp` for MOE models and adds a new
            # `mlp2` that operates _before_ `mlp`. In Hugging Face Transformer we keep the type of `mlp` constant
            # and add an `mlp2` that operates after `mlp`, so we need to invert this assignment when using MOE arch.

            # pre_ffw_norm is the pre-norm for ffw1 (MoE); in HF, MoE is mlp_2
            converted_paths.append(f"{base_path}.pre_feedforward_layernorm_2.weight")
            converted_weights.append(matrix)
        elif path.endswith(layer_str) and param == "skip_scale":
            converted_paths.append(f"{base_path}.layer_scalar")
            converted_weights.append(matrix)
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
        elif path.endswith("mlp/router_logits"):
            # MoE router: matrix shape [hidden_size, num_experts]
            # -> router.proj.weight (nn.Linear, shape [num_experts, hidden_size])
            converted_paths.append(f"{base_path}.router.proj.weight")
            converted_weights.append(matrix.transpose())
        elif path.endswith("post_ffw1_norm"):
            converted_paths.append(f"{base_path}.post_feedforward_layernorm_2.weight")
            converted_weights.append(matrix)
        elif path.endswith("post_ffw2_norm"):
            converted_paths.append(f"{base_path}.post_feedforward_layernorm_1.weight")
            converted_weights.append(matrix)
        elif path.endswith("pre_ffw2_norm"):
            converted_paths.append(f"{base_path}.pre_feedforward_layernorm.weight")
            converted_weights.append(matrix)
        else:
            logging.warning(f"Possibly unused path in transformers layer: {path}. (param: {param})")

    elif path == _TRANSFORMER_EMBEDDER:
        if param == "input_embedding":
            converted_paths.append("embed_tokens.weight")
            converted_weights.append(weights)
        else:
            logging.warning(f"Possibly unused path in embedder: {path}. (param: {param})")
    elif path == _TRANSFORMER_FINAL_NORM:
        converted_paths = ["norm.weight"]
        converted_weights = [weights]
    else:
        logging.warning(f"Possibly unused path in transformers: {path}. (param: {param})")

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


def convert(checkpoint_path: str, config: DiffusionGemmaConfig) -> dict[str, torch.Tensor]:
    """Loads Orbax checkpoint from `input_path` and converts it to HF tree."""
    ckpt = _restore_checkpoint(checkpoint_path)

    new_tree = {}
    new_tree["transformer"] = ckpt
    ckpt = new_tree

    hf_tree: dict[str, torch.Tensor] = {}

    text_path_prefix = "model.encoder.language_model"

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

        if config.vision_config is not None and path.endswith("mm_input_projection"):
            update_tree(
                "model.encoder.embed_vision.embedding_projection.weight", value.transpose(), config.vision_config.dtype
            )
        elif "self_conditioner" in path:
            for hf_path, weights in convert_self_conditioner_weights(config, path, param, value):
                update_tree(hf_path, weights, config.text_config.dtype)
        elif config.vision_config is not None and path.startswith(_VISION_ENCODER_PARAMETER):
            for hf_path, weights in convert_vision_encoder_weights(config.vision_config, path, param, value):
                update_tree(f"model.encoder.vision_tower.{hf_path}", weights, config.vision_config.dtype)
        elif path.startswith(_TRANSFORMER_PARAMETER):
            for hf_path, weights in convert_transformer_weights(config.text_config, path, param, value):
                update_tree(f"{text_path_prefix}.{hf_path}", weights, config.text_config.dtype)
        else:
            logging.warning(f"Possibly unused path in Diffusion Gemma 4: {path}. (param: {param})")

    # tied weights -> make a pointer copy (buffers within these blocks need a deep copy)
    decoder_dict = {}
    for key, value in hf_tree.items():
        if key.startswith(text_path_prefix):
            decoder_path = key.replace(text_path_prefix, "model.decoder")
            if "layer_scalar" in decoder_path:  # buffers are not tied, need a separate copy
                decoder_dict[decoder_path] = value.clone()
            else:
                decoder_dict[decoder_path] = value
    hf_tree.update(decoder_dict)
    hf_tree["lm_head.weight"] = hf_tree[f"{text_path_prefix}.embed_tokens.weight"]

    return hf_tree


def main(*args):
    del args

    output_path = _OUTPUT_PATH.value

    config = _MODEL_CONFIG
    config.text_config.dtype = getattr(torch, _TEXT_DTYPE.value)
    if config.vision_config is not None:
        config.vision_config.dtype = getattr(torch, _VISION_DTYPE.value)

    if _INCLUDE_CHAT_TEMPLATE.value:
        # Chat template is included for instruction tuned models, which treat
        # both "<eos>" and "<end_of_turn>" as generation stoppers.
        config.eos_token_id = [1, 106]

    logging.info(
        "Converting Diffusion Gemma 4 @ %s (language) and %s (vision)",
        _TEXT_DTYPE.value,
        _VISION_DTYPE.value,
    )
    state_tree = convert(_CHECKPOINT_PATH.value, config)
    logging.info("Converted Diffusion Gemma 4 state tree from Orbax to Hugging Face.")

    with accelerate.init_empty_weights():
        model = DiffusionGemmaForBlockDiffusion(config=config)

    model.load_state_dict(state_tree, assign=True, strict=True)
    logging.info(
        "Loaded Diffusion Gemma 4 in Hugging Face Transformers as a %s instance.",
        type(model).__name__,
    )
    model.save_pretrained(output_path, max_shard_size="5GB")
    logging.info(
        "Saved Diffusion Gemma 4 to SafeTensors in %s using %s",
        output_path,
        type(model).__name__,
    )
    del model
    del state_tree

    # We take a copy of the Gemma4 MoE processor -- it's the same as in DiffusionGemma
    processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
    processor.save_pretrained(output_path)
    logging.info("Saved Gemma4Processor to %s", output_path)

    generation_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=256,
        max_denoising_steps=48,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        t_min=0.4,
        t_max=0.8,
        stability_threshold=1,
        confidence_threshold=0.005,
        eos_token_id=[1, 106, 50],
        pad_token_id=0,
    )
    generation_config.save_pretrained(output_path)


if __name__ == "__main__":
    app.run(main)
