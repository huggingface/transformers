# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Convert original TinyModel checkpoints to the Transformers format."""

import re
from collections.abc import Mapping
from pathlib import Path

import torch

from .configuration_tiny_model import TinyModelConfig
from .modeling_tiny_model import TinyModelForCausalLM


def _convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    num_attention_heads: int = 16,
    expected_num_hidden_layers: int | None = None,
) -> tuple[TinyModelConfig, dict[str, torch.Tensor]]:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected a state-dict mapping, got {type(state_dict).__name__}.")

    non_string_keys = [repr(key) for key in state_dict if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"TinyModel checkpoint keys must be strings, got: {sorted(non_string_keys)}")

    non_tensor_keys = [key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        raise TypeError(f"TinyModel checkpoint values must be tensors, invalid keys: {sorted(non_tensor_keys)}")

    invalid_dtype_keys = [key for key, value in state_dict.items() if value.dtype != torch.bfloat16]
    if invalid_dtype_keys:
        raise ValueError(
            "TinyModel checkpoints must contain only bfloat16 tensors, invalid keys: "
            f"{sorted(invalid_dtype_keys)}"
        )

    layer_indices = {
        int(match.group(1))
        for key in state_dict
        if (match := re.fullmatch(r"torso\.(\d+)\..+", key)) is not None
    }
    if not layer_indices:
        raise ValueError("TinyModel checkpoint does not contain any decoder layers.")

    num_hidden_layers = max(layer_indices) + 1
    expected_layer_indices = set(range(num_hidden_layers))
    if layer_indices != expected_layer_indices:
        raise ValueError(
            f"TinyModel decoder layers must be contiguous from zero, got {sorted(layer_indices)}."
        )
    if expected_num_hidden_layers is not None and num_hidden_layers != expected_num_hidden_layers:
        raise ValueError(
            f"Expected {expected_num_hidden_layers} decoder layers, but the checkpoint contains {num_hidden_layers}."
        )

    expected_source_keys = {"embed.weight", "pos_embed", "lm_head.weight", "lm_head.bias"}
    for layer_idx in range(num_hidden_layers):
        expected_source_keys.update(
            {
                f"torso.{layer_idx}.attn.Q.weight",
                f"torso.{layer_idx}.attn.K.weight",
                f"torso.{layer_idx}.attn.V.weight",
                f"torso.{layer_idx}.attn.O.weight",
                f"torso.{layer_idx}.attn.O.bias",
                f"torso.{layer_idx}.mlp.read_in.weight",
                f"torso.{layer_idx}.mlp.read_in.bias",
                f"torso.{layer_idx}.mlp.write_out.weight",
                f"torso.{layer_idx}.mlp.write_out.bias",
            }
        )

    actual_source_keys = set(state_dict)
    missing_source_keys = sorted(expected_source_keys - actual_source_keys)
    unexpected_source_keys = sorted(actual_source_keys - expected_source_keys)
    if missing_source_keys or unexpected_source_keys:
        raise ValueError(
            "TinyModel checkpoint keys do not match the expected architecture. "
            f"Missing keys: {missing_source_keys}. Unexpected keys: {unexpected_source_keys}."
        )

    embed_shape = tuple(state_dict["embed.weight"].shape)
    if len(embed_shape) != 2:
        raise ValueError(f"Expected `embed.weight` to have 2 dimensions, got shape {embed_shape}.")
    vocab_size, hidden_size = embed_shape

    position_shape = tuple(state_dict["pos_embed"].shape)
    if len(position_shape) != 3 or position_shape[0] != 1 or position_shape[2] != hidden_size:
        raise ValueError(
            f"Expected `pos_embed` to have shape (1, max_position_embeddings, {hidden_size}), "
            f"got {position_shape}."
        )
    max_position_embeddings = position_shape[1]

    read_in_shape = tuple(state_dict["torso.0.mlp.read_in.weight"].shape)
    if len(read_in_shape) != 2 or read_in_shape[1] != hidden_size:
        raise ValueError(
            f"Expected `torso.0.mlp.read_in.weight` to have shape (intermediate_size, {hidden_size}), "
            f"got {read_in_shape}."
        )
    intermediate_size = read_in_shape[0]
    if intermediate_size != 4 * hidden_size:
        raise ValueError(
            f"Expected the intermediate size to equal 4 * hidden_size ({4 * hidden_size}), "
            f"got {intermediate_size}."
        )
    if num_attention_heads <= 0 or hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"The hidden size ({hidden_size}) must be divisible by a positive number of attention heads, "
            f"got {num_attention_heads}."
        )

    expected_shapes = {
        "embed.weight": (vocab_size, hidden_size),
        "pos_embed": (1, max_position_embeddings, hidden_size),
        "lm_head.weight": (vocab_size, hidden_size),
        "lm_head.bias": (vocab_size,),
    }
    for layer_idx in range(num_hidden_layers):
        expected_shapes.update(
            {
                f"torso.{layer_idx}.attn.Q.weight": (hidden_size, hidden_size),
                f"torso.{layer_idx}.attn.K.weight": (hidden_size, hidden_size),
                f"torso.{layer_idx}.attn.V.weight": (hidden_size, hidden_size),
                f"torso.{layer_idx}.attn.O.weight": (hidden_size, hidden_size),
                f"torso.{layer_idx}.attn.O.bias": (hidden_size,),
                f"torso.{layer_idx}.mlp.read_in.weight": (intermediate_size, hidden_size),
                f"torso.{layer_idx}.mlp.read_in.bias": (intermediate_size,),
                f"torso.{layer_idx}.mlp.write_out.weight": (hidden_size, intermediate_size),
                f"torso.{layer_idx}.mlp.write_out.bias": (hidden_size,),
            }
        )

    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(state_dict[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Expected `{key}` to have shape {expected_shape}, got {actual_shape}.")

    config = TinyModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        hidden_act="relu",
        attention_bias=False,
        attention_output_bias=True,
        mlp_bias=True,
        lm_head_bias=True,
        embedding_initializer_range=1e-4,
        bos_token_id=9_996,
        eos_token_id=9_997,
        pad_token_id=9_998,
        tie_word_embeddings=False,
        dtype=torch.bfloat16,
    )

    converted_state_dict = {
        "model.embed_tokens.weight": state_dict["embed.weight"],
        "model.embed_positions.weight": state_dict["pos_embed"].squeeze(0).contiguous(),
        "lm_head.weight": state_dict["lm_head.weight"],
        "lm_head.bias": state_dict["lm_head.bias"],
    }
    for layer_idx in range(num_hidden_layers):
        source_prefix = f"torso.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        converted_state_dict.update(
            {
                f"{target_prefix}.self_attn.q_proj.weight": state_dict[f"{source_prefix}.attn.Q.weight"],
                f"{target_prefix}.self_attn.k_proj.weight": state_dict[f"{source_prefix}.attn.K.weight"],
                f"{target_prefix}.self_attn.v_proj.weight": state_dict[f"{source_prefix}.attn.V.weight"],
                f"{target_prefix}.self_attn.o_proj.weight": state_dict[f"{source_prefix}.attn.O.weight"],
                f"{target_prefix}.self_attn.o_proj.bias": state_dict[f"{source_prefix}.attn.O.bias"],
                f"{target_prefix}.mlp.fc1.weight": state_dict[f"{source_prefix}.mlp.read_in.weight"],
                f"{target_prefix}.mlp.fc1.bias": state_dict[f"{source_prefix}.mlp.read_in.bias"],
                f"{target_prefix}.mlp.fc2.weight": state_dict[f"{source_prefix}.mlp.write_out.weight"],
                f"{target_prefix}.mlp.fc2.bias": state_dict[f"{source_prefix}.mlp.write_out.bias"],
            }
        )

    if len(converted_state_dict) != len(state_dict):
        raise ValueError(
            f"Converted {len(converted_state_dict)} tensors from a checkpoint containing {len(state_dict)} tensors."
        )
    return config, converted_state_dict


def convert_tiny_model_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_attention_heads: int = 16,
    expected_num_hidden_layers: int | None = None,
) -> TinyModelForCausalLM:
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    config, converted_state_dict = _convert_state_dict(
        state_dict,
        num_attention_heads=num_attention_heads,
        expected_num_hidden_layers=expected_num_hidden_layers,
    )

    with torch.device("meta"):
        model = TinyModelForCausalLM(config)

    expected_target_keys = set(model.state_dict())
    actual_target_keys = set(converted_state_dict)
    missing_target_keys = sorted(expected_target_keys - actual_target_keys)
    unexpected_target_keys = sorted(actual_target_keys - expected_target_keys)
    if missing_target_keys or unexpected_target_keys:
        raise ValueError(
            "Converted TinyModel keys do not match the native model. "
            f"Missing keys: {missing_target_keys}. Unexpected keys: {unexpected_target_keys}."
        )

    model.load_state_dict(converted_state_dict, strict=True, assign=True)
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    reloaded_model = TinyModelForCausalLM.from_pretrained(output_dir, dtype=torch.bfloat16)
    reloaded_state_dict = reloaded_model.state_dict()
    if set(reloaded_state_dict) != actual_target_keys:
        raise ValueError("The saved TinyModel checkpoint changed its state-dict keys during reload.")
    for key, expected_tensor in converted_state_dict.items():
        actual_tensor = reloaded_state_dict[key]
        if actual_tensor.dtype != torch.bfloat16:
            raise ValueError(f"Expected `{key}` to remain bfloat16 after reload, got {actual_tensor.dtype}.")
        if not torch.equal(actual_tensor.cpu(), expected_tensor.cpu()):
            raise ValueError(f"Tensor `{key}` changed during the save and reload round trip.")

    if reloaded_model.model.embed_tokens.weight.data_ptr() == reloaded_model.lm_head.weight.data_ptr():
        raise ValueError("TinyModel token embeddings and language-modeling head must remain untied.")
    return reloaded_model
