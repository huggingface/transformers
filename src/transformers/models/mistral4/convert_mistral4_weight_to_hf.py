# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from transformers import (
    GenerationConfig,
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Mistral4Config,
    PixtralImageProcessorFast,
    PixtralProcessor,
    PixtralVisionConfig,
)
from transformers.core_model_loading import (
    Concatenate,
    ConversionOps,
    MergeModulelist,
    WeightRenaming,
)
from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear
from transformers.integrations.mistral import convert_tekken_tokenizer
from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM
from transformers.quantizers.auto import AutoQuantizationConfig


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max

EXPERT_KEY_PATTERN = re.compile(r"^layers\.(\d+)\.experts\.(\d+)\.(w[123])\.(weight|qscale_act|qscale_weight)$")


class FP8RescaleMergeAndConcatenate(ConversionOps):
    r"""FP8-aware gate+up expert fusion with per-expert scale rescaling.

    Takes per-expert gate (w1) and up (w3) weight tensors together with their
    FP8 ``weight_scale_inv`` values, rescales both to a common scale per expert
    (the max of the two), concatenates gate+up along ``dim=0``, and stacks
    across experts along a new leading dimension.

    Expected ``input_dict`` keys::

        "w1.weight":        list of per-expert gate weight tensors
        "w3.weight":        list of per-expert up weight tensors
        "w1.qscale_weight": list of per-expert gate scale_inv tensors
        "w3.qscale_weight": list of per-expert up scale_inv tensors

    Produced output keys::

        "gate_up_proj":           fused weight  (n_experts, 2*intermediate, hidden)
        "gate_up_proj_scale_inv": fused scale   (n_experts,)
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        w1_weights = input_dict["w1.weight"]
        w3_weights = input_dict["w3.weight"]
        w1_scales = input_dict["w1.qscale_weight"]
        w3_scales = input_dict["w3.qscale_weight"]

        gate_up_list: list[torch.Tensor] = []
        scale_inv_list: list[torch.Tensor] = []

        for e in range(len(w1_weights)):
            fused_scale_inv = torch.max(w1_scales[e], w3_scales[e])
            gate = _rescale_fp8(w1_weights[e], w1_scales[e], fused_scale_inv)
            up = _rescale_fp8(w3_weights[e], w3_scales[e], fused_scale_inv)
            gate_up_list.append(torch.cat([gate, up], dim=0))
            scale_inv_list.append(fused_scale_inv)

        gate_up_proj_scale_inv = torch.stack(scale_inv_list)
        while gate_up_proj_scale_inv.ndim < 3:
            gate_up_proj_scale_inv = gate_up_proj_scale_inv.unsqueeze(-1)

        return {
            "gate_up_proj": torch.stack(gate_up_list, dim=0),
            "gate_up_proj_scale_inv": gate_up_proj_scale_inv,
        }


def _get_text_renamings(prefix: str) -> list[WeightRenaming]:
    r"""Build ``WeightRenaming`` list for text-model keys."""
    return [
        WeightRenaming("^output", "lm_head"),
        WeightRenaming("^norm", f"{prefix}.norm"),
        WeightRenaming("^tok_embeddings", f"{prefix}.embed_tokens"),
        WeightRenaming("^layers", f"{prefix}.layers"),
        WeightRenaming("attention_norm", "input_layernorm"),
        WeightRenaming("ffn_norm", "post_attention_layernorm"),
        WeightRenaming(r"attention\.wkv_a_with_mqa", "self_attn.kv_a_proj_with_mqa"),
        WeightRenaming(r"attention\.wkv_b", "self_attn.kv_b_proj"),
        WeightRenaming(r"attention\.wq_a", "self_attn.q_a_proj"),
        WeightRenaming(r"attention\.wq_b", "self_attn.q_b_proj"),
        WeightRenaming(r"attention\.wo", "self_attn.o_proj"),
        WeightRenaming(r"attention\.q_a_norm", "self_attn.q_a_layernorm"),
        WeightRenaming(r"attention\.kv_a_norm", "self_attn.kv_a_layernorm"),
        WeightRenaming(r"\.gate\.weight", ".mlp.gate.weight"),
        WeightRenaming(r"shared_experts\.w1", "mlp.shared_experts.gate_proj"),
        WeightRenaming(r"shared_experts\.w2", "mlp.shared_experts.down_proj"),
        WeightRenaming(r"shared_experts\.w3", "mlp.shared_experts.up_proj"),
        WeightRenaming(r"\.qscale_act", ".activation_scale"),
        WeightRenaming(r"\.qscale_weight", ".weight_scale_inv"),
    ]


def _get_vision_renamings() -> list[WeightRenaming]:
    r"""Build ``WeightRenaming`` list for vision-model keys."""
    return [
        WeightRenaming("^vision_encoder", "model.vision_tower"),
        WeightRenaming(r"^vision_language_adapter\.w_in", "model.multi_modal_projector.linear_1"),
        WeightRenaming(r"^vision_language_adapter\.w_out", "model.multi_modal_projector.linear_2"),
        WeightRenaming("^patch_merger", "model.multi_modal_projector.patch_merger"),
        WeightRenaming("^pre_mm_projector_norm", "model.multi_modal_projector.norm"),
        WeightRenaming(r"attention\.wq\.", "attention.q_proj."),
        WeightRenaming(r"attention\.wk\.", "attention.k_proj."),
        WeightRenaming(r"attention\.wv\.", "attention.v_proj."),
        WeightRenaming(r"attention\.wo\.", "attention.o_proj."),
        WeightRenaming(r"feed_forward\.w1", "feed_forward.gate_proj"),
        WeightRenaming(r"feed_forward\.w2", "feed_forward.down_proj"),
        WeightRenaming(r"feed_forward\.w3", "feed_forward.up_proj"),
    ]


_VISION_KEY_PREFIXES = ("vision_encoder.", "vision_language_adapter.", "patch_merger.", "pre_mm_projector_norm.")


def _is_vision_key(key: str) -> bool:
    r"""Return whether *key* belongs to the vision / projector components."""
    return key.startswith(_VISION_KEY_PREFIXES)


def _rename_key(
    key: str,
    text_renamings: list[WeightRenaming],
    vision_renamings: list[WeightRenaming],
) -> str:
    r"""Apply the appropriate ``WeightRenaming`` chain to *key*."""
    renamings = vision_renamings if _is_vision_key(key) else text_renamings
    for renaming in renamings:
        key, _ = renaming.rename_source_key(key)
    return key


def _rescale_fp8(
    tensor: torch.Tensor,
    original_scale_inv: torch.Tensor,
    target_scale_inv: torch.Tensor,
) -> torch.Tensor:
    r"""Rescale an FP8 tensor from *original_scale_inv* to *target_scale_inv*."""
    ratio = original_scale_inv / target_scale_inv
    return (tensor.to(torch.bfloat16) * ratio).clamp(min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)


def _descale_fp8_to_bf16(tensor: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    r"""Descale an FP8 tensor back to BF16 using its ``weight_scale_inv``."""
    return (tensor.to(torch.bfloat16) * scale_inv.to(torch.bfloat16)).to(torch.bfloat16)


def _permute_for_rope(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    r"""Permute Q/K weight matrices from Mistral's interleaved layout to HF's contiguous-half layout."""
    tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(dim1, dim2)
    return tensor


def _maybe_permute_vision_rope(
    new_key: str,
    tensor: torch.Tensor,
    vision_config: PixtralVisionConfig,
) -> torch.Tensor:
    r"""Apply RoPE permutation to vision Q/K weight matrices if applicable."""
    num_attention_heads = vision_config.num_attention_heads
    hidden_size = vision_config.hidden_size
    head_dim = vision_config.head_dim
    attn_dim = head_dim * num_attention_heads

    if "q_proj" in new_key and new_key.endswith("weight"):
        tensor = _permute_for_rope(tensor, num_attention_heads, attn_dim, hidden_size)
    elif "k_proj" in new_key and new_key.endswith("weight"):
        tensor = _permute_for_rope(tensor, num_attention_heads, attn_dim, hidden_size)

    return tensor


def _fuse_experts_for_layer(
    grouped: dict[tuple, dict[int, torch.Tensor]],
    layer_idx: int,
    n_experts: int,
    base: str,
    output_fp8: bool,
) -> dict[str, torch.Tensor]:
    r"""Fuse per-expert weights for a single layer.

    When *output_fp8* is ``True``, uses :class:`FP8RescaleMergeAndConcatenate`
    to rescale gate/up weights to a common FP8 scale before concatenation and
    emits the fused scale tensors alongside the weights.

    When *output_fp8* is ``False``, descales FP8 weights to BF16 (if the source
    is FP8) and uses :class:`MergeModulelist` + :class:`Concatenate` for the
    standard fuse-and-stack pipeline.
    """
    merge_op = MergeModulelist(dim=0)

    w1 = grouped[(layer_idx, "w1", "weight")]
    w2 = grouped[(layer_idx, "w2", "weight")]
    w3 = grouped[(layer_idx, "w3", "weight")]

    w1_scales = grouped.get((layer_idx, "w1", "qscale_weight"))
    w2_scales = grouped.get((layer_idx, "w2", "qscale_weight"))
    w3_scales = grouped.get((layer_idx, "w3", "qscale_weight"))

    result: dict[str, torch.Tensor] = {}

    if output_fp8:
        fp8_fuse_op = FP8RescaleMergeAndConcatenate()
        gate_up_result = fp8_fuse_op.convert(
            input_dict={
                "w1.weight": [w1[e] for e in range(n_experts)],
                "w3.weight": [w3[e] for e in range(n_experts)],
                "w1.qscale_weight": [w1_scales[e] for e in range(n_experts)],
                "w3.qscale_weight": [w3_scales[e] for e in range(n_experts)],
            },
            source_patterns=["w1.weight", "w3.weight", "w1.qscale_weight", "w3.qscale_weight"],
            target_patterns=["gate_up_proj", "gate_up_proj_scale_inv"],
        )
        result[f"{base}.gate_up_proj"] = gate_up_result["gate_up_proj"]
        result[f"{base}.gate_up_proj_scale_inv"] = gate_up_result["gate_up_proj_scale_inv"]

        down_result = merge_op.convert(
            input_dict={"w2": [w2[e] for e in range(n_experts)]},
            source_patterns=["w2"],
            target_patterns=["down_proj"],
        )
        result[f"{base}.down_proj"] = down_result["down_proj"]
        down_proj_scale_inv = torch.stack([w2_scales[e] for e in range(n_experts)])
        while down_proj_scale_inv.ndim < 3:
            down_proj_scale_inv = down_proj_scale_inv.unsqueeze(-1)
        result[f"{base}.down_proj_scale_inv"] = down_proj_scale_inv

        w1_act = grouped.get((layer_idx, "w1", "qscale_act"))
        if w1_act is not None:
            w2_act = grouped[(layer_idx, "w2", "qscale_act")]
            w3_act = grouped[(layer_idx, "w3", "qscale_act")]
            result[f"{base}.gate_up_proj_activation_scale"] = torch.stack(
                [torch.max(w1_act[e], w3_act[e]) for e in range(n_experts)]
            )
            result[f"{base}.down_proj_activation_scale"] = torch.stack([w2_act[e] for e in range(n_experts)])
    else:
        concat_op = Concatenate(dim=1)

        w1_list = [_descale_fp8_to_bf16(w1[e], w1_scales[e]) if w1_scales else w1[e] for e in range(n_experts)]
        w3_list = [_descale_fp8_to_bf16(w3[e], w3_scales[e]) if w3_scales else w3[e] for e in range(n_experts)]
        w2_list = [_descale_fp8_to_bf16(w2[e], w2_scales[e]) if w2_scales else w2[e] for e in range(n_experts)]

        step1 = merge_op.convert(
            input_dict={"w1": w1_list, "w3": w3_list},
            source_patterns=["w1", "w3"],
            target_patterns=["gate_up_proj"],
        )
        gate_up = concat_op.convert(step1, source_patterns=["w1", "w3"], target_patterns=["gate_up_proj"])
        result[f"{base}.gate_up_proj"] = gate_up["gate_up_proj"]

        down = merge_op.convert(
            input_dict={"w2": w2_list},
            source_patterns=["w2"],
            target_patterns=["down_proj"],
        )
        result[f"{base}.down_proj"] = down["down_proj"]

    return result


def fuse_experts(
    expert_weights: dict[tuple, torch.Tensor],
    n_experts: int,
    has_vision: bool,
    output_fp8: bool,
) -> dict[str, torch.Tensor]:
    r"""Fuse per-expert weights across all layers.

    Args:
        expert_weights: Mapping from ``(layer_idx, expert_idx, param_type, suffix)``
            to tensor.
        n_experts: Number of routed experts.
        has_vision: Whether the model is a VLM.
        output_fp8: If ``True``, keep FP8 format with rescaled scales.
            If ``False``, descale to BF16.
    """
    prefix = "model.language_model" if has_vision else "model"

    grouped: dict[tuple, dict[int, torch.Tensor]] = defaultdict(dict)
    for (layer_idx, expert_idx, param_type, suffix), tensor in expert_weights.items():
        grouped[(layer_idx, param_type, suffix)][int(expert_idx)] = tensor

    consumed_keys: set[tuple] = set()
    result: dict[str, torch.Tensor] = {}
    layers = sorted({layer_idx for (layer_idx, _, _) in grouped})

    for layer_idx in layers:
        base = f"{prefix}.layers.{layer_idx}.mlp.experts"

        w1_weight_key = (layer_idx, "w1", "weight")
        assert w1_weight_key in grouped, f"Layer {layer_idx}: missing w1 weights"
        assert len(grouped[w1_weight_key]) == n_experts, (
            f"Layer {layer_idx}: expected {n_experts} w1 experts, got {len(grouped[w1_weight_key])}"
        )

        for param in ("w1", "w2", "w3"):
            for suffix in ("weight", "qscale_weight", "qscale_act"):
                key = (layer_idx, param, suffix)
                if key in grouped:
                    consumed_keys.add(key)

        layer_result = _fuse_experts_for_layer(grouped, layer_idx, n_experts, base, output_fp8)

        result.update(layer_result)

    unconsumed = set(grouped.keys()) - consumed_keys
    assert not unconsumed, f"Unconsumed expert groups: {unconsumed}"

    return result


def convert_state_dict(
    original_state_dict: dict[str, torch.Tensor],
    text_renamings: list[WeightRenaming],
    vision_renamings: list[WeightRenaming],
    total_keys_seen: set[str],
    vision_config: PixtralVisionConfig | None = None,
    is_fp8_source: bool = False,
    output_bf16: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[tuple, torch.Tensor]]:
    r"""Rename and optionally descale one shard of the original state dict.

    Args:
        original_state_dict: Weights from a single safetensors shard.
        text_renamings: ``WeightRenaming`` list for text keys.
        vision_renamings: ``WeightRenaming`` list for vision keys.
        total_keys_seen: Accumulator for duplicate detection across shards.
        vision_config: Vision config for RoPE permutation (``None`` for text-only).
        is_fp8_source: Whether the source checkpoint uses FP8 quantization.
        output_bf16: If ``True`` and source is FP8, descale weights to BF16.

    Returns:
        Tuple of (renamed state dict, expert weights dict).
    """
    new_dict: dict[str, torch.Tensor] = {}
    expert_weights: dict[tuple, torch.Tensor] = {}

    for old_key, tensor in original_state_dict.items():
        assert old_key not in total_keys_seen, f"Duplicate key across shards: {old_key}"
        total_keys_seen.add(old_key)

        match = EXPERT_KEY_PATTERN.match(old_key)
        if match:
            layer_idx, expert_idx, param_type, suffix = int(match[1]), int(match[2]), match[3], match[4]
            expert_weights[(layer_idx, expert_idx, param_type, suffix)] = tensor
            continue

        if output_bf16 and is_fp8_source:
            if old_key.endswith((".qscale_act", ".qscale_weight")):
                continue
            if old_key.endswith(".weight"):
                scale_key = old_key.rsplit(".weight", 1)[0] + ".qscale_weight"
                if scale_key in original_state_dict:
                    tensor = _descale_fp8_to_bf16(tensor, original_state_dict[scale_key])

        new_key = _rename_key(old_key, text_renamings, vision_renamings)

        if vision_config is not None and "vision_tower" in new_key:
            tensor = _maybe_permute_vision_rope(new_key, tensor, vision_config)

        new_dict[new_key] = tensor

    return new_dict, expert_weights


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def convert_config(
    original_config: dict,
    max_position_embeddings: int = 1_048_576,
    is_vision: bool = True,
    output_fp8: bool = True,
) -> Mistral3Config | Mistral4Config:
    r"""Convert original Mistral ``params.json`` to a HF config object.

    Args:
        original_config: Parsed ``params.json`` dict (will be mutated).
        max_position_embeddings: Fallback value when not in config.
        is_vision: Whether the model includes a vision encoder.
        output_fp8: Whether to include FP8 ``quantization_config``.
    """
    original_vision_config = original_config.pop("vision_encoder", None)
    assert is_vision == (original_vision_config is not None)

    text_kwargs: dict[str, Any] = {
        "hidden_size": original_config["dim"],
        "num_hidden_layers": original_config["n_layers"],
        "intermediate_size": original_config["hidden_dim"],
        "num_attention_heads": original_config["n_heads"],
        "num_key_value_heads": original_config["n_kv_heads"],
        "rms_norm_eps": original_config["norm_eps"],
        "vocab_size": original_config["vocab_size"],
        "tie_word_embeddings": original_config.get("tied_embeddings", False),
        "sliding_window": int(original_config["sliding_window"])
        if original_config.get("sliding_window") is not None
        else None,
        "max_position_embeddings": original_config.get(
            "max_position_embeddings", original_config.get("max_seq_len", max_position_embeddings)
        ),
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 11,
    }

    for key in ["q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "kv_lora_rank", "v_head_dim"]:
        if key in original_config:
            text_kwargs[key] = original_config[key]

    moe = original_config.get("moe")
    assert moe is not None
    text_kwargs.update(
        {
            "n_routed_experts": moe.get("num_experts", 128),
            "num_experts_per_tok": moe.get("num_experts_per_tok", 4),
            "first_k_dense_replace": moe.get("first_k_dense_replace", 0),
            "n_shared_experts": moe.get("num_shared_experts", 1),
            "moe_intermediate_size": moe.get("expert_hidden_dim", 2048),
            "routed_scaling_factor": moe.get("routed_scale", 1.0),
            "n_group": moe.get("num_expert_groups", 1),
            "topk_group": moe.get("num_expert_groups_per_tok", 1),
            "norm_topk_prob": True,
        }
    )

    text_kwargs["rope_parameters"] = {
        "type": "yarn",
        "rope_theta": original_config.get("rope_theta", 10_000.0),
        "factor": float(original_config["yarn"]["factor"]),
        "original_max_position_embeddings": original_config["yarn"]["original_max_position_embeddings"],
        "beta_fast": float(original_config["yarn"]["beta"]),
        "beta_slow": float(original_config["yarn"]["alpha"]),
        "mscale_all_dim": 1.0,
        "mscale": 1.0,
        "llama_4_scaling_beta": original_config.get("llama_4_scaling", {}).get("beta", 0.1),
    }

    quant_kwargs: dict[str, Any] = {}
    quant = original_config.get("quantization", {})
    if output_fp8 and quant.get("qformat_weight") == "fp8_e4m3":
        assert quant["qscheme_act"] == "TENSOR"
        quant_kwargs["quantization_config"] = AutoQuantizationConfig.from_dict(
            {
                "activation_scheme": "static",
                "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head"],
                "quant_method": "fp8",
                "weight_block_size": None,
            }
        )

    if not is_vision:
        return Mistral4Config(**text_kwargs, **quant_kwargs)

    text_config = Mistral4Config(**text_kwargs)
    adapter_bias = original_vision_config.pop("adapter_bias", False)
    spatial_merge_size = original_vision_config.pop("spatial_merge_size")
    image_token_id = original_vision_config.pop("image_token_id", 10)
    for drop_key in [
        "mm_projector_id",
        "add_pre_mm_projector_layer_norm",
        "image_break_token_id",
        "image_end_token_id",
        "max_image_size",
    ]:
        original_vision_config.pop(drop_key, None)
    vision_config = PixtralVisionConfig(hidden_act="silu", **original_vision_config)

    return Mistral3Config(
        vision_config=vision_config,
        text_config=text_config,
        multimodal_projector_bias=adapter_bias,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
        vision_feature_layer=-1,
        tie_word_embeddings=text_kwargs["tie_word_embeddings"],
        **quant_kwargs,
    )


def convert_and_write_model(
    input_dir: Path,
    output_dir: Path,
    max_position_embeddings: int,
    output_format: str,
) -> Mistral3Config | Mistral4Config:
    r"""Convert weights and write the HF model to *output_dir*.

    Args:
        input_dir: Directory with ``params.json`` and ``*.safetensors``.
        output_dir: Output directory for the HF model.
        max_position_embeddings: Fallback ``max_position_embeddings``.
        output_format: ``"fp8"`` to keep FP8 quantization, ``"bf16"`` to descale.

    Returns:
        The converted HF config object.
    """
    params = _read_json(input_dir / "params.json")
    is_vision = params.get("vision_encoder") is not None
    is_fp8_source = params.get("quantization", {}).get("qformat_weight") == "fp8_e4m3"
    output_fp8 = output_format == "fp8" and is_fp8_source
    output_bf16 = not output_fp8

    config = convert_config(params, max_position_embeddings, is_vision, output_fp8)

    text_config = config.text_config if isinstance(config, Mistral3Config) else config
    n_experts = text_config.n_routed_experts
    vision_config = config.vision_config if isinstance(config, Mistral3Config) else None

    model_prefix = "model.language_model" if is_vision else "model"
    text_renamings = _get_text_renamings(model_prefix)
    vision_renamings = _get_vision_renamings() if is_vision else []

    full_state_dict: dict[str, torch.Tensor] = {}
    all_expert_weights: dict[tuple, torch.Tensor] = {}
    total_keys_seen: set[str] = set()
    shards = sorted(p for p in input_dir.iterdir() if p.suffix == ".safetensors")
    assert shards, f"No .safetensors files found in {input_dir}"

    for shard_path in shards:
        print(f"Processing shard: {shard_path.name}")
        original = load_file(str(shard_path))
        new_dict, expert_weights = convert_state_dict(
            original,
            text_renamings,
            vision_renamings,
            total_keys_seen,
            vision_config,
            is_fp8_source,
            output_bf16,
        )
        del original
        full_state_dict.update(new_dict)
        del new_dict
        all_expert_weights.update(expert_weights)
        del expert_weights

    print(f"Fusing {len(all_expert_weights)} expert weight entries...")
    fused = fuse_experts(all_expert_weights, n_experts, is_vision, output_fp8)
    del all_expert_weights
    full_state_dict.update(fused)
    del fused

    if text_config.tie_word_embeddings:
        full_state_dict["lm_head.weight"] = full_state_dict[f"{model_prefix}.embed_tokens.weight"]

    with torch.device("meta"):
        if isinstance(config, Mistral3Config):
            model = Mistral3ForConditionalGeneration(config)
        else:
            model = Mistral4ForCausalLM(config)

        if output_fp8 and hasattr(model.config, "quantization_config"):
            qconfig = model.config.quantization_config
            model = replace_with_fp8_linear(model, qconfig.modules_to_not_convert, qconfig)

    model.load_state_dict(full_state_dict, strict=True, assign=True)
    model.save_pretrained(str(output_dir))
    return config


def convert_and_write_processor_and_tokenizer(
    input_dir: Path,
    output_dir: Path,
    model_config: Mistral3Config | Mistral4Config,
) -> None:
    r"""Convert and write tokenizer (and processor for VLMs) to *output_dir*."""
    tokenizer = convert_tekken_tokenizer(str(input_dir / "tekken.json"))

    if isinstance(model_config, Mistral4Config):
        tokenizer.save_pretrained(str(output_dir))
        return

    params = _read_json(input_dir / "params.json")
    ve = params["vision_encoder"]

    processor = PixtralProcessor(
        tokenizer=tokenizer,
        image_processor=PixtralImageProcessorFast(
            patch_size=ve["patch_size"], size={"longest_edge": ve["max_image_size"]}
        ),
        image_token="[IMG]",
        patch_size=ve["patch_size"],
        spatial_merge_size=ve["spatial_merge_size"],
    )
    processor.save_pretrained(str(output_dir))

    text_config = model_config.text_config if hasattr(model_config, "text_config") else model_config
    GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_length=text_config.max_position_embeddings,
    ).save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Mistral4 weights to HuggingFace format.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Mistral weights (params.json, tekken.json, *.safetensors)",
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for HF model")
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=1_048_576,
        help="max_position_embeddings (used when not specified in params.json)",
    )
    parser.add_argument(
        "--output_format",
        choices=["fp8", "bf16"],
        default="fp8",
        help="Output format: 'fp8' keeps FP8 quantization (default), 'bf16' descales to BF16",
    )
    args = parser.parse_args()

    config = convert_and_write_model(args.input_dir, args.output_dir, args.max_position_embeddings, args.output_format)
    convert_and_write_processor_and_tokenizer(args.input_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
