# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
import os
import pickle
import re
from pathlib import Path
from typing import Callable, Optional

import torch
from safetensors.torch import save_file


# Avoid Using Megatron Lib
class UnpicklerWrapper(pickle.Unpickler):
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm") or mod_name.startswith("__main__"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


def dict_access_multi(a_dict, keys):
    if len(keys) == 0:
        return a_dict
    return dict_access_multi(a_dict[keys[0]], keys[1:])


def merge_qkv(
    sd_list,
    original_tp,
    num_attention_heads,
    multi_query_group_num,
    attention_dim,
    multi_query_attention,
    interleaved_qkv,
):
    if not multi_query_attention and interleaved_qkv:
        return torch.cat(sd_list, dim=0)
    q, k, v = [], [], []
    for sd in sd_list:
        if multi_query_attention:
            q_, k_, v_ = sd.split(
                [
                    num_attention_heads * attention_dim // original_tp,
                    multi_query_group_num * attention_dim // original_tp,
                    multi_query_group_num * attention_dim // original_tp,
                ],
                dim=0,
            )
        else:
            q_, k_, v_ = sd.chunk(dim=0, chunks=3)
        q.append(q_.clone())
        k.append(k_.clone())
        v.append(v_.clone())
    q = torch.cat(q, dim=0)
    k = torch.cat(k, dim=0)
    v = torch.cat(v, dim=0)
    if not interleaved_qkv:
        rotary_dim = attention_dim // 2
        half_rot = rotary_dim // 2
        perm_rot = torch.empty(rotary_dim, dtype=torch.long)
        perm_rot[0::2] = torch.arange(0, half_rot)
        perm_rot[1::2] = torch.arange(half_rot, rotary_dim)
        if q.dim() == 2:
            qh = q.view(num_attention_heads, attention_dim, -1)
            kh = k.view(multi_query_group_num, attention_dim, -1)
            qh[:, :rotary_dim, :] = qh[:, perm_rot, :]
            kh[:, :rotary_dim, :] = kh[:, perm_rot, :]
            q = qh.reshape(-1, q.size(-1))
            k = kh.reshape(-1, k.size(-1))
        else:
            qh = q.view(num_attention_heads, attention_dim)
            kh = k.view(multi_query_group_num, attention_dim)
            qh[:, :rotary_dim] = qh[:, perm_rot]
            kh[:, :rotary_dim] = kh[:, perm_rot]
            q = qh.reshape(-1)
            k = kh.reshape(-1)
    return q, k, v


def merge_glu(sd_list):
    return torch.cat(
        [sd.chunk(dim=0, chunks=2)[0].clone() for sd in sd_list]
        + [sd.chunk(dim=0, chunks=2)[1].clone() for sd in sd_list],
        dim=0,
    )


def merge_glu_vit(sd_list, original_tp=None):
    gate_proj = torch.cat([sd.chunk(dim=0, chunks=2)[0].clone() for sd in sd_list], dim=0)
    up_proj = torch.cat([sd.chunk(dim=0, chunks=2)[1].clone() for sd in sd_list], dim=0)
    return gate_proj, up_proj


def split_glu(sd, cnt, idx):
    return torch.cat(
        (
            sd.chunk(dim=0, chunks=2)[0].chunk(cnt, dim=0)[idx].clone(),
            sd.chunk(dim=0, chunks=2)[1].chunk(cnt, dim=0)[idx].clone(),
        ),
        dim=0,
    )


def merge_qkv_vit(sd_list, original_tp=None):
    q, k, v = [], [], []
    for sd in sd_list:
        q_, k_, v_ = sd.chunk(dim=0, chunks=3)
        q.append(q_.clone().contiguous())
        k.append(k_.clone().contiguous())
        v.append(v_.clone().contiguous())
    q = torch.cat(q, dim=0)
    k = torch.cat(k, dim=0)
    v = torch.cat(v, dim=0)
    combined = torch.cat([q, k, v], dim=0)
    return combined


def merge_tensors_vit(
    tp_sd: list[dict],
    keys: list[str],
    original_tp: int,
    target_tp: int,
    slice_dim: Optional[int] = None,
    merge_fn: Optional[Callable] = None,
):
    cnt = original_tp // target_tp
    sd_list = [dict_access_multi(tp_sd[i], keys) for i in range(cnt)]
    if slice_dim is not None:
        return torch.cat(sd_list, dim=slice_dim)
    assert merge_fn is not None
    return merge_fn(sd_list, original_tp)


def merge_tensors(
    tp_sd,
    keys,
    original_tp,
    target_tp,
    current_tp,
    slice_dim=None,
    merge_fn=None,
):
    cnt = original_tp // target_tp
    offset = cnt * current_tp
    sd_list = [dict_access_multi(tp_sd[i + offset], keys) for i in range(cnt)]
    if slice_dim is not None:
        return torch.cat(sd_list, dim=slice_dim)
    assert merge_fn is not None
    return merge_fn(sd_list)


def save_sharded_model(state_dict, output_path, max_shard_size_gb=5, num_layers=40, vision_num_layers=24):
    os.makedirs(output_path, exist_ok=True)

    layered_dict = {}
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        layered_dict[layer_key] = {}

        for key, value in state_dict.items():
            if f"model.language_model.layers.{layer_idx}." in key:
                layered_dict[layer_key][key] = value

    for layer_idx in range(vision_num_layers):
        layer_key = f"visual_layer_{layer_idx}"
        layered_dict[layer_key] = {}

        for key, value in state_dict.items():
            if f"model.visual.blocks.{layer_idx}." in key:
                layered_dict[layer_key][key] = value

    layered_dict["others"] = {}
    for key, value in state_dict.items():
        if not any(f"model.language_model.layers.{i}." in key for i in range(num_layers)) and not any(
            f"model.visual.blocks.{i}." in key for i in range(vision_num_layers)
        ):
            layered_dict["others"][key] = value

    # Determine layer ordering
    layer_order = []
    for i in range(40):
        layer_order.append(f"layer_{i}")
    for i in range(24):
        layer_order.append(f"visual_layer_{i}")
    layer_order.append("others")

    # Calculate sizes and create shards by layer
    param_sizes = {}
    shards = []
    current_shard = {}
    current_shard_size = 0
    max_shard_size_bytes = max_shard_size_gb * 1024 * 1024 * 1024

    for layer_key in layer_order:
        layer_weights = layered_dict[layer_key]
        layer_size = sum(param.numel() * param.element_size() for param in layer_weights.values())
        if current_shard_size + layer_size > max_shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0
        for param_name, param in layer_weights.items():
            current_shard[param_name] = param
            current_shard_size += param.numel() * param.element_size()
            param_sizes[param_name] = param.numel() * param.element_size()
    if current_shard:
        shards.append(current_shard)
    index_dict = {"metadata": {"total_size": sum(param_sizes.values())}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_filename = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        shard_path = os.path.join(output_path, shard_filename)

        for param_name in shard:
            index_dict["weight_map"][param_name] = shard_filename

        save_file(shard, shard_path, metadata={"format": "pt"})
        print(f"Saved shard {i + 1}/{len(shards)}: {shard_filename}")
        print(f"  Shard size: {sum(p.numel() * p.element_size() for p in shard.values()) / (1024**3):.2f} GB")
        print(f"  Keys in shard: {len(shard)}")

    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_dict, f, indent=2)

    return len(shards)


def merge_tp_weights(model_path, output_path, vllm_config_path=None):
    tp_size = 0
    for item in Path(model_path).iterdir():
        if item.is_dir():
            match = re.match(r"mp_rank_(\d{2})", item.name)
            if match:
                tp = int(match.group(1))
                tp_size = max(tp_size, tp + 1)

    print(f"Detected tensor parallel degree TP={tp_size}")

    if tp_size <= 1:
        print("Model is already at TP=1, no need to merge")
        return

    print(f"Loading vLLM configuration file: {vllm_config_path}")
    with open(vllm_config_path, "r") as f:
        model_config = json.load(f)
        num_layers = model_config.get("num_layers", 40)
        vision_num_layers = model_config.get("vision_config", {}).get("num_hidden_layers", 24)
        num_heads = model_config.get("num_attention_heads", 32)
        num_kv_heads = model_config.get("num_query_groups", 2)
        hidden_size = model_config.get("hidden_size", 4096)
        head_dim = model_config.get("attention_dim", hidden_size // num_heads)

    print(
        f"Model parameters: num_layers={num_layers}, vision_num_layers={vision_num_layers}, "
        f"num_heads={num_heads}, multi_query_group_num={num_kv_heads}, hidden_size={hidden_size}"
    )

    weights = []
    for tp_rank in range(tp_size):
        print(f"Loading TP shard {tp_rank}...")
        weight_path = Path(model_path) / f"mp_rank_{tp_rank:02d}" / "model_optim_rng.pt"
        sd = torch.load(weight_path, map_location="cpu", pickle_module=pickle)

        for k in list(sd.keys()):
            if "_extra_state" in k or "dummy_parameter" in k:
                sd.pop(k)

        if "model" in sd:
            weights.append(sd["model"])
        else:
            raise ValueError(f"'model' key not found in {weight_path}")

    if not weights:
        raise ValueError("No valid weight files found")

    print("Merging tensor parallel weights...")
    original_pp_enabled = os.path.exists(Path(model_path) / "mp_rank_00_000")
    original_tp, original_pp = tp_size, 1
    target_tp = 1
    print(f"TP and PP INFO: original_tp: {original_tp}, original_pp:{original_pp}, target_tp: {target_tp}")
    mgt_sd = [
        [
            torch.load(
                Path(model_path)
                / (f"mp_rank_{j:02d}_{i:03d}" if original_pp_enabled else f"mp_rank_{j:02d}")
                / "model_optim_rng.pt",
                map_location="cpu",
                pickle_module=pickle,
            )
            for j in range(original_tp)
        ]
        for i in range(original_pp)
    ]

    interleaved_qkv = False
    multi_query_attention = True
    num_attention_heads = num_heads
    multi_query_group_num = num_kv_heads
    attention_dim = head_dim
    complete_state_dict = {}
    keys = ["model"]
    rank = 0

    # LLM
    for pp in range(original_pp):
        layer_i = 0
        mgt_encoder_tp_0 = dict_access_multi(mgt_sd[pp][rank], keys)

        while f"decoder.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight" in mgt_encoder_tp_0:
            complete_state_dict.update(
                {
                    f"model.language_model.layers.{layer_i}.input_layernorm.weight": mgt_encoder_tp_0[
                        f"decoder.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight"
                    ],
                    f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight": mgt_encoder_tp_0[
                        f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight"
                    ],
                    f"model.language_model.layers.{layer_i}.post_self_attn_layernorm.weight": mgt_encoder_tp_0[
                        f"decoder.layers.{layer_i}.post_self_attn_layernorm.weight"
                    ],
                    f"model.language_model.layers.{layer_i}.post_mlp_layernorm.weight": mgt_encoder_tp_0[
                        f"decoder.layers.{layer_i}.post_mlp_layernorm.weight"
                    ],
                }
            )

            q, k, v = merge_tensors(
                tp_sd=mgt_sd[pp],
                keys=keys + [f"decoder.layers.{layer_i}.self_attention.linear_qkv.weight"],
                original_tp=original_tp,
                target_tp=target_tp,
                current_tp=0,
                merge_fn=lambda sd_list: merge_qkv(
                    sd_list,
                    original_tp,
                    num_attention_heads,
                    multi_query_group_num,
                    attention_dim,
                    multi_query_attention,
                    interleaved_qkv,
                ),
            )

            complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.q_proj.weight"] = q.clone()
            complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.k_proj.weight"] = k.clone()
            complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.v_proj.weight"] = v.clone()

            if f"decoder.layers.{layer_i}.self_attention.linear_qkv.bias" in mgt_encoder_tp_0:
                q_bias, k_bias, v_bias = merge_tensors(
                    tp_sd=mgt_sd[pp],
                    keys=keys + [f"decoder.layers.{layer_i}.self_attention.linear_qkv.bias"],
                    original_tp=original_tp,
                    target_tp=target_tp,
                    current_tp=0,
                    merge_fn=lambda sd_list: merge_qkv(
                        sd_list,
                        original_tp,
                        num_attention_heads,
                        multi_query_group_num,
                        attention_dim,
                        multi_query_attention,
                        interleaved_qkv,
                    ),
                )
                complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.q_proj.bias"] = q_bias.clone()
                complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.k_proj.bias"] = k_bias.clone()
                complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.v_proj.bias"] = v_bias.clone()

            o_proj = merge_tensors(
                tp_sd=mgt_sd[pp],
                keys=keys + [f"decoder.layers.{layer_i}.self_attention.linear_proj.weight"],
                original_tp=original_tp,
                target_tp=target_tp,
                current_tp=0,
                slice_dim=1,
            )
            complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.o_proj.weight"] = o_proj.clone()

            # MLP - Use gate_up_proj
            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.gate_up_proj.weight"] = merge_tensors(
                tp_sd=mgt_sd[pp],
                keys=keys + [f"decoder.layers.{layer_i}.mlp.linear_fc1.weight"],
                original_tp=original_tp,
                target_tp=target_tp,
                current_tp=0,
                merge_fn=merge_glu,
            ).clone()
            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.down_proj.weight"] = merge_tensors(
                tp_sd=mgt_sd[pp],
                keys=keys + [f"decoder.layers.{layer_i}.mlp.linear_fc2.weight"],
                original_tp=original_tp,
                target_tp=target_tp,
                current_tp=0,
                slice_dim=1,
            )
            layer_i += 1

    # Embedd Model, LM Head, and Norm
    embed_tokens = merge_tensors(
        tp_sd=mgt_sd[0],
        keys=["model", "embedding.word_embeddings.weight"],
        original_tp=original_tp,
        target_tp=target_tp,
        current_tp=0,
        slice_dim=0,
    )
    complete_state_dict["model.language_model.embed_tokens.weight"] = embed_tokens.clone()
    lm_head = merge_tensors(
        tp_sd=mgt_sd[-1],
        keys=["model", "output_layer.weight"],
        original_tp=original_tp,
        target_tp=target_tp,
        current_tp=0,
        slice_dim=0,
    )
    complete_state_dict["lm_head.weight"] = lm_head.clone()
    complete_state_dict["model.language_model.norm.weight"] = mgt_sd[-1][rank]["model"][
        "decoder.final_layernorm.weight"
    ].clone()
    mgt_encoder_tp_0 = dict_access_multi(mgt_sd[0][0], keys)

    # VLM
    for layer_i in range(vision_num_layers):
        complete_state_dict[f"model.visual.blocks.{layer_i}.norm1.weight"] = mgt_encoder_tp_0[
            f"vision_model.transformer.layers.{layer_i}.input_layernorm.weight"
        ]
        complete_state_dict[f"model.visual.blocks.{layer_i}.norm2.weight"] = mgt_encoder_tp_0[
            f"vision_model.transformer.layers.{layer_i}.pre_mlp_layernorm.weight"
        ]

        qkv_weight = merge_tensors_vit(
            tp_sd=mgt_sd[0],
            keys=keys + [f"vision_model.transformer.layers.{layer_i}.self_attention.linear_qkv.weight"],
            original_tp=original_tp,
            target_tp=target_tp,
            merge_fn=merge_qkv_vit,
        )
        complete_state_dict[f"model.visual.blocks.{layer_i}.attn.qkv.weight"] = qkv_weight.clone()

        proj_weight = merge_tensors_vit(
            tp_sd=mgt_sd[0],
            keys=keys + [f"vision_model.transformer.layers.{layer_i}.self_attention.linear_proj.weight"],
            original_tp=original_tp,
            target_tp=target_tp,
            slice_dim=1,
        )
        complete_state_dict[f"model.visual.blocks.{layer_i}.attn.proj.weight"] = proj_weight.clone()

        gate_proj_weight, up_proj_weight = merge_tensors_vit(
            tp_sd=mgt_sd[0],
            keys=keys + [f"vision_model.transformer.layers.{layer_i}.mlp.linear_fc1.weight"],
            original_tp=original_tp,
            target_tp=target_tp,
            merge_fn=lambda sd_list, original_tp: merge_glu_vit(sd_list, original_tp),
        )
        complete_state_dict[f"model.visual.blocks.{layer_i}.mlp.gate_proj.weight"] = gate_proj_weight.clone()
        complete_state_dict[f"model.visual.blocks.{layer_i}.mlp.up_proj.weight"] = up_proj_weight.clone()

        down_proj_weight = merge_tensors_vit(
            tp_sd=mgt_sd[0],
            keys=keys + [f"vision_model.transformer.layers.{layer_i}.mlp.linear_fc2.weight"],
            original_tp=original_tp,
            target_tp=target_tp,
            slice_dim=1,
        )
        complete_state_dict[f"model.visual.blocks.{layer_i}.mlp.down_proj.weight"] = down_proj_weight.clone()

    complete_state_dict["model.visual.downsample.weight"] = (
        mgt_sd[0][0]["model"]["vision_model.downsample.weight"].clone().contiguous()
    )
    complete_state_dict["model.visual.downsample.bias"] = (
        mgt_sd[0][0]["model"]["vision_model.downsample.bias"].clone().contiguous()
    )

    # Merger
    gate_proj, up_proj = merge_tensors_vit(
        tp_sd=mgt_sd[0],
        keys=keys + ["vision_projection.encoder.linear_fc1.weight"],
        original_tp=original_tp,
        target_tp=target_tp,
        merge_fn=merge_glu_vit,
    )

    down_proj = merge_tensors_vit(
        tp_sd=mgt_sd[0],
        keys=keys + ["vision_projection.encoder.linear_fc2.weight"],
        original_tp=original_tp,
        target_tp=target_tp,
        slice_dim=1,
    )
    proj = merge_tensors_vit(
        tp_sd=mgt_sd[0],
        keys=keys + ["vision_projection.encoder.linear_fc_extra.weight"],
        original_tp=original_tp,
        target_tp=target_tp,
        slice_dim=0,
    )

    complete_state_dict["model.visual.merger.gate_proj.weight"] = gate_proj.clone().contiguous()
    complete_state_dict["model.visual.merger.up_proj.weight"] = up_proj.clone().contiguous()
    complete_state_dict["model.visual.merger.down_proj.weight"] = down_proj.clone().contiguous()
    complete_state_dict["model.visual.merger.proj.weight"] = proj.clone().contiguous()

    complete_state_dict["model.visual.merger.post_projection_norm.weight"] = (
        mgt_sd[0][0]["model"]["vision_projection.encoder.layer_norm.weight"].clone().contiguous()
    )
    complete_state_dict["model.visual.merger.post_projection_norm.bias"] = (
        mgt_sd[0][0]["model"]["vision_projection.encoder.layer_norm.bias"].clone().contiguous()
    )
    complete_state_dict["model.visual.embeddings.position_embedding.weight"] = (
        mgt_sd[0][0]["model"]["vision_model.position_embeddings.weight"].clone().contiguous()
    )
    complete_state_dict["model.visual.patch_embed.proj.weight"] = (
        mgt_sd[0][0]["model"]["vision_model.conv3d.weight"].clone().contiguous()
    )
    complete_state_dict["model.visual.patch_embed.proj.bias"] = (
        mgt_sd[0][0]["model"]["vision_model.conv3d.bias"].clone().contiguous()
    )

    # Check for additional vision model norm layers mentioned in the expected output
    if "vision_model.post_conv_layernorm.weight" in mgt_encoder_tp_0:
        complete_state_dict["model.visual.post_conv_layernorm.weight"] = (
            mgt_sd[0][0]["model"]["vision_model.post_conv_layernorm.weight"].clone().contiguous()
        )

    if "vision_model.post_layernorm.weight" in mgt_encoder_tp_0:
        complete_state_dict["model.visual.post_layernorm.weight"] = (
            mgt_sd[0][0]["model"]["vision_model.post_layernorm.weight"].clone().contiguous()
        )

    print(f"Total keys in state dict: {len(complete_state_dict)}")

    for key, value in complete_state_dict.items():
        if isinstance(value, torch.Tensor):
            complete_state_dict[key] = value.to(torch.bfloat16)
    print("Converted all tensors to bfloat16")
    # Save Model weight
    save_sharded_model(
        complete_state_dict,
        output_path=output_path,
        max_shard_size_gb=5,
        num_layers=num_layers,
        vision_num_layers=vision_num_layers,
    )

    hf_config = {
        "architectures": ["Glm4vForConditionalGeneration"],
        "model_type": "glm4v",
        "attention_bias": model_config.get("add_qkv_bias", True),
        "attention_dropout": 0.0,
        "pad_token_id": model_config.get("pad_token_id", 151329),
        "eos_token_id": model_config.get("eos_token_id", [151329, 151336, 151338]),
        "image_start_token_id": model_config.get("image_start_token_id", 151339),
        "image_end_token_id": model_config.get("image_end_token_id", 151340),
        "video_start_token_id": model_config.get("video_start_token_id", 151341),
        "video_end_token_id": model_config.get("video_end_token_id", 151342),
        "image_token_id": model_config.get("image_token_id", 151343),
        "video_token_id": model_config.get("video_token_id", 151344),
        "hidden_act": model_config.get("hidden_act", "silu"),
        "hidden_size": model_config.get("hidden_size", 4096),
        "initializer_range": 0.02,
        "intermediate_size": model_config.get("ffn_hidden_size", 13696),
        "max_position_embeddings": model_config.get("seq_length", 32768),
        "num_attention_heads": model_config.get("num_attention_heads", 32),
        "num_hidden_layers": model_config.get("num_layers", 40),
        "num_key_value_heads": model_config.get("multi_query_group_num", 2),
        "rms_norm_eps": model_config.get("layernorm_epsilon", 1e-05),
        "rope_theta": model_config.get("rotary_base", 10000.0),
        "tie_word_embeddings": False,
        "dtype": model_config.get("dtype", "bfloat16"),
        "transformers_version": "4.53.0dev",
        "use_cache": model_config.get("use_cache", True),
        "vocab_size": model_config.get("vocab_size", 151552),
        "partial_rotary_factor": 0.5,
    }

    if "vision_config" in model_config:
        vision_config = {
            "hidden_size": model_config["vision_config"].get("hidden_size", 1536),
            "depth": model_config["vision_config"].get("num_layers", 24),
            "num_heads": model_config["vision_config"].get("num_attention_heads", 12),
            "attention_bias": model_config["vision_config"].get("attention_bias", False),
            "intermediate_size": model_config.get("ffn_hidden_size", 13696),
            "hidden_act": model_config["vision_config"].get("hidden_act", "silu"),
            "hidden_dropout_prob": model_config["vision_config"].get("hidden_dropout_prob", 0.0),
            "initializer_range": 0.02,
            "image_size": model_config["vision_config"].get("image_size", 336),
            "patch_size": model_config["vision_config"].get("patch_size", 14),
            "out_hidden_size": model_config.get("hidden_size", 4096),
            "rms_norm_eps": model_config["vision_config"].get("layernorm_epsilon", 1e-05),
            "spatial_merge_size": model_config["vision_config"].get("downsample_ratio", 2),
            "temporal_patch_size": model_config["vision_config"].get("t_patch", 2),
        }
        hf_config["vision_config"] = vision_config

    if "rope_scaling" in model_config:
        hf_config["rope_scaling"] = model_config["rope_scaling"]

    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)

    print(f"Conversion complete! Model saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Megatron model to HuggingFace format")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Megatron model directory",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output path for HuggingFace model directory")
    parser.add_argument(
        "--config_path", type=str, help="Path to vLLM configuration file for creating HuggingFace config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_tp_weights(args.model_path, args.output_path, args.config_path)
