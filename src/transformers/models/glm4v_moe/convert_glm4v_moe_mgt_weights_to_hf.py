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
    interleaved_qkv,
):
    group_size = (num_attention_heads // multi_query_group_num + 2) * attention_dim
    q, k, v = [], [], []
    for sd in sd_list:
        if interleaved_qkv:
            shape = sd.shape
            q_, k_, v_ = sd.view((multi_query_group_num // original_tp, group_size) + (shape[1:])).split(
                [
                    (num_attention_heads // multi_query_group_num * attention_dim),
                    attention_dim,
                    attention_dim,
                ],
                dim=1,
            )
            q_ = q_.reshape((-1,) + (shape[1:]))
            k_ = k_.reshape((-1,) + (shape[1:]))
            v_ = v_.reshape((-1,) + (shape[1:]))
        else:
            q_, k_, v_ = sd.split(
                [
                    num_attention_heads * attention_dim // original_tp,
                    multi_query_group_num * attention_dim // original_tp,
                    multi_query_group_num * attention_dim // original_tp,
                ],
                dim=0,
            )

        q.append(q_.clone())
        k.append(k_.clone())
        v.append(v_.clone())
    q = torch.cat(q, dim=0)
    k = torch.cat(k, dim=0)
    v = torch.cat(v, dim=0)

    return q, k, v


def merge_glu(sd_list):
    return torch.cat(
        [sd.chunk(dim=0, chunks=2)[0].clone() for sd in sd_list]
        + [sd.chunk(dim=0, chunks=2)[1].clone() for sd in sd_list],
        dim=0,
    )


def merge_glu_vit(sd_list, original_tp=None):
    if not isinstance(sd_list, list):
        sd_list = [sd_list]
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


def find_expert_weight(input_dict, layer_num, fc1=True):
    if fc1:
        pattern = re.compile(rf"^decoder\.layers\.{layer_num}\.mlp\.experts\.linear_fc1\.weight(\d+)$")
    else:
        pattern = re.compile(rf"^decoder\.layers\.{layer_num}\.mlp\.experts\.linear_fc2\.weight(\d+)$")
    matched = []
    for key in input_dict:
        match = pattern.match(key)
        if match:
            weight_num = int(match.group(1))
            matched.append((weight_num, key))
    matched.sort(key=lambda x: x[0])

    weights = [None for _ in range(len(matched) * len(input_dict[matched[0][1]]))]
    for idx, key in matched:
        for i, weight in enumerate(input_dict[key]):
            weights[i * len(matched) + idx] = weight

    return weights


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


def save_sharded_model(state_dict, output_path, max_shard_size_gb=5, num_layers=46, vision_num_layers=24):
    os.makedirs(output_path, exist_ok=True)

    layered_dict = {}
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        layered_dict[layer_key] = {}

        for key, value in state_dict.items():
            if f"model.language_model.layers.{layer_idx}." in key:
                if isinstance(value, list):
                    assert len(value) == 1, f"{key} {value}"
                    value = value[0]
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
    for i in range(num_layers):
        layer_order.append(f"layer_{i}")
    for i in range(vision_num_layers):
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
    origin_tp, origin_ep, origin_pp = -1, -1, -1

    check_ep_or_pp_later = False
    for item in Path(model_path).iterdir():
        if item.is_dir():
            match = re.match(r"mp_rank_(\d{2})(?:_(\d{3}))?(?:_(\d{3}))?", item.name)
            if match:
                groups = match.groups()
                tp = int(groups[0])
                origin_tp = max(origin_tp, tp + 1)
                # maybe TP-EP or TP-PP, need check later
                if groups[1] is not None and groups[2] is None:
                    pp = int(groups[1])
                    origin_pp = max(origin_pp, pp + 1)
                    origin_ep = 1
                    check_ep_or_pp_later = True
                elif groups[1] is not None and groups[2] is not None:
                    pp = int(groups[1])
                    ep = int(groups[2])
                    origin_pp = max(origin_pp, pp + 1)
                    origin_ep = max(origin_ep, ep + 1)
                else:
                    origin_ep = 1
                    origin_pp = 1

    tensor_names_by_file = {}
    mgt_sd = {}
    for item in Path(model_path).iterdir():
        if item.is_dir():
            match = re.match(r"mp_rank_(\d{2})(?:_(\d{3}))?(?:_(\d{3}))?$", item.name)
            if match:
                groups = match.groups()
                tp = int(groups[0])
                pp = int(groups[1]) if groups[1] is not None else 0
                ep = int(groups[2]) if groups[2] is not None else 0

                file_path = item / "model_optim_rng.pt"
                assert file_path.exists(), f"model_optim_rng.pt not found in {item}"

                file_sd = torch.load(file_path, map_location="cpu", weights_only=False)

                for k in list(file_sd.keys()):
                    if "_extra_state" in k or "dummy_parameter" in k:
                        file_sd.pop(k)

                mgt_sd[(tp, pp, ep)] = file_sd

                tensor_names = set()
                if "model" in file_sd:
                    for key in file_sd["model"].keys():
                        tensor_names.add(key)
                tensor_names_by_file[(tp, pp, ep)] = tensor_names

    change_pp_to_ep = False
    if check_ep_or_pp_later:
        prefix_distribution = {}

        for (tp, pp, ep), prefixes in tensor_names_by_file.items():
            for prefix in prefixes:
                if prefix not in prefix_distribution:
                    prefix_distribution[prefix] = set()
                prefix_distribution[prefix].add((tp, pp, ep))

        for prefix, locations in prefix_distribution.items():
            if len(locations) > 1:
                pp_values = {loc[1] for loc in locations}
                if len(pp_values) > 1:
                    print(f"find '{prefix}' in multi ranks {pp_values} the parallelism should be TP-EP")
                    origin_ep = origin_pp
                    origin_pp = 1
                    change_pp_to_ep = True
                    break
                else:
                    print(f"find '{prefix}' only in one ep, parallelism should be TP-PP")
                    break

    print(f"Detected tensor parallel degree TP={origin_tp} EP={origin_ep} PP={origin_pp}")
    if origin_tp <= 1 and origin_ep <= 1 and origin_pp <= 1:
        print("Model is already at TP=1 EP=1 PP=1, no need to merge")
        return
    assert max(origin_tp, origin_ep) * origin_pp == len(tensor_names_by_file), "maybe some problem in origin weight"

    organized_sd = {}
    for (tp, pp, ep), file_sd in mgt_sd.items():
        if change_pp_to_ep:
            pp, ep = ep, pp
        organized_sd.setdefault(pp, {})
        organized_sd[pp][(ep, tp)] = file_sd
        find_vpp = "model0" in file_sd

    # support VPP, if each pp rank has n vpp blocks, we will treat the original model
    # was parallel as pp n * origin_pp
    if find_vpp:
        organized_sd_vpp = {}
        for i in range(origin_pp):
            for (ep, tp), file_sd in organized_sd[i].items():
                model_keys = sorted(
                    [key for key in file_sd.keys() if key.startswith("model") and key[5:].isdigit()],
                    key=lambda x: int(x[5:]),
                )
                vp_blocks = len(model_keys)
                for idx, key in enumerate(model_keys):
                    assert key in file_sd, f"model {key} not found"
                    organized_sd_vpp.setdefault(idx * origin_pp + i, {})
                    organized_sd_vpp[idx * origin_pp + i][(ep, tp)] = {"model": file_sd[key]}
        origin_pp = origin_pp * vp_blocks
        organized_sd = organized_sd_vpp

    ignore_list = ["_extra_state", "dummy_parameter"]
    layer_share_list = [
        "norm",
        "conv3d",
        "downsample",
        "router",
        "mlp.linear_fc2.bias",
        "self_attention.linear_proj.bias",
        "position_embeddings",
    ]

    full_weights = {}

    vit_layer_offset = 0
    llm_layer_offset = 0
    llm_layer_pattern = re.compile(r"^(decoder\.layers\.)(\d+)(\..*)$")
    vit_layer_pattern = re.compile(r"^(vision_model\.transformer\.layers\.)(\d+)(\..*)$")
    for pp in sorted(organized_sd.keys()):
        pp_dict = organized_sd[pp]
        next_llm_layer_offset = llm_layer_offset
        next_vit_layer_offset = vit_layer_offset
        ep_map = {}
        tp_map = {}
        tp_seen = set()
        for (ep, tp), item in pp_dict.items():
            if tp not in tp_seen:
                tp_seen.add(tp)
                tp_map[tp] = item
            ep_map[ep] = item

        for tp in sorted(tp_map.keys()):
            sd = tp_map[tp]
            for full_name, tensor in sd["model"].items():
                if any(x in full_name for x in ignore_list):
                    continue
                llm_name_match = llm_layer_pattern.match(full_name)
                if llm_name_match:
                    # Use a closure to avoid global variable issues
                    def offset_layer(x, offset=llm_layer_offset):
                        nonlocal next_llm_layer_offset
                        _real_layer = int(x.group(2)) + offset
                        next_llm_layer_offset = max(next_llm_layer_offset, _real_layer + 1)
                        return f"{x.group(1)}{_real_layer}{x.group(3)}"

                    full_name = llm_layer_pattern.sub(offset_layer, full_name)
                vit_name_match = vit_layer_pattern.match(full_name)
                if vit_name_match:
                    # Use a closure to avoid global variable issues
                    def offset_layer(x, offset=vit_layer_offset):
                        nonlocal next_vit_layer_offset
                        _real_layer = int(x.group(2)) + offset
                        next_vit_layer_offset = max(next_vit_layer_offset, _real_layer + 1)
                        return f"{x.group(1)}{_real_layer}{x.group(3)}"

                    full_name = vit_layer_pattern.sub(offset_layer, full_name)
                if layer_share_list and any(x in full_name for x in layer_share_list):
                    if full_name not in full_weights:
                        full_weights[full_name] = tensor
                    else:
                        assert torch.equal(tensor, full_weights[full_name]), (
                            f"detect diff param in tp named: {full_name}"
                        )
                elif not re.search(r"\.experts\.", full_name):
                    full_weights.setdefault(full_name, [None for _ in range(origin_tp)])
                    full_weights[full_name][tp] = tensor

        for ep in sorted(ep_map.keys()):
            sd = ep_map[ep]
            for full_name, tensor in sd["model"].items():
                if any(x in full_name for x in ignore_list):
                    continue
                name_match = llm_layer_pattern.match(full_name)
                if name_match:
                    # Use a closure to avoid global variable issues
                    def offset_layer(x, offset=llm_layer_offset):
                        nonlocal next_llm_layer_offset
                        _real_layer = int(x.group(2)) + offset
                        next_llm_layer_offset = max(next_llm_layer_offset, _real_layer + 1)
                        return f"{x.group(1)}{_real_layer}{x.group(3)}"

                    full_name = llm_layer_pattern.sub(offset_layer, full_name)
                if re.search(r"\.experts\.", full_name):
                    full_weights.setdefault(full_name, [None for _ in range(origin_ep)])
                    full_weights[full_name][ep] = tensor
        llm_layer_offset = next_llm_layer_offset
        vit_layer_offset = next_vit_layer_offset

    for k in sorted(full_weights.keys()):
        item = full_weights[k]
        if isinstance(item, list):
            print(f"{k} {len(item)} {item[0].shape} {item[0].dtype}", flush=True)
        else:
            print(f"{k} {item.shape} {item.dtype}", flush=True)

    print(f"Loading vLLM configuration file: {vllm_config_path}")
    with open(vllm_config_path, "r") as f:
        model_config = json.load(f)
        print(model_config)
        text_config = model_config.get("text_config", {})
        vision_config = model_config.get("vision_config", {})

        num_layers = text_config.get("num_hidden_layers", 46)
        llm_num_heads = text_config.get("num_attention_heads", 96)
        num_kv_heads = text_config.get("num_key_value_heads", 8)
        llm_attn_query_size = text_config.get("llm_attn_query_size", 12288)
        head_dim = text_config.get("attention_dim", llm_attn_query_size // llm_num_heads)
        vision_num_layers = vision_config.get("depth", 24)
        vit_n_head = vision_config.get("num_heads", 12)

    print(
        f"Model parameters: num_layers={num_layers}, vision_num_layers={vision_num_layers}, "
        f"num_heads={llm_num_heads}, multi_query_group_num={num_kv_heads}, llm_attn_query_size={llm_attn_query_size}"
    )

    print("Merging tensor parallel weights...")

    interleaved_qkv = True
    num_attention_heads = llm_num_heads
    multi_query_group_num = num_kv_heads
    attention_dim = head_dim
    complete_state_dict = {}

    # LLM
    layer_i = 0
    while f"decoder.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight" in full_weights:
        if f"decoder.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight" in full_weights:
            complete_state_dict[f"model.language_model.layers.{layer_i}.input_layernorm.weight"] = full_weights[
                f"decoder.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight"
            ]

        if f"decoder.layers.{layer_i}.pre_mlp_layernorm.weight" in full_weights:
            complete_state_dict[f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight"] = (
                full_weights[f"decoder.layers.{layer_i}.pre_mlp_layernorm.weight"]
            )
        elif f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight" in full_weights:
            complete_state_dict[f"model.language_model.layers.{layer_i}.post_attention_layernorm.weight"] = (
                full_weights[f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight"]
            )

        q, k, v = merge_qkv(
            sd_list=full_weights[f"decoder.layers.{layer_i}.self_attention.linear_qkv.weight"],
            original_tp=origin_tp,
            num_attention_heads=num_attention_heads,
            multi_query_group_num=multi_query_group_num,
            attention_dim=attention_dim,
            interleaved_qkv=interleaved_qkv,
        )

        complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.q_proj.weight"] = q.clone()
        complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.k_proj.weight"] = k.clone()
        complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.v_proj.weight"] = v.clone()

        if f"decoder.layers.{layer_i}.self_attention.linear_qkv.bias" in full_weights:
            q_bias, k_bias, v_bias = merge_qkv(
                sd_list=full_weights[f"decoder.layers.{layer_i}.self_attention.linear_qkv.bias"],
                original_tp=origin_tp,
                num_attention_heads=num_attention_heads,
                multi_query_group_num=multi_query_group_num,
                attention_dim=attention_dim,
                interleaved_qkv=interleaved_qkv,
            )
            complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.q_proj.bias"] = q_bias.clone()
            complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.k_proj.bias"] = k_bias.clone()
            complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.v_proj.bias"] = v_bias.clone()

        o_proj = torch.cat(full_weights[f"decoder.layers.{layer_i}.self_attention.linear_proj.weight"], dim=1)
        complete_state_dict[f"model.language_model.layers.{layer_i}.self_attn.o_proj.weight"] = o_proj.clone()

        if f"decoder.layers.{layer_i}.mlp.shared_experts.linear_fc1.weight" in full_weights:
            routed_expert_fc1_weights = find_expert_weight(full_weights, layer_i, fc1=True)
            for idx, weight in enumerate(routed_expert_fc1_weights):
                gate_proj_weight, up_proj_weight = merge_glu_vit([weight])
                complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.experts.{idx}.gate_proj.weight"] = (
                    gate_proj_weight.clone()
                )
                complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.experts.{idx}.up_proj.weight"] = (
                    up_proj_weight.clone()
                )

            routed_expert_fc2_weights = find_expert_weight(full_weights, layer_i, fc1=False)
            for idx, weight in enumerate(routed_expert_fc2_weights):
                complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.experts.{idx}.down_proj.weight"] = (
                    weight.clone()
                )

            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.gate.e_score_correction_bias"] = (
                full_weights[f"decoder.layers.{layer_i}.mlp.router.expert_bias"]
            )

            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.gate.weight"] = full_weights[
                f"decoder.layers.{layer_i}.mlp.router.weight"
            ]

            gate_proj_weight, up_proj_weight = merge_glu_vit(
                full_weights[f"decoder.layers.{layer_i}.mlp.shared_experts.linear_fc1.weight"]
            )

            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.shared_experts.gate_proj.weight"] = (
                gate_proj_weight.clone()
            )

            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.shared_experts.up_proj.weight"] = (
                up_proj_weight.clone()
            )

            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.shared_experts.down_proj.weight"] = (
                full_weights[f"decoder.layers.{layer_i}.mlp.shared_experts.linear_fc2.weight"]
            )

        else:
            # MLP - Use gate_up_proj
            gate_proj_weight, up_proj_weight = merge_glu_vit(
                full_weights[f"decoder.layers.{layer_i}.mlp.linear_fc1.weight"]
            )
            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.gate_proj.weight"] = (
                gate_proj_weight.clone()
            )
            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.up_proj.weight"] = up_proj_weight.clone()
            complete_state_dict[f"model.language_model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                full_weights[f"decoder.layers.{layer_i}.mlp.linear_fc2.weight"], dim=1
            )
        layer_i += 1

    # Embedd Model, LM Head, and Norm
    embed_tokens = torch.cat(full_weights["embedding.word_embeddings.weight"], dim=0)
    complete_state_dict["model.language_model.embed_tokens.weight"] = embed_tokens.clone()

    lm_head = torch.cat(full_weights["output_layer.weight"], dim=0)
    complete_state_dict["lm_head.weight"] = lm_head.clone()
    complete_state_dict["model.language_model.norm.weight"] = full_weights["decoder.final_layernorm.weight"].clone()

    # VLM
    for layer_i in range(vision_num_layers):
        complete_state_dict[f"model.visual.blocks.{layer_i}.norm1.weight"] = full_weights[
            f"vision_model.transformer.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight"
        ]
        complete_state_dict[f"model.visual.blocks.{layer_i}.norm2.weight"] = full_weights[
            f"vision_model.transformer.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight"
        ]

        q, k, v = merge_qkv(
            sd_list=full_weights[f"vision_model.transformer.layers.{layer_i}.self_attention.linear_qkv.weight"],
            original_tp=origin_tp,
            num_attention_heads=vit_n_head,
            multi_query_group_num=vit_n_head,
            attention_dim=attention_dim,
            interleaved_qkv=interleaved_qkv,
        )
        complete_state_dict[f"model.visual.blocks.{layer_i}.attn.qkv.weight"] = torch.cat((q, k, v), dim=0)

        proj_weight = torch.cat(
            full_weights[f"vision_model.transformer.layers.{layer_i}.self_attention.linear_proj.weight"], dim=1
        )
        complete_state_dict[f"model.visual.blocks.{layer_i}.attn.proj.weight"] = proj_weight.clone()

        gate_proj_weight, up_proj_weight = merge_glu_vit(
            full_weights[f"vision_model.transformer.layers.{layer_i}.mlp.linear_fc1.weight"]
        )

        complete_state_dict[f"model.visual.blocks.{layer_i}.mlp.gate_proj.weight"] = gate_proj_weight.clone()
        complete_state_dict[f"model.visual.blocks.{layer_i}.mlp.up_proj.weight"] = up_proj_weight.clone()

        down_proj_weight = torch.cat(
            full_weights[f"vision_model.transformer.layers.{layer_i}.mlp.linear_fc2.weight"], dim=1
        )
        complete_state_dict[f"model.visual.blocks.{layer_i}.mlp.down_proj.weight"] = down_proj_weight.clone()

    complete_state_dict["model.visual.downsample.weight"] = (
        full_weights["vision_model.downsample.weight"].clone().contiguous()
    )
    complete_state_dict["model.visual.downsample.bias"] = (
        full_weights["vision_model.downsample.bias"].clone().contiguous()
    )

    # Merger
    gate_proj, up_proj = merge_glu_vit(full_weights["vision_projection.encoder.linear_fc1.weight"])

    down_proj = torch.cat(full_weights["vision_projection.encoder.linear_fc2.weight"], dim=1)
    proj = torch.cat(full_weights["vision_projection.linear_fc_extra.weight"], dim=0)

    complete_state_dict["model.visual.merger.gate_proj.weight"] = gate_proj.clone().contiguous()
    complete_state_dict["model.visual.merger.up_proj.weight"] = up_proj.clone().contiguous()
    complete_state_dict["model.visual.merger.down_proj.weight"] = down_proj.clone().contiguous()
    complete_state_dict["model.visual.merger.proj.weight"] = proj.clone().contiguous()

    if "vision_projection.layer_norm.weight" in full_weights:
        complete_state_dict["model.visual.merger.post_projection_norm.weight"] = full_weights[
            "vision_projection.layer_norm.weight"
        ]
    if "vision_projection.layer_norm.bias" in full_weights:
        complete_state_dict["model.visual.merger.post_projection_norm.bias"] = full_weights[
            "vision_projection.layer_norm.bias"
        ]

    complete_state_dict["model.visual.embeddings.position_embedding.weight"] = (
        full_weights["vision_model.position_embeddings.weight"].clone().contiguous()
    )
    complete_state_dict["model.visual.patch_embed.proj.weight"] = (
        full_weights["vision_model.conv3d.weight"].clone().contiguous()
    )
    complete_state_dict["model.visual.patch_embed.proj.bias"] = (
        full_weights["vision_model.conv3d.bias"].clone().contiguous()
    )

    # Check for additional vision model norm layers mentioned in the expected output
    if "vision_model.post_conv_layernorm.weight" in full_weights:
        complete_state_dict["model.visual.post_conv_layernorm.weight"] = (
            full_weights["vision_model.post_conv_layernorm.weight"].clone().contiguous()
        )

    if "vision_model.post_layernorm.weight" in full_weights:
        complete_state_dict["model.visual.post_layernorm.weight"] = (
            full_weights["vision_model.post_layernorm.weight"].clone().contiguous()
        )

    print(f"Total keys in state dict: {len(complete_state_dict)}")

    print("bias use Float32")

    save_sharded_model(
        complete_state_dict,
        output_path=output_path,
        max_shard_size_gb=5,
        num_layers=num_layers,
        vision_num_layers=vision_num_layers,
    )

    hf_config = {
        "architectures": ["Glm4vMoeForConditionalGeneration"],
        "model_type": "glm4v_moe",
        "image_start_token_id": model_config.get("image_start_token_id", 151339),
        "image_end_token_id": model_config.get("image_end_token_id", 151340),
        "video_start_token_id": model_config.get("video_start_token_id", 151341),
        "video_end_token_id": model_config.get("video_end_token_id", 151342),
        "transformers_version": "4.57.0.dev0",
    }
    txt_config = {
        "model_type": "glm4v_moe_text",
        "attention_bias": model_config.get("add_qkv_bias", True),
        "use_qk_norm": model_config.get("use_qk_norm", False),
        "attention_dropout": 0.0,
        "pad_token_id": model_config.get("pad_token_id", 151329),
        "eos_token_id": model_config.get("eos_token_id", [151329, 151336, 151338]),
        "image_token_id": model_config.get("image_token_id", 151363),
        "video_token_id": model_config.get("video_token_id", 151364),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "hidden_size": text_config.get("hidden_size", 4096),
        "initializer_range": 0.02,
        "intermediate_size": text_config.get("intermediate_size", 10944),
        "max_position_embeddings": text_config.get("seq_length", 131072),
        "num_attention_heads": text_config.get("num_attention_heads", 96),
        "num_hidden_layers": text_config.get("num_layers", 46),
        "num_key_value_heads": text_config.get("multi_query_group_num", 2),
        "rms_norm_eps": text_config.get("layernorm_epsilon", 1e-05),
        "dtype": text_config.get("torch_dtype", "bfloat16"),
        "use_cache": text_config.get("use_cache", True),
        "vocab_size": text_config.get("vocab_size", 151424),
        "partial_rotary_factor": 0.5,
        "tie_word_embeddings": False,
        "moe_intermediate_size": text_config.get("moe_intermediate_size", 1408),
        "n_group": text_config.get("n_group", 1),
        "n_routed_experts": text_config.get("n_routed_experts", 128),
        "n_shared_experts": text_config.get("n_shared_experts", 1),
        "norm_topk_prob": text_config.get("norm_topk_prob", True),
        "num_experts_per_tok": text_config.get("num_experts_per_tok", 8),
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [8, 12, 12]},
    }
    hf_config["text_config"] = txt_config

    if "vision_config" in model_config:
        vision_config = {
            "model_type": "glm4v_moe_vision",
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
