# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
# Copyright 2026 SGLang Team.
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
"""Load original MiMo V2 text and MiMo V2 VL checkpoints into their native model families.

The in-memory loader streams shards directly to the requested devices; it does not require a second, dequantized
checkpoint on disk. The CLI additionally saves a standard BF16 checkpoint that can use AutoModelForImageTextToText.
QKV de-interleaving follows SGLang's `load_mimo_v2_qkv_proj_weight` and `_get_ckpt_qkv_shard_sizes` layout.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .configuration_utils import PreTrainedConfig
from .utils import logging


logger = logging.get_logger(__name__)


def _load_generation_config(path):
    """Preserve legacy generation defaults while removing inactive sampling options that prevent native saves."""
    from .generation.configuration_utils import GenerationConfig

    config = GenerationConfig.from_pretrained(path, local_files_only=True)
    if config.do_sample is not True:
        defaults = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "typical_p": 1.0,
            "min_p": None,
            "top_h": None,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
        }
        removed = []
        for name, default in defaults.items():
            value = getattr(config, name)
            if value is not None and value != default:
                setattr(config, name, None)
                removed.append(name)
        if removed:
            logger.info("Removed inactive sampling options from the legacy greedy generation config: %s", removed)
    return config


def dequantize_mxfp4(weight: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Unpack low/high E2M1 nibbles and apply per-block E8M0 scales."""
    if weight.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 weights and scales must be uint8 tensors.")
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=weight.device, dtype=torch.float32
    )
    indices = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    if tuple(scales.shape) != (weight.shape[0], indices.shape[1] // block_size):
        raise ValueError("MXFP4 scale shape does not match the packed weight shape.")
    scale_values = torch.exp2(scales.float() - 127).repeat_interleave(block_size, dim=-1)
    return values[indices] * scale_values


def dequantize_fp8(
    weight: torch.Tensor, scales: torch.Tensor, block_size: tuple[int, int] = (128, 128)
) -> torch.Tensor:
    expanded = scales.float().repeat_interleave(block_size[0], 0).repeat_interleave(block_size[1], 1)
    if expanded.shape[0] < weight.shape[0] or expanded.shape[1] < weight.shape[1]:
        raise ValueError("FP8 scales do not cover the weight matrix.")
    return weight.float() * expanded[: weight.shape[0], : weight.shape[1]]


def split_interleaved_qkv(weight, config, layer_idx, tp_size, scales=None, block_size=(128, 128)):
    """Undo [Q_rank0,K_rank0,V_rank0,Q_rank1,...] packing before forming the canonical Q/K/V tensors."""
    sliding = config.layer_types[layer_idx] == "sliding_attention"
    num_heads = config.swa_num_attention_heads if sliding else config.num_attention_heads
    num_kv_heads = config.swa_num_key_value_heads if sliding else config.num_key_value_heads
    head_dim = config.swa_head_dim if sliding else config.head_dim
    value_dim = config.swa_v_head_dim if sliding else config.v_head_dim
    if (
        num_heads % tp_size
        or (num_kv_heads >= tp_size and num_kv_heads % tp_size)
        or (num_kv_heads < tp_size and tp_size % num_kv_heads)
    ):
        raise ValueError("Checkpoint TP size is incompatible with the configured query/KV heads.")
    q_size = num_heads // tp_size * head_dim
    k_size = max(1, num_kv_heads // tp_size) * head_dim
    v_size = max(1, num_kv_heads // tp_size) * value_dim
    shard_size = q_size + k_size + v_size
    if weight.shape[0] != shard_size * tp_size:
        raise ValueError(f"QKV shape {tuple(weight.shape)} does not match TP={tp_size} interleaved layout.")
    scale_shards = scales.chunk(tp_size, dim=0) if scales is not None else [None] * tp_size
    queries, keys, values = [], [], []
    for rank, (shard, shard_scales) in enumerate(zip(weight.split(shard_size), scale_shards)):
        if shard_scales is not None:
            shard = dequantize_fp8(shard, shard_scales, block_size)
        query, key, value = shard.split((q_size, k_size, v_size), dim=0)
        queries.append(query)
        # When TP exceeds the number of KV heads, consecutive ranks contain replicated KV heads.
        if num_kv_heads >= tp_size or rank % (tp_size // num_kv_heads) == 0:
            keys.append(key)
            values.append(value)
    return torch.cat(queries), torch.cat(keys), torch.cat(values)


def _native_key(name, is_multimodal=True):
    if not is_multimodal:
        return name.replace(".attention_sink_bias", ".sinks")
    for prefix in ("visual.merger.", "audio_encoder.projection."):
        name = name.replace(prefix + "mlp.0.", prefix + "linear_fc1.")
        name = name.replace(prefix + "mlp.2.", prefix + "linear_fc2.")
    if name.startswith("model."):
        return "model.language_model." + name.removeprefix("model.")
    if name.startswith(("visual.", "audio_encoder.", "speech_embeddings.")):
        return "model." + name
    return name


def _native_audio_tokenizer_key(name):
    name = name.replace(".quantizer.vq.", ".quantizer.").replace(".down_sample_layer.0.", ".down_sample_layer.")
    return "model.audio_tokenizer." + name


def _infer_checkpoint_tp_size(config, layer_idx, rows):
    """Infer TP only when KV replication makes the packed row count unambiguous."""
    sliding = config.layer_types[layer_idx] == "sliding_attention"
    num_heads = config.swa_num_attention_heads if sliding else config.num_attention_heads
    num_kv_heads = config.swa_num_key_value_heads if sliding else config.num_key_value_heads
    head_dim = config.swa_head_dim if sliding else config.head_dim
    value_dim = config.swa_v_head_dim if sliding else config.v_head_dim
    candidates = []
    for size in range(1, num_heads + 1):
        if (
            num_heads % size
            or (num_kv_heads >= size and num_kv_heads % size)
            or (num_kv_heads < size and size % num_kv_heads)
        ):
            continue
        if num_heads * head_dim + max(size, num_kv_heads) * (head_dim + value_dim) == rows:
            candidates.append(size)
    if len(candidates) != 1:
        raise ValueError(
            "Fused QKV layout is ambiguous. Supply checkpoint_tp_size or metadata.tp_size in the weight index."
        )
    return candidates[0]


@torch.no_grad()
def load_original_checkpoint(
    pretrained_model_name_or_path,
    device_map="auto",
    dtype=torch.bfloat16,
    max_memory=None,
    local_files_only=False,
    token=None,
    revision=None,
    cache_dir=None,
    config=None,
    attn_implementation=None,
    checkpoint_tp_size=None,
    generation_config=None,
):
    """Stream a released checkpoint into the native model, verifying that every required tensor was loaded.

    Args:
        pretrained_model_name_or_path (`str` or `Path`):
            Original checkpoint directory or Hub ID, for example `XiaomiMiMo/MiMo-V2.6-Flash-RL`.
        device_map (`str` or `dict`, *optional*, defaults to `"auto"`):
            Accelerate device map, or a single device such as `"cpu"` or `"cuda:0"`.
        dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            Floating-point dtype used for dequantized model parameters.
        max_memory (`dict`, *optional*):
            Per-device memory budget passed to Accelerate when inferring a device map.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether Hub downloads are restricted to the local cache.
        token (`str`, *optional*):
            Hub access token.
        revision (`str`, *optional*):
            Hub checkpoint revision.
        generation_config ([`GenerationConfig`], *optional*):
            Explicit generation defaults. When unset, load the original checkpoint's generation configuration and
            normalize inactive sampling options so the native checkpoint can be saved.
    """
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_max_memory, set_module_tensor_to_device

    from .models.auto.configuration_auto import AutoConfig
    from .models.auto.modeling_auto import AutoModelForCausalLM

    path = Path(pretrained_model_name_or_path)
    if not path.is_dir():
        path = Path(
            snapshot_download(
                str(pretrained_model_name_or_path),
                revision=revision,
                token=token,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "audio_tokenizer/*"],
            )
        )
    with open(path / "config.json", encoding="utf-8") as reader:
        original_config = json.load(reader)
    with open(path / "model.safetensors.index.json", encoding="utf-8") as reader:
        index = json.load(reader)
    is_multimodal = any(original_config.get(name) for name in ("vision_config", "audio_config"))
    tokenizer_path = path / "audio_tokenizer"
    config_dict = config.to_dict() if isinstance(config, PreTrainedConfig) else dict(original_config)
    for name in ("quantization_config", "auto_map", "architectures", "model_type", "attention_projection_layout"):
        config_dict.pop(name, None)
    if is_multimodal:
        with open(tokenizer_path / "config.json", encoding="utf-8") as reader:
            config_dict["audio_tokenizer_config"] = json.load(reader)
    config_dict["dtype"] = dtype
    config = AutoConfig.for_model("mimo_v2_vl" if is_multimodal else "mimo_v2_flash", **config_dict)
    config._name_or_path = str(pretrained_model_name_or_path)
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    if device_map == "auto":
        if max_memory is None and torch.cuda.is_available():
            max_memory = {
                device: max(0, torch.cuda.mem_get_info(device)[0] - 8 * 1024**3)
                for device in range(torch.cuda.device_count())
            }
            max_memory["cpu"] = max(0, get_max_memory()["cpu"] - 8 * 1024**3)
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=sorted(model._no_split_modules), dtype=dtype
        )
    elif isinstance(device_map, (str, torch.device, int)):
        device_map = {"": device_map}
    if any(device == "disk" for device in device_map.values()):
        raise ValueError(
            "Streaming original checkpoints requires a CPU/GPU device map with enough memory; disk offload is not supported."
        )
    staging_device = next(
        (
            torch.device(f"cuda:{device}" if isinstance(device, int) else device)
            for device in device_map.values()
            if isinstance(device, int) or str(device).startswith("cuda")
        ),
        None,
    )

    def target_device(name):
        candidates = [
            prefix for prefix in device_map if name == prefix or name.startswith(prefix + ".") or prefix == ""
        ]
        if not candidates:
            raise ValueError(f"No device specified for {name}.")
        device = device_map[max(candidates, key=len)]
        return torch.device(f"cuda:{device}" if isinstance(device, int) else device)

    expected_shapes = {name: tensor.shape for name, tensor in model.state_dict().items()}
    expected = set(expected_shapes)
    loaded = set()
    expert_parts = defaultdict(set)

    def assign(name, value):
        if name not in expected:
            raise ValueError(f"Checkpoint tensor has no native destination: {name}")
        keep_float32 = ".quantizer." in name or name.endswith("e_score_correction_bias")
        target_dtype = torch.float32 if keep_float32 else dtype
        if not value.dtype.is_floating_point:
            target_dtype = value.dtype
        set_module_tensor_to_device(model, name, target_device(name), value=value, dtype=target_dtype)
        loaded.add(name)

    def assign_expert(name, value, expert_index, projection):
        if not expert_parts[name]:
            assign(name, torch.empty(expected_shapes[name], device=target_device(name), dtype=dtype))
            loaded.remove(name)
        parameter = model.get_parameter(name)
        value = value.to(dtype=parameter.dtype)
        if projection == "down_proj":
            parameter[expert_index].copy_(value)
        else:
            size = config.moe_intermediate_size
            start = 0 if projection == "gate_proj" else size
            parameter[expert_index, start : start + size].copy_(value)
        expert_parts[name].add((expert_index, projection))
        expected_parts = expected_shapes[name][0] * (1 if projection == "down_proj" else 2)
        if len(expert_parts[name]) == expected_parts:
            loaded.add(name)

    weight_map = index["weight_map"]
    tp_size = index.get("metadata", {}).get("tp_size", checkpoint_tp_size)
    qkv_names = [name for name in weight_map if re.fullmatch(r"model\.layers\.\d+\.self_attn\.qkv_proj\.weight", name)]
    if qkv_names and tp_size is None:
        qkv_name = qkv_names[0]
        with safe_open(path / weight_map[qkv_name], framework="pt", device="cpu") as shard:
            rows = shard.get_slice(qkv_name).get_shape()[0]
        layer_idx = int(qkv_name.split(".")[2])
        tp_size = _infer_checkpoint_tp_size(config.get_text_config(), layer_idx, rows)
    quantization_config = original_config.get("quantization_config") or {}
    block_size = tuple(quantization_config.get("weight_block_size", (128, 128)))
    grouped_keys = defaultdict(list)
    for name, filename in weight_map.items():
        grouped_keys[filename].append(name)

    def read_tensor(name, current_shard, current_filename, device):
        filename = weight_map[name]
        if filename == current_filename:
            return current_shard.get_tensor(name).to(device)
        # HF sharding may put a weight and its quantization scale in different files.
        with safe_open(path / filename, framework="pt", device="cpu") as source:
            return source.get_tensor(name).to(device)

    for filename, names in sorted(grouped_keys.items()):
        with safe_open(path / filename, framework="pt", device="cpu") as shard:
            for name in names:
                if name.endswith((".weight_scale_inv", ".weight_scale")) or name.startswith(
                    ("model.mtp.", "model.decoder.")
                ):
                    continue
                native_name = _native_key(name, is_multimodal=is_multimodal)
                qkv_match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)", name)
                expert_match = (
                    None
                    if is_multimodal
                    else re.fullmatch(
                        r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name
                    )
                )
                if qkv_match:
                    layer_idx = int(qkv_match[1])
                    suffix = qkv_match[2]
                    prefix = f"model.{'language_model.' if is_multimodal else ''}layers.{layer_idx}.self_attn."
                    device = target_device(prefix + "q_proj." + suffix)
                elif expert_match:
                    projection = expert_match[3]
                    native_name = expert_match[1] + (".down_proj" if projection == "down_proj" else ".gate_up_proj")
                    device = target_device(native_name)
                elif native_name in expected:
                    device = target_device(native_name)
                elif name.endswith("rotary_emb.inv_freq"):
                    continue
                else:
                    raise ValueError(f"Unexpected tensor in original checkpoint: {name}")
                value = shard.get_tensor(name)
                # Decode CPU-offloaded quantized weights on an already-used GPU, then transfer only the final dtype.
                compute_device = device
                if (
                    device.type == "cpu"
                    and staging_device is not None
                    and value.dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)
                ):
                    compute_device = staging_device
                value = value.to(compute_device)
                scale_name = name.replace(".weight", ".weight_scale_inv")
                scales = (
                    read_tensor(scale_name, shard, filename, compute_device)
                    if name.endswith(".weight") and scale_name in weight_map
                    else None
                )
                if qkv_match:
                    projections = split_interleaved_qkv(
                        value, config.get_text_config(), layer_idx, tp_size, scales, block_size
                    )
                    for projection, tensor in zip(("q_proj", "k_proj", "v_proj"), projections):
                        assign(prefix + projection + "." + suffix, tensor)
                else:
                    if value.dtype == torch.uint8:
                        scale_name = name.replace(".weight", ".weight_scale")
                        value = dequantize_mxfp4(
                            value,
                            read_tensor(scale_name, shard, filename, compute_device),
                            quantization_config.get("mxfp4_block_size", 32),
                        )
                    elif scales is not None:
                        value = dequantize_fp8(value, scales, block_size)
                    if expert_match:
                        assign_expert(native_name, value, int(expert_match[2]), projection)
                    else:
                        assign(native_name, value)
        logger.info("Loaded %s (%d/%d tensors)", filename, len(loaded), len(expected))

    if is_multimodal:
        with safe_open(tokenizer_path / "model.safetensors", framework="pt", device="cpu") as shard:
            for name in shard.keys():
                native_name = _native_audio_tokenizer_key(name)
                if native_name in expected:
                    assign(native_name, shard.get_tensor(name))
    missing = expected - loaded
    if missing:
        raise ValueError(f"Original checkpoint is missing required tensors: {sorted(missing)}")
    model = dispatch_model(model, device_map=device_map, skip_keys=["past_key_values"])
    if generation_config is not None:
        model.generation_config = generation_config
    elif (path / "generation_config.json").is_file():
        model.generation_config = _load_generation_config(path)
    model.eval()
    model.hf_device_map = device_map
    return model


def load_original_checkpoint_if_needed(model_class, pretrained_model_name_or_path, **kwargs):
    """Use the legacy weight loader only for repositories still declaring the ambiguous `mimo_v2` format."""
    if pretrained_model_name_or_path is None:
        return None
    hub_kwargs = {
        key: kwargs[key]
        for key in ("cache_dir", "local_files_only", "token", "revision", "subfolder")
        if key in kwargs
    }
    config_dict, _ = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **hub_kwargs)
    if config_dict.get("model_type") != "mimo_v2":
        return None
    from .models.auto.configuration_auto import _get_mimo_v2_model_type

    model_type = _get_mimo_v2_model_type(config_dict)
    if model_class.config_class.model_type != model_type:
        raise ValueError(
            f"Checkpoint configuration selects {model_type}, not {model_class.config_class.model_type}. Use an AutoModel class."
        )
    if kwargs.get("quantization_config") is not None:
        raise ValueError(
            "Original MiMo V2 weights are dequantized during loading. Save a native checkpoint before applying another quantizer."
        )
    output_loading_info = kwargs.pop("output_loading_info", False)
    if hub_kwargs.pop("subfolder", None):
        raise ValueError("Original MiMo V2 checkpoints must be loaded from their root directory.")
    dtype = kwargs.get("dtype") or kwargs.get("torch_dtype") or "auto"
    if dtype == "auto":
        dtype = getattr(torch, config_dict.get("dtype") or config_dict.get("torch_dtype") or "bfloat16")
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    model = load_original_checkpoint(
        pretrained_model_name_or_path,
        device_map=kwargs.get("device_map", "auto"),
        dtype=dtype,
        max_memory=kwargs.get("max_memory"),
        config=kwargs.get("config"),
        attn_implementation=kwargs.get("attn_implementation"),
        checkpoint_tp_size=kwargs.get("checkpoint_tp_size"),
        generation_config=kwargs.get("generation_config"),
        **hub_kwargs,
    )
    if not isinstance(model, model_class):
        if isinstance(model.model, model_class):
            base_model = model.model
            # Accelerate dispatches the complete generation model. Keep its placement metadata when returning
            # only the backbone, using the backbone's parameter names rather than the generation-head prefix.
            device_map = model.hf_device_map
            prefix = model.base_model_prefix
            base_device_map = {}
            if prefix in device_map:
                base_device_map[""] = device_map[prefix]
            elif "" in device_map:
                base_device_map[""] = device_map[""]
            base_device_map.update(
                {
                    name.removeprefix(prefix + "."): device
                    for name, device in device_map.items()
                    if name.startswith(prefix + ".")
                }
            )
            base_model.hf_device_map = base_device_map
            model = base_model
        else:
            raise ValueError(
                f"Checkpoint configuration selects {type(model).__name__}, not {model_class.__name__}. Use an AutoModel class."
            )
    if output_loading_info:
        return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    return model
