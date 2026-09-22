# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Pre-trace adapters for ExecuTorch's off-graph KV cache."""

import contextlib
import threading

import torch

from ..cache_utils import DynamicCache, StaticCache, get_layer_types_and_kwargs


# HF's attention registry is process-global.
_EXPORT_LOCK = threading.RLock()
_ATTENTION_NAME = "executorch_native"


def _validate_model(model):
    from ..exporters.utils import is_multimodal

    try:
        from executorch.extension.llm.cache.update_and_attend import update_and_attend  # noqa: F401
        from executorch.extension.llm.export.model_metadata import write_cache_geometry  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "executorch_native requires an ExecuTorch build providing "
            "extension.llm.cache.update_and_attend and extension.llm.export.model_metadata.write_cache_geometry."
        ) from error

    config = model.config.get_text_config(decoder=True)
    if model.config.is_encoder_decoder or is_multimodal(model) or not model._supports_default_dynamic_cache():
        raise ValueError(
            "executorch_native currently supports decoder-only text models with standard attention caches."
        )
    layer_types, _ = get_layer_types_and_kwargs(config)
    if set(layer_types) != {"full_attention"} or getattr(config, "num_kv_shared_layers", 0):
        raise ValueError("executorch_native currently supports full attention without shared-KV layers.")
    if not model._supports_attention_backend:
        raise ValueError("executorch_native requires the standard attention interface.")
    if model.training:
        raise ValueError("executorch_native requires model.eval().")


def validate_native_generation(model, sample_inputs, generation_config):
    """Check the supported capture scope without modifying HF generation or its cache selection."""
    _validate_model(model)
    if generation_config.num_beams != 1 or generation_config.num_return_sequences != 1:
        raise ValueError("executorch_native currently supports one sequence without beam expansion.")
    if generation_config.get_generation_mode() not in ("greedy_search", "sample") or generation_config.is_assistant:
        raise ValueError(
            "executorch_native supports only greedy decoding and sampling without speculative generation."
        )
    if generation_config.output_attentions:
        raise ValueError("executorch_native does not return attention weights.")
    if generation_config.use_cache is False:
        raise ValueError("executorch_native requires use_cache=True during generation capture.")
    if sample_inputs.get("past_key_values") is not None or sample_inputs.get("assistant_model") is not None:
        raise ValueError("executorch_native does not accept an existing cache or assistant model during capture.")
    mask = sample_inputs.get("attention_mask")
    if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim != 2 or not bool((mask == 1).all())):
        raise ValueError("executorch_native does not support padding or custom attention masks.")
    tokens = sample_inputs.get("input_ids")
    if tokens is None:
        tokens = sample_inputs.get("inputs_embeds")
    if tokens is None or tokens.shape[0] != 1:
        raise ValueError("executorch_native requires a single input sequence.")
    positions = sample_inputs.get("position_ids")
    if positions is not None:
        expected = torch.arange(tokens.shape[1], device=positions.device).unsqueeze(0)
        if not torch.equal(positions, expected):
            raise ValueError("executorch_native capture must start at position zero with contiguous positions.")


def native_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    position_ids=None,
    scaling=None,
    dropout=0.0,
    softcap=None,
    head_mask=None,
    **kwargs,
):
    if dropout or softcap is not None or head_mask is not None:
        raise ValueError("executorch_native does not support dropout, softcap, or head masks.")
    if attention_mask is not None or not getattr(module, "is_causal", False):
        raise ValueError("executorch_native supports only unpadded causal self-attention without custom masks.")
    if position_ids is None or position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise ValueError("executorch_native requires explicit single-sequence position_ids.")
    if scaling is None or getattr(module, "layer_idx", None) is None:
        raise ValueError("executorch_native requires an attention scale and layer index.")
    if query.shape[0] != 1 or key.shape != value.shape or key.shape[-1] != query.shape[-1]:
        raise ValueError("executorch_native requires batch size one and equal query/key/value head dimensions.")
    if key.shape[-2] != query.shape[-2] or position_ids.shape[-1] != query.shape[-2]:
        raise ValueError("executorch_native expects only this step's K/V and one position per query token.")
    if getattr(module, "is_kv_shared_layer", False):
        raise ValueError("executorch_native does not yet support shared-KV layers.")
    config = module.config
    n_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if key.shape[1] != n_heads or key.shape[-1] != head_dim:
        raise ValueError("executorch_native currently requires uniform KV geometry matching the model config.")

    output = torch.ops.kvcache.update_and_attend(
        query, key, value, position_ids[0].reshape(-1, 1), module.layer_idx, scaling, query.dtype
    )
    return output.transpose(1, 2).contiguous(), None


def _validate_captured_mask(mask, positions, kv_length):
    """Accept only masks equivalent to the native operator's full causal attention."""
    if isinstance(mask, dict) and set(mask) == {"full_attention"}:
        mask = mask["full_attention"]
    if mask is None:
        return
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 2 and mask.shape == (1, kv_length) and bool((mask == 1).all()):
            return
        if mask.ndim == 4 and mask.shape[:3] == (1, 1, positions.shape[1]) and mask.shape[3] >= kv_length:
            allowed = torch.arange(mask.shape[3], device=positions.device) <= positions.unsqueeze(-1)
            allowed = allowed.unsqueeze(1)
            if mask.dtype == torch.bool and torch.equal(mask, allowed):
                return
            if mask.is_floating_point():
                blocked = (mask == torch.finfo(mask.dtype).min) | torch.isneginf(mask)
                if bool(torch.where(allowed, mask == 0, blocked).all()):
                    return
    raise ValueError("executorch_native does not support padding or custom attention masks.")


def prepare_native_inputs(inputs):
    """Remove the captured HF cache before tracing, keeping positions as explicit tensor inputs."""
    inputs = dict(inputs)
    cache = inputs.pop("past_key_values", None)
    if cache is not None and type(cache) not in (DynamicCache, StaticCache):
        raise ValueError("executorch_native export requires an ordinary DynamicCache or StaticCache capture.")
    tokens = inputs.get("input_ids")
    if tokens is None:
        tokens = inputs.get("inputs_embeds")
    if tokens is None or tokens.shape[0] != 1:
        raise ValueError("executorch_native requires a single input sequence.")
    past_length = int(cache.get_seq_length()) if cache is not None else 0
    expected_positions = (torch.arange(tokens.shape[1], device=tokens.device) + past_length).unsqueeze(0)
    positions = inputs.get("position_ids")
    if positions is None:
        positions = expected_positions
    if positions.dtype not in (torch.int32, torch.int64) or not torch.equal(positions, expected_positions):
        raise ValueError("executorch_native requires contiguous position_ids matching the captured cache length.")
    inputs["position_ids"] = positions
    _validate_captured_mask(inputs.pop("attention_mask", None), positions, past_length + tokens.shape[1])
    if inputs.get("output_attentions"):
        raise ValueError("executorch_native does not return attention weights.")
    inputs["use_cache"] = False
    return inputs


@contextlib.contextmanager
def native_cache_export(model, sample_inputs):
    """Install native attention only for tracing/lowering; generation capture stays unchanged."""
    from ..exporters.utils import patch_attribute
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    _validate_model(model)
    inputs = prepare_native_inputs(sample_inputs)
    with _EXPORT_LOCK, contextlib.ExitStack() as stack:
        stack.enter_context(
            patch_attribute(
                ALL_ATTENTION_FUNCTIONS,
                "_global_mapping",
                lambda original: {**original, _ATTENTION_NAME: native_attention_forward},
            )
        )
        configs = {id(m.config): m.config for m in model.modules() if hasattr(m, "config")}
        for subconfig in configs.values():
            stack.enter_context(patch_attribute(subconfig, "_attn_implementation", lambda original: _ATTENTION_NAME))
        yield inputs


def native_cache_geometry(exported_program):
    """Read actual cache slots from the graph and publish geometry, never allocation limits."""
    from executorch.extension.llm.export.model_metadata import write_cache_geometry

    geometry = {}
    for node in exported_program.graph.nodes:
        if node.target != torch.ops.kvcache.update_and_attend.default:
            continue
        key = node.args[1].meta["val"]
        layer_id = node.args[4]
        if layer_id in geometry:
            raise ValueError("executorch_native does not support multiple writes to the same cache layer per forward.")
        geometry[layer_id] = (int(key.shape[1]), int(key.shape[-1]))
    if not geometry or sorted(geometry) != list(range(len(geometry))):
        raise ValueError("executorch_native requires contiguous cache layer IDs starting at zero.")
    return write_cache_geometry(
        [geometry[i][0] for i in range(len(geometry))],
        [geometry[i][1] for i in range(len(geometry))],
        [0] * len(geometry),
    )
