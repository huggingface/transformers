# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
import copy
import inspect
from typing import TYPE_CHECKING, Any

from ..cache_utils import Cache, DynamicCache, DynamicLayer, EncoderDecoderCache
from ..utils.import_utils import is_torch_available
from ..utils.logging import get_logger


logger = get_logger(__name__)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


def _is_pure_python_object(obj: Any) -> bool:
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return True
    elif isinstance(obj, DynamicCache):
        return _is_pure_python_object(_dict_from_dynamic_cache(obj))
    elif isinstance(obj, EncoderDecoderCache):
        return _is_pure_python_object(_dict_from_encoder_decoder_cache(obj))
    elif isinstance(obj, (list, tuple, set)):
        return all(_is_pure_python_object(o) for o in obj)
    elif isinstance(obj, dict):
        return all(_is_pure_python_object(v) for v in obj.values())
    else:
        return False


def get_leaf_tensors(obj: Any) -> dict[str, torch.Tensor]:
    """
    Recursively retrieves all leaf tensors from a potentialy nested structure.
    Args:
        obj (`Any`):
            The object from which to retrieve leaf tensors.
    Returns:
        `dict[str, torch.Tensor]`: A dictionary mapping names to leaf tensors.
    """
    if _is_pure_python_object(obj):
        return {}
    elif isinstance(obj, torch.Tensor):
        return {"": obj}
    elif isinstance(obj, (list, tuple, set)):
        return get_leaf_tensors(dict(enumerate(obj)))
    elif isinstance(obj, DynamicCache):
        return get_leaf_tensors(_dict_from_dynamic_cache(obj))
    elif isinstance(obj, EncoderDecoderCache):
        return get_leaf_tensors(_dict_from_encoder_decoder_cache(obj))
    elif isinstance(obj, dict):
        leaf_tensors = {}
        for key, value in obj.items():
            for sub_key, tensor in get_leaf_tensors(value).items():
                full_key = f"{key}.{sub_key}" if sub_key else key
                leaf_tensors[full_key] = tensor
        return leaf_tensors
    else:
        raise ValueError(f"Unexpected object type: {type(obj)}")


def get_inputs_outputs_names(inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Utility function to get the names of input and output tensors, adding prefixes to avoid name collisions.
    Args:
        inputs (`dict[str, Any]`):
            A dictionary of model inputs.
        outputs (`dict[str, Any]`):
            A dictionary of model outputs.
    Returns:
        `tuple[list[str], list[str]]`: A tuple containing two lists: input names and output names.
    """
    inputs_names = list(get_leaf_tensors(inputs).keys())
    outputs_names = list(get_leaf_tensors(outputs).keys())
    for name in set(inputs_names).intersection(set(outputs_names)):
        inputs_names[inputs_names.index(name)] = f"input.{name}"
        outputs_names[outputs_names.index(name)] = f"output.{name}"
    return inputs_names, outputs_names


# Pytree registration utilities
def _dict_from_dynamic_cache(cache: DynamicCache):
    return {
        "keys": [layer.keys for layer in cache.layers if layer.keys is not None],
        "values": [layer.values for layer in cache.layers if layer.values is not None],
    }


def _dict_from_encoder_decoder_cache(cache: EncoderDecoderCache):
    return {
        "self_attention_cache": _dict_from_dynamic_cache(cache.self_attention_cache),
        "cross_attention_cache": _dict_from_dynamic_cache(cache.cross_attention_cache),
    }


def _dynamic_cache_from_dict(dictionary):
    cache = DynamicCache()
    key_list = dictionary["keys"]
    value_list = dictionary["values"]
    assert len(key_list) == len(value_list), "Mismatched keys and values lengths in DynamicCache."
    for idx in range(len(key_list)):
        cache_layer = DynamicLayer()
        cache_layer.keys = key_list[idx]
        cache_layer.values = value_list[idx]
        cache_layer.is_initialized = True
        cache.layers.append(cache_layer)
    return cache


def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = _dynamic_cache_from_dict(dictionary)
    return cache


def _unflatten_encoder_decoder_cache(values, context: torch.utils._pytree.Context):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    self_attention_cache = _dynamic_cache_from_dict(dictionary["self_attention_cache"])
    cross_attention_cache = _dynamic_cache_from_dict(dictionary["cross_attention_cache"])
    enocder_decoder_cache = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
    return enocder_decoder_cache


def register_dynamic_cache_for_export():
    try:
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            lambda dynamic_cache: torch.utils._pytree._dict_flatten(_dict_from_dynamic_cache(dynamic_cache)),
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
                _dict_from_dynamic_cache(dynamic_cache)
            ),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def register_encoder_decoder_cache_for_export():
    try:
        torch.utils._pytree.register_pytree_node(
            EncoderDecoderCache,
            lambda cache: torch.utils._pytree._dict_flatten(_dict_from_encoder_decoder_cache(cache)),
            _unflatten_encoder_decoder_cache,
            serialized_type_name=f"{EncoderDecoderCache.__module__}.{EncoderDecoderCache.__name__}",
            flatten_with_keys_fn=lambda cache: torch.utils._pytree._dict_flatten_with_keys(
                _dict_from_encoder_decoder_cache(cache)
            ),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


# Inputs utilities
MODEL_TYPES_WITH_UNSUPPORTED_CACHE_CLASS: set[str] = {
    "falcon_mamba",
    "jamba",
    "lfm2",
    "lfm2_moe",
    "lfm2_vl",
    "mamba",
    "mamba2",
    "minimax",
    "qwen3_next",
    "reformer",
    "xlstm",
    "zamba2",
}


def prepare_for_export(
    model: "PreTrainedModel",
    inputs: dict[str, torch.Tensor | Cache],
    outputs: dict[str, torch.Tensor | Cache] | None = None,
) -> tuple["PreTrainedModel", dict[str, torch.Tensor | Cache]]:
    # filter out None inputs
    inputs = {k: v for k, v in inputs.items() if v is not None}

    # handle output flags passed in inputs
    for output_flag in ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss"):
        if output_flag in inputs:
            logger.info(f"Found an output flag '{output_flag}' in inputs. Setting model.config.{output_flag} instead.")
            setattr(model.config, output_flag, inputs.pop(output_flag))

    # handle models with unsupported cache classes
    if model.config.model_type in MODEL_TYPES_WITH_UNSUPPORTED_CACHE_CLASS:
        for submodule in model.modules():
            if hasattr(submodule, "config") and getattr(submodule.config, "use_cache", False):
                setattr(submodule.config, "use_cache", False)

    if (
        getattr(model.config, "use_cache", False)
        and not getattr(model.config, "is_encoder_decoder", False)
        and "past_key_values" in inspect.signature(model.forward).parameters
        and "past_key_values" not in inputs
    ):
        if outputs is None:
            with torch.no_grad():
                outputs = model(**copy.deepcopy(inputs))

        if hasattr(outputs, "past_key_values") and isinstance(outputs.past_key_values, DynamicCache):
            inputs["past_key_values"] = outputs.past_key_values
            if model.config.model_type not in {"qwen2_vl", "qwen2_5_vl"}:
                dtype = inputs["input_ids"].dtype
                device = inputs["input_ids"].device
                batch_size, seq_len = inputs["input_ids"].shape[:2]
                pkv_len = inputs["past_key_values"].get_seq_length()
                inputs["attention_mask"] = torch.ones((batch_size, seq_len + pkv_len), device=device, dtype=dtype)

    return model, inputs


# Dynamic shapes utilities
def _auto_dynamic_shape(tensor: torch.Tensor) -> dict[int, torch.export.Dim]:
    """
    Utility function to generate a dynamic shape with all dimensions set to Dim.AUTO for a given tensor.
    Args:
        tensor (`torch.Tensor`):
            The tensor for which to generate the dynamic shape.
    Returns:
        `dict[int, torch.export.Dim]`: A dictionary mapping dimension indices to Dim.AUTO.
    """
    from torch.export import Dim

    return dict.fromkeys(range(len(tensor.shape)), Dim.AUTO)


def get_auto_dynamic_shapes(inputs: dict[str, torch.Tensor | Cache]) -> dict[str, dict[int, torch.export.Dim]]:
    """
    Utility function to automatically generate dynamic shapes for all tensor inputs and DynamicCache inputs.
    Args:
        inputs (`dict[str, torch.Tensor | Cache]`):
            A dictionary of model inputs.
    Returns:
        `dict[str, dict[int, torch.export.Dim]]`: A dictionary mapping input names to their dynamic shapes.
    """
    dynamic_shapes = {}
    for name, input in inputs.items():
        if isinstance(input, DynamicCache):
            dynamic_shapes[name] = [
                [_auto_dynamic_shape(layer.keys) for layer in input.layers],
                [_auto_dynamic_shape(layer.values) for layer in input.layers],
            ]
        elif isinstance(input, torch.Tensor):
            dynamic_shapes[name] = _auto_dynamic_shape(input)
        elif isinstance(input, (int, float, bool, str)):
            dynamic_shapes[name] = None
        elif isinstance(input, dict):
            dynamic_shapes[name] = get_auto_dynamic_shapes(input)
        elif isinstance(input, (list, tuple, set)):
            dynamic_shapes[name] = type(input)(get_auto_dynamic_shapes(dict(enumerate(input))).values())
        else:
            raise ValueError(
                f"Input '{name}' is of unsupported type '{type(input)}'. "
                "Only torch.Tensor, DynamicCache, int, float, bool, str, dict, list, tuple, and set are supported."
            )

    return dynamic_shapes
