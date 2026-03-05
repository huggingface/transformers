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
import importlib
import inspect
from typing import TYPE_CHECKING, Any

from ..cache_utils import Cache, DynamicCache
from ..utils.import_utils import is_torch_available
from ..utils.logging import get_logger


logger = get_logger(__name__)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


def _iter_subclasses(cls: type):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _iter_subclasses(subclass)


def _class_to_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _path_to_class(path: str) -> type:
    module_name, class_qualname = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    class_obj = module
    for item in class_qualname.split("."):
        class_obj = getattr(class_obj, item)
    return class_obj


class DynamoSerializer:
    _SPECIAL_KEY = "__special_key__"
    _SPECIAL_VALUE = "__special_value__"

    @classmethod
    def _is_leaf_type(cls, obj: Any) -> bool:
        return (
            (isinstance(obj, (int, float, bool, str, torch.Tensor, torch.Size)))
            or obj.__class__.__module__ == "torch.export.dynamic_shapes"
            or obj is None
        )

    @classmethod
    def _is_special_type(cls, obj: Any) -> bool:
        return isinstance(obj, (torch.dtype, torch.device, torch.layout, type, frozenset))

    @classmethod
    def _get_special_key(cls, obj: Any) -> str:
        if isinstance(obj, torch.dtype):
            return "dtype"
        if isinstance(obj, torch.device):
            return "device"
        if isinstance(obj, torch.layout):
            return "layout"
        if isinstance(obj, type):
            return "class_ref"
        if isinstance(obj, frozenset):
            return "frozenset"
        raise TypeError(f"Object of type '{type(obj)}' is not a recognized special type.")

    @classmethod
    def _encode_special_value(cls, obj: Any) -> Any:
        if isinstance(obj, torch.dtype):
            return str(obj).removeprefix("torch.")
        if isinstance(obj, torch.layout):
            return str(obj).removeprefix("torch.")
        if isinstance(obj, torch.device):
            return f"{obj.type}:{obj.index}" if obj.index is not None else obj.type
        if isinstance(obj, type):
            return _class_to_path(obj)
        if isinstance(obj, frozenset):
            return [cls.serialize(item) for item in obj]
        raise TypeError(f"Object of type '{type(obj)}' is not a recognized special type.")

    @classmethod
    def _decode_special_value(cls, kind: str, value: Any) -> Any:
        if kind == "dtype":
            return getattr(torch, value)
        if kind == "layout":
            return getattr(torch, value)
        if kind == "device":
            return torch.device(value)
        if kind == "class_ref":
            return _path_to_class(value)
        if kind == "frozenset":
            return frozenset(cls.deserialize(item) for item in value)
        raise TypeError(f"Unknown special kind encountered during decoding: {kind}")

    @classmethod
    def _serialize_special(cls, obj: Any) -> dict[str, str]:
        return {cls._SPECIAL_KEY: cls._get_special_key(obj), cls._SPECIAL_VALUE: cls._encode_special_value(obj)}

    @classmethod
    def _deserialize_special(cls, obj: dict[str, Any]) -> Any:
        return cls._decode_special_value(obj[cls._SPECIAL_KEY], obj[cls._SPECIAL_VALUE])

    @classmethod
    def serialize(cls, obj: Any) -> Any:
        if cls._is_leaf_type(obj):
            return obj
        elif cls._is_special_type(obj):
            return cls._serialize_special(obj)
        elif isinstance(obj, (list, tuple, set)):
            return [cls.serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: cls.serialize(value) for key, value in obj.items()}

        state = getattr(obj, "__dict__", None)
        if state is None:
            logger.warning(
                f"Object of type '{type(obj)}' does not have a __dict__ attribute and is not a recognized leaf or special type. "
                "Serializing it as an empty object. This may lead to issues during deserialization."
            )
            state = {}

        return {
            "__class__": _class_to_path(type(obj)),
            "__state__": cls.serialize(dict(state)),
        }

    @classmethod
    def deserialize(cls, obj: Any) -> Any:
        if cls._is_leaf_type(obj):
            return obj
        elif isinstance(obj, list):
            return [cls.deserialize(item) for item in obj]
        elif isinstance(obj, dict):
            special_key = obj.get(cls._SPECIAL_KEY)
            if special_key is not None:
                return cls._deserialize_special(obj)

            class_path = obj.get("__class__")
            if class_path is None:
                return {key: cls.deserialize(value) for key, value in obj.items()}

            class_obj = _path_to_class(class_path)
            state = cls.deserialize(obj["__state__"])
            instance = class_obj.__new__(class_obj)
            instance.__dict__.update(state)
            return instance

        raise TypeError(f"Malformed serialized object encountered: {type(obj)}")


def _flatten_for_export(cache: Cache):
    return torch.utils._pytree._dict_flatten(DynamoSerializer.serialize(cache))


def _flatten_for_export_with_keys(cache: Cache):
    return torch.utils._pytree._dict_flatten_with_keys(DynamoSerializer.serialize(cache))


def _unflatten_for_export(values, context: torch.utils._pytree.Context):
    return DynamoSerializer.deserialize(torch.utils._pytree._dict_unflatten(values, context))


def _register_class_for_export(cls: type):
    try:
        torch.utils._pytree.register_pytree_node(
            cls,
            _flatten_for_export,
            _unflatten_for_export,
            serialized_type_name=f"{cls.__module__}.{cls.__name__}",
            flatten_with_keys_fn=_flatten_for_export_with_keys,
        )
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def register_cache_subclasses_as_pytree_nodes():
    for cache_type in _iter_subclasses(Cache):
        _register_class_for_export(cache_type)


def register_custom_model_cache_as_pytree_nodes(model: "PreTrainedModel"):
    for _, obj in inspect.getmembers(inspect.getmodule(model)):
        if (
            inspect.isclass(obj)
            and obj.__module__ == model.__class__.__module__
            and obj.__name__.endswith("Cache")
            and not issubclass(obj, Cache)
        ):
            _register_class_for_export(obj)


def _is_tensor_free(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return False
    elif isinstance(obj, (list, tuple, set)):
        return all(_is_tensor_free(item) for item in obj)
    elif isinstance(obj, dict):
        return all(_is_tensor_free(value) for value in obj.values())
    elif isinstance(obj, Cache) or obj.__class__.__name__.endswith("Cache"):
        return _is_tensor_free(DynamoSerializer.serialize(obj))
    else:
        return True


def get_leaf_tensors(obj: Any) -> dict[str, torch.Tensor]:
    """
    Recursively retrieves all leaf tensors from a potentialy nested structure.
    Args:
        obj (`Any`):
            The object from which to retrieve leaf tensors.
    Returns:
        `dict[str, torch.Tensor]`: A dictionary mapping names to leaf tensors.
    """
    if _is_tensor_free(obj):
        return {}
    elif isinstance(obj, torch.Tensor):
        return {"": obj}
    elif isinstance(obj, (list, tuple, set)):
        return get_leaf_tensors(dict(enumerate(obj)))
    elif isinstance(obj, dict):
        leaf_tensors = {}
        for key, value in obj.items():
            for sub_key, tensor in get_leaf_tensors(value).items():
                full_key = f"{key}.{sub_key}" if sub_key else key
                leaf_tensors[full_key] = tensor
        return leaf_tensors
    elif isinstance(obj, Cache) or obj.__class__.__name__.endswith("Cache"):
        return get_leaf_tensors(DynamoSerializer.serialize(obj))
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


def prepare_for_export(
    model: "PreTrainedModel",
    inputs: dict[str, torch.Tensor | Cache],
    outputs: dict[str, torch.Tensor | Cache] | None = None,
) -> tuple["PreTrainedModel", dict[str, torch.Tensor | Cache]]:
    # filter out None inputs
    inputs = {k: v for k, v in inputs.items() if v is not None}

    for input_name in ("labels", "future_values"):
        if input_name in inputs:
            logger.info(f"Found an input '{input_name}' which is not supported for export. Popping it from inputs.")
            inputs.pop(input_name)

    # handle output flags passed in inputs
    for output_flag in ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss"):
        if output_flag in inputs:
            logger.info(f"Found an output flag '{output_flag}' in inputs. Setting model.config.{output_flag} instead.")
            setattr(model.config, output_flag, inputs.pop(output_flag))

    # set experts implementation to batched_mm for export
    if model._can_set_experts_implementation():
        model.set_experts_implementation("batched_mm")

    # set attention implementation to sdpa for export
    if model._can_set_attn_implementation() and model.config.model_type != "videomae":
        try:
            model.set_attn_implementation("sdpa")
        except Exception as e:
            print(
                f"Could not set attention implementation to sdpa for {model} of type {model.config.model_type} : {e}"
            )

    for module in model.modules():
        if hasattr(module, "config"):
            # disable returning loss for every submodel
            if hasattr(module.config, "return_loss"):
                module.config.return_loss = False
            # disable mamba kernels for every submodel (mamba, jamba)
            if hasattr(module.config, "use_mamba_kernels"):
                module.config.use_mamba_kernels = False
        # disable classifier cast for nllb-moe
        if hasattr(module, "_cast_classifier"):
            module._cast_classifier = lambda *args, **kwargs: None
        # disable mamba mask update for ssms
        if hasattr(module, "_update_mamba_mask"):
            module._update_mamba_mask = lambda attention_mask, *args, **kwargs: attention_mask
        if hasattr(module, "_update_linear_attn_mask"):
            module._update_linear_attn_mask = lambda attention_mask, *args, **kwargs: attention_mask

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

    return model, inputs, outputs


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
