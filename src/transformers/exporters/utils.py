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
from typing import TYPE_CHECKING, Any, NamedTuple

from ..cache_utils import Cache
from ..generation.utils import ALL_CACHE_NAMES
from ..utils.import_utils import is_torch_available
from ..utils.logging import get_logger


logger = get_logger(__name__)


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


class _SerializedObject(NamedTuple):
    class_type: type
    state: Any


class _FlattenedContext(NamedTuple):
    template: Any
    tensor_paths: list[Any]


def _iter_subclasses(cls: type):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _iter_subclasses(subclass)


def _class_to_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class PytreeRegistry:
    _registered_classes: set[type] = set()

    @classmethod
    def is_leaf_type(cls, obj: Any) -> bool:
        return (
            isinstance(
                obj,
                (int, float, bool, str, type, torch.Tensor, torch.Size, torch.dtype, torch.device, torch.layout),
            )
            or obj is None
        )

    @classmethod
    def serialize(cls, obj: Any) -> Any:
        if cls.is_leaf_type(obj):
            return obj
        elif isinstance(obj, (list, tuple, set)):
            return type(obj)(cls.serialize(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: cls.serialize(value) for key, value in obj.items()}
        elif hasattr(obj, "__dict__"):
            state = getattr(obj, "__dict__", None)
            return _SerializedObject(class_type=type(obj), state=cls.serialize(dict(state)))
        else:
            raise TypeError(f"Object of type '{type(obj)}' is not serializable.")

    @classmethod
    def deserialize(cls, obj: Any) -> Any:
        if cls.is_leaf_type(obj):
            return obj
        elif isinstance(obj, _SerializedObject):
            state = cls.deserialize(obj.state)
            instance = obj.class_type.__new__(obj.class_type)
            instance.__dict__.update(state)
            return instance
        elif isinstance(obj, (list, tuple, set)):
            return type(obj)(cls.deserialize(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: cls.deserialize(value) for key, value in obj.items()}

        raise TypeError(f"Malformed serialized object encountered: {type(obj)}")

    @classmethod
    def flatten(cls, obj: Any):
        serialized = cls.serialize(obj)
        keypath_leaves, _ = torch.utils._pytree.tree_flatten_with_path(serialized)
        tensor_paths = [path for path, leaf in keypath_leaves if isinstance(leaf, torch.Tensor)]
        values = [leaf for _, leaf in keypath_leaves if isinstance(leaf, torch.Tensor)]
        template = torch.utils._pytree.tree_map_only(torch.Tensor, lambda _: None, serialized)
        return values, _FlattenedContext(template=template, tensor_paths=tensor_paths)

    @classmethod
    def flatten_with_keys(cls, obj: Any):
        values, context = cls.flatten(obj)
        key_entries = [(torch.utils._pytree.SequenceKey(index), value) for index, value in enumerate(values)]
        return key_entries, context

    @classmethod
    def unflatten(cls, values, context):
        tensor_values_by_path = dict(zip(context.tensor_paths, values, strict=True))
        deserialized = torch.utils._pytree.tree_map_with_path(
            lambda path, leaf: tensor_values_by_path.get(path, leaf), context.template
        )
        return cls.deserialize(deserialized)

    @classmethod
    def register_class(cls, object_cls: type):
        if object_cls in cls._registered_classes:
            return

        try:
            torch.utils._pytree.register_pytree_node(
                object_cls,
                cls.flatten,
                cls.unflatten,
                serialized_type_name=_class_to_path(object_cls),
                flatten_with_keys_fn=cls.flatten_with_keys,
            )
            cls._registered_classes.add(object_cls)
        except ValueError as e:
            if "already registered as pytree node" in str(e):
                cls._registered_classes.add(object_cls)
            else:
                raise


def register_cache_subclasses_as_pytree_nodes():
    for cache_type in _iter_subclasses(Cache):
        PytreeRegistry.register_class(cache_type)


def register_custom_model_cache_as_pytree_nodes(model: "PreTrainedModel"):
    for _, obj in inspect.getmembers(inspect.getmodule(model)):
        if (
            inspect.isclass(obj)
            and obj.__module__ == model.__class__.__module__
            and obj.__name__.endswith("Cache")
            and not issubclass(obj, Cache)
        ):
            PytreeRegistry.register_class(obj)


def _iter_leaf_tensors(obj: Any, prefix: str = ""):
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, (list, tuple, set)):
        for index, item in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_leaf_tensors(item, path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _iter_leaf_tensors(value, path)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix)


def get_leaf_tensors(obj: Any) -> dict[str, torch.Tensor]:
    """
    Recursively retrieves all leaf tensors from a potentialy nested structure.
    Args:
        obj (`Any`):
            The object from which to retrieve leaf tensors.
    Returns:
        `dict[str, torch.Tensor]`: A dictionary mapping names to leaf tensors.
    """
    return dict(_iter_leaf_tensors(obj))


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
) -> tuple["PreTrainedModel", dict[str, torch.Tensor | Cache], dict[str, Any] | None]:
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

    with torch.no_grad():
        outputs = model(**copy.deepcopy(inputs))

    # Inject the prefill KV cache into inputs and widen the attention mask to cover all cached
    # positions, mirroring what the generation loop does between prefill and first decode step.
    # We delegate to _update_model_kwargs_for_generation so any upstream changes stay in sync,
    # but we must tell it how many tokens were just cached (num_new_tokens = prefill length) so
    # it widens the attention mask by the right amount.
    #
    # One subtlety: unlike in the generation loop, here we keep input_ids at the full prefill
    # length rather than slicing it to 1 token, so position_ids / cache_position / token_type_ids
    # must NOT be extended. We temporarily remove them, call the method, then restore them.
    is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
    cache_in_inputs = next((inputs[n] for n in ALL_CACHE_NAMES if inputs.get(n) is not None), None)
    cache_in_outputs = next((outputs[n] for n in ALL_CACHE_NAMES if outputs.get(n) is not None), None)
    seq_len_tied_keys = (
        "cache_position",
        "token_type_ids",
        "decoder_position_ids" if is_encoder_decoder else "position_ids",
    )
    if (
        cache_in_inputs is None
        and cache_in_outputs is not None
        and getattr(model.config, "use_cache", False)
        and hasattr(model, "_update_model_kwargs_for_generation")
    ):
        if isinstance(cache_in_outputs, Cache):
            num_new_tokens = cache_in_outputs.get_seq_length()
        elif isinstance(cache_in_outputs, (list, tuple)) and isinstance(cache_in_outputs[0], (list, tuple)):
            num_new_tokens = cache_in_outputs[0][0].shape[2]  # legacy tuple-of-tuples
        else:
            logger.warning(
                f"Unexpected cache structure in model outputs: {type(cache_in_outputs)}. "
                "Expected a Cache or a tuple of tuples of tensors. Skipping cache injection into inputs."
            )
            num_new_tokens = None

        if num_new_tokens is not None:
            saved = {k: inputs.pop(k) for k in seq_len_tied_keys if k in inputs}
            inputs = (
                model._update_model_kwargs_for_generation(
                    outputs,
                    inputs,
                    is_encoder_decoder=is_encoder_decoder,
                    num_new_tokens=num_new_tokens,
                )
                | saved
            )

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

    return dict.fromkeys(range(len(tensor.shape)), torch.export.Dim.AUTO)


def get_auto_dynamic_shapes(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Utility function to automatically generate dynamic shapes for all tensor inputs and cache inputs.
    Args:
        inputs (`dict[str, Any]`):
            A dictionary of model inputs.
    Returns:
        `dict[str, Any]`: A dictionary mapping input names to their dynamic shapes.
    """
    dynamic_shapes = {}
    for name, input in inputs.items():
        if isinstance(input, Cache):
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
        elif hasattr(input, "__dict__"):
            dynamic_shapes[name] = get_auto_dynamic_shapes(
                {k: v for k, v in vars(input).items() if not k.startswith("_")}
            )
        else:
            raise ValueError(
                f"Input '{name}' is of unsupported type '{type(input)}'. "
                "Only torch.Tensor, Cache, int, float, bool, str, dict, list, tuple, set, "
                "and objects with a __dict__ (e.g. custom cache classes) are supported."
            )

    return dynamic_shapes
