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

from ..cache_utils import Cache
from ..generation.utils import ALL_CACHE_NAMES
from ..utils.import_utils import is_torch_available
from ..utils.logging import get_logger


logger = get_logger(__name__)


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


def _path_to_class(path: str) -> type:
    module_name, qualname = path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _flatten_to_context(obj: Any, tensors: list) -> Any:
    """Single-pass: recursively build a JSON-native context while collecting tensors into `tensors`."""
    # --- Pure Python / JSON-native (exact type check — subclasses fall through to stateful objects) ---
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    if type(obj) is list:
        return [_flatten_to_context(i, tensors) for i in obj]
    if type(obj) is dict:
        return {k: _flatten_to_context(v, tensors) for k, v in obj.items()}

    # --- Torch objects ---
    if isinstance(obj, torch.Tensor):
        idx = len(tensors)
        tensors.append(obj)
        return {"_t": "tensor", "i": idx}
    if isinstance(obj, torch.Size):
        return {"_t": "size", "v": list(obj)}
    if isinstance(obj, torch.device):
        return {"_t": "device", "s": str(obj)}
    if isinstance(obj, torch.dtype):
        return {"_t": "dtype", "n": str(obj).removeprefix("torch.")}
    if isinstance(obj, torch.layout):
        return {"_t": "layout", "n": str(obj).removeprefix("torch.")}

    # --- Python types ---
    if isinstance(obj, type):
        return {"_t": "type", "p": _class_to_path(obj)}

    # --- Generic Python objects (by structural category) ---
    cls = type(obj)
    if isinstance(obj, dict):  # dict subclasses (OrderedDict, etc.)
        return {
            "_t": "map",
            "p": _class_to_path(cls),
            "v": {k: _flatten_to_context(v, tensors) for k, v in obj.items()},
        }
    if isinstance(obj, (tuple, list, set, frozenset)):  # sequences/sets incl. NamedTuple
        return {
            "_t": "seq",
            "p": _class_to_path(cls),
            "v": [_flatten_to_context(i, tensors) for i in obj],
        }
    if hasattr(obj, "__dict__"):
        return {
            "_t": "obj",
            "p": _class_to_path(cls),
            "s": {k: _flatten_to_context(v, tensors) for k, v in vars(obj).items()},
        }

    raise TypeError(f"Cannot flatten {type(obj).__name__} for pytree context")


def _unflatten_from_context(ctx: Any, tensors: list) -> Any:
    """Reconstruct an object from its JSON-native context, substituting tensor index markers."""
    # --- Pure Python / JSON-native ---
    if ctx is None or type(ctx) in (bool, int, float, str):
        return ctx
    if type(ctx) is list:
        return [_unflatten_from_context(i, tensors) for i in ctx]
    if type(ctx) is dict and "_t" not in ctx:
        return {k: _unflatten_from_context(v, tensors) for k, v in ctx.items()}

    # --- Torch objects ---
    t = ctx["_t"]
    if t == "tensor":
        return tensors[ctx["i"]]
    if t == "layout":
        return getattr(torch, ctx["n"])
    if t == "dtype":
        return getattr(torch, ctx["n"])
    if t == "device":
        return torch.device(ctx["s"])
    if t == "size":
        return torch.Size(ctx["v"])

    # --- Python types ---
    if t == "type":
        return _path_to_class(ctx["p"])

    # --- Generic Python objects ---
    if t == "map":
        cls = _path_to_class(ctx["p"])
        return cls({k: _unflatten_from_context(v, tensors) for k, v in ctx["v"].items()})
    if t == "seq":
        cls = _path_to_class(ctx["p"])
        items = [_unflatten_from_context(i, tensors) for i in ctx["v"]]
        try:
            return cls(items)  # tuple, list subclass, set, frozenset, etc.
        except TypeError:
            return cls(*items)  # NamedTuple (requires positional args)
    if t == "obj":
        cls = _path_to_class(ctx["p"])
        state = {k: _unflatten_from_context(v, tensors) for k, v in ctx["s"].items()}
        instance = cls.__new__(cls)
        instance.__dict__.update(state)
        return instance

    raise TypeError(f"Unknown tag {t!r} in pytree context")


def _iter_subclasses(cls: type):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _iter_subclasses(subclass)


def _class_to_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _pytree_flatten(obj: Any) -> tuple[list, Any]:
    tensors: list = []
    context = _flatten_to_context(obj, tensors)
    return tensors, context


def _pytree_flatten_with_keys(obj: Any):
    leaves, context = _pytree_flatten(obj)
    return [(torch.utils._pytree.SequenceKey(i), leaf) for i, leaf in enumerate(leaves)], context


def _pytree_unflatten(values, context: Any) -> Any:
    return _unflatten_from_context(context, list(values))


def register_for_export(object_cls: type):
    try:
        torch.utils._pytree.register_pytree_node(
            object_cls,
            _pytree_flatten,
            _pytree_unflatten,
            serialized_type_name=_class_to_path(object_cls),
            flatten_with_keys_fn=_pytree_flatten_with_keys,
        )
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


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
    """Recursively retrieves all leaf tensors from a potentially nested structure."""
    return dict(_iter_leaf_tensors(obj))


def register_pytrees_for_model(model: "PreTrainedModel"):
    """Register all relevant cache types as pytree nodes for torch.export."""
    # All transformers Cache subclasses
    for cache_type in _iter_subclasses(Cache):
        register_for_export(cache_type)
    # Model-specific cache classes not inheriting from Cache (e.g. custom per-model caches)
    for _, obj in inspect.getmembers(inspect.getmodule(model)):
        if (
            inspect.isclass(obj)
            and obj.__module__ == model.__class__.__module__
            and obj.__name__.endswith("Cache")
            and not issubclass(obj, Cache)
        ):
            register_for_export(obj)


def prepare_model_for_export(
    model: "PreTrainedModel",
    inputs: dict[str, Any],
) -> tuple["PreTrainedModel", dict[str, Any]]:
    """Configure the model for export (no inference). Moves output flags from inputs to model.config,
    sets optimal attention/experts implementations, and patches non-exportable module behaviours."""
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

    # Cast floating-point inputs to match the model's dtype and device
    try:
        model_param = next(iter(model.parameters()))
        inputs = _cast_inputs(inputs, model_param.device, model_param.dtype)
    except StopIteration:
        pass  # model has no parameters (e.g. pure embedding model)

    return model, inputs


def inject_cache_into_inputs(
    model: "PreTrainedModel",
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Run one prefill forward pass and inject the resulting KV cache back into inputs.

    Mirrors what the generation loop does between prefill and the first decode step: the
    attention mask is widened to cover all cached positions via
    ``_update_model_kwargs_for_generation``.  Keys that are tied to the *prefill* sequence
    length (``cache_position``, ``position_ids``, ``token_type_ids``) are temporarily removed
    during that call so they are not incorrectly extended, then restored.
    """
    with torch.no_grad():
        outputs = model(**copy.deepcopy(inputs))

    is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
    cache_in_inputs = next((inputs[n] for n in ALL_CACHE_NAMES if inputs.get(n) is not None), None)
    cache_in_outputs = (
        next((outputs[n] for n in ALL_CACHE_NAMES if outputs.get(n) is not None), None)
        if isinstance(outputs, dict)
        else None
    )
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

    return inputs, outputs


def get_inputs_outputs_names(inputs: dict[str, Any], outputs: dict[str, Any]) -> tuple[list[str], list[str]]:
    inputs_names = list(get_leaf_tensors(inputs).keys())
    outputs_names = list(get_leaf_tensors(outputs).keys())
    for name in set(inputs_names).intersection(set(outputs_names)):
        inputs_names[inputs_names.index(name)] = f"input.{name}"
        outputs_names[outputs_names.index(name)] = f"output.{name}"

    if outputs_names == [""]:
        outputs_names = ["output"]

    return inputs_names, outputs_names


def dedup_output_tensors(obj: Any, seen: dict | None = None) -> Any:
    """Clone tensors that appear more than once in an output structure.

    When a model returns the same tensor under two output names (e.g. ``last_hidden_state``
    and ``hidden_states[0]``), the ONNX optimizer deduplicates the two output nodes and
    renames one, breaking the expected name mapping. Cloning duplicates gives each output
    leaf a distinct identity so the optimizer has nothing to merge.
    """
    if seen is None:
        seen = {}
    if isinstance(obj, type):  # class objects: not tensors, can't be reconstructed generically
        return obj
    if isinstance(obj, torch.Tensor):
        if id(obj) in seen:
            return obj.clone()
        seen[id(obj)] = True
        return obj
    if isinstance(obj, dict):
        return type(obj)({k: dedup_output_tensors(v, seen) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        items = [dedup_output_tensors(v, seen) for v in obj]
        try:
            return type(obj)(items)
        except TypeError:
            return type(obj)(*items)  # NamedTuple
    if hasattr(obj, "__dict__"):
        cls = type(obj)
        instance = cls.__new__(cls)
        instance.__dict__.update({k: dedup_output_tensors(v, seen) for k, v in vars(obj).items()})
        return instance
    return obj


def _cast_inputs(obj: Any, device: "torch.device", dtype: "torch.dtype") -> Any:
    """Recursively move tensors to `device`, casting floating-point tensors to `dtype`."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype) if obj.is_floating_point() else obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _cast_inputs(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cast = [_cast_inputs(v, device, dtype) for v in obj]
        return type(obj)(cast)
    return obj


def prepare_for_export(
    model: "PreTrainedModel",
    inputs: dict[str, torch.Tensor | Cache],
) -> tuple["PreTrainedModel", dict[str, torch.Tensor | Cache], dict[str, Any] | None]:
    inputs = {k: v for k, v in inputs.items() if v is not None}
    for input_name in ("labels", "future_values"):
        if input_name in inputs:
            logger.info(f"Found an input '{input_name}' which is not supported for export. Popping it from inputs.")
            inputs.pop(input_name)
    model, inputs = prepare_model_for_export(model, inputs)
    inputs, outputs = inject_cache_into_inputs(model, inputs)
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
        if isinstance(input, torch.Tensor):
            dynamic_shapes[name] = _auto_dynamic_shape(input)
        elif isinstance(input, (int, float, bool, str)):
            dynamic_shapes[name] = None
        elif isinstance(input, dict):
            dynamic_shapes[name] = get_auto_dynamic_shapes(input)
        elif isinstance(input, (list, tuple, set)):
            dynamic_shapes[name] = type(input)(get_auto_dynamic_shapes(dict(enumerate(input))).values())
        elif hasattr(input, "__dict__"):
            leaves, _ = _pytree_flatten(input)
            dynamic_shapes[name] = [_auto_dynamic_shape(t) for t in leaves]
        else:
            raise ValueError(
                f"Input '{name}' is of unsupported type '{type(input)}'. "
                "Only torch.Tensor, Cache, int, float, bool, str, dict, list, tuple, set, "
                "and objects with a __dict__ (e.g. custom cache classes) are supported."
            )

    return dynamic_shapes
