# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Generic utilities
"""

from __future__ import annotations

import inspect
import json
import os
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Callable, Iterable, MutableMapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from ..utils import logging
from .import_utils import is_mlx_available, is_torch_available, is_torch_fx_proxy


_is_torch_available = False
if is_torch_available():
    # required for @can_return_tuple decorator to work with torchdynamo
    import torch
    from torch.types import _dtype

    from ..model_debugging_utils import model_addition_debugger_context

    _is_torch_available = True


if TYPE_CHECKING:
    from torch import nn


logger = logging.get_logger(__name__)


# required for @can_return_tuple decorator to work with torchdynamo
_is_mlx_available = False
if is_mlx_available():
    _is_mlx_available = True


# vendored from distutils.util
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    elif representation.startswith("<class 'numpy."):
        return "np"
    elif representation.startswith("<class 'mlx."):
        return "mlx"


def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "pt": is_torch_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    # We will test this one first, then numpy, then the others.
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    return {f: framework_to_test[f] for f in frameworks}


def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `np.ndarray` or `mlx.array` in the order defined by `infer_framework_from_repr`
    """
    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    # Tracers
    if is_torch_fx_proxy(x):
        return True

    return False


def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    return isinstance(x, np.ndarray)


def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    return _is_torch_available and isinstance(x, torch.Tensor)


def is_torch_device(x):
    """
    Tests if `x` is a torch device or not. Safe to call even if torch is not installed.
    """
    return _is_torch_available and isinstance(x, torch.device)


def is_torch_dtype(x):
    """
    Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed.
    """
    if not _is_torch_available:
        return False
    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    return isinstance(x, torch.dtype)


def _is_tensor_or_array_like(value):
    """
    Check if a value is array-like (includes ragged arrays)
    """
    if is_numpy_array(value):
        return True
    if is_torch_tensor(value):
        return True
    if isinstance(value, (int, float, bool, np.number)):
        return True

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            # consider empty list or nested list as array-like
            return True
        return _is_tensor_or_array_like(value[0])

    return False


def maybe_autocast(
    device_type: str,
    dtype: _dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()


def _is_mlx(x):
    import mlx.core as mx

    return isinstance(x, mx.array)


def is_mlx_array(x):
    """
    Tests if `x` is a mlx array or not. Safe to call even when mlx is not installed.
    """
    return False if not _is_mlx_available else _is_mlx(x)


def is_flash_attention_requested(config=None, requested_attention_implementation: str | None = None):
    """
    Checks whether some flavor of flash attention is requested or not.

    This is checked against one of the two arguments, i.e. either the `config` or the directly passed value
    `requested_attention_implementation`. Otherwise, an error will be raised (ambiguity).

    The different versions of flash attention are usually
    - Implementations based on the original flash attention repo: https://github.com/Dao-AILab/flash-attention
    - Kernels implementations such as: https://huggingface.co/kernels-community/vllm-flash-attn3
    """
    if config is not None and requested_attention_implementation is not None:
        raise ValueError(
            "Requested attention implementation is ambiguous: "
            "Please pass either the config or the name of the attention implementation, not both."
        )

    if config is not None:
        checked_attention_implementation = config._attn_implementation
    else:
        checked_attention_implementation = requested_attention_implementation

    return "flash" in checked_attention_implementation


def to_py_obj(obj):
    """
    Convert a PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Only convert directly if all elements are numeric scalars
        if all(isinstance(x, (int, float, np.number)) for x in obj):
            return list(obj)

        # Otherwise recurse element-wise
        return [to_py_obj(o) for o in obj]

    framework_to_py_obj = {
        "pt": lambda obj: obj.tolist(),
        "np": lambda obj: obj.tolist(),
    }

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_py_obj[framework](obj)

    # tolist also works on 0d np arrays
    if isinstance(obj, np.number):
        return obj.tolist()
    else:
        return obj


def to_numpy(obj):
    """
    Convert a PyTorch tensor, Numpy array or python list to a Numpy array.
    """

    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "np": lambda obj: obj,
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


def safe_load_json_file(json_file: str):
    "A helper to load safe config files and raise a proper error message if it wasn't serialized correctly"
    try:
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
    except json.JSONDecodeError:
        raise OSError(f"It looks like the config file at '{json_file}' is not a valid JSON file.")
    return config_dict


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        if _is_torch_available:
            from torch.utils._pytree import register_pytree_node

            register_pytree_node(
                cls,
                _model_output_flatten,
                partial(_model_output_unflatten, output_type=cls),
                serialized_type_name=f"{cls.__module__}.{cls.__name__}",
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                # reset first field to None and remove it from the internal dictionary
                setattr(self, class_fields[0].name, None)
                super().__delitem__(class_fields[0].name)
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        field_names = {field.name for field in fields(self)}
        if name in field_names and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


if _is_torch_available:
    import torch.utils._pytree as _torch_pytree

    def _model_output_flatten(output: ModelOutput) -> tuple[list[Any], _torch_pytree.Context]:
        return list(output.values()), list(output.keys())

    def _model_output_unflatten(
        values: Iterable[Any],
        context: _torch_pytree.Context,
        output_type: type[ModelOutput] | None = None,
    ) -> ModelOutput:
        return output_type(**dict(zip(context, values)))

    _torch_pytree.register_pytree_node(
        ModelOutput,
        _model_output_flatten,
        partial(_model_output_unflatten, output_type=ModelOutput),
        serialized_type_name=f"{ModelOutput.__module__}.{ModelOutput.__name__}",
    )


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    NUMPY = "np"
    MLX = "mlx"


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: list[AbstractContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def can_return_loss(model_class):
    """
    Check if a given model can return loss.

    Args:
        model_class (`type`): The class of the model.
    """
    signature = inspect.signature(model_class.forward)

    for p in signature.parameters:
        if p == "return_loss" and signature.parameters[p].default is True:
            return True

    return False


def find_labels(model_class):
    """
    Find the labels used by a given model.

    Args:
        model_class (`type`): The class of the model.
    """
    model_name = model_class.__name__
    signature = inspect.signature(model_class.forward)

    if "QuestionAnswering" in model_name:
        return [p for p in signature.parameters if "label" in p or p in ("start_positions", "end_positions")]
    else:
        return [p for p in signature.parameters if "label" in p]


def flatten_dict(d: MutableMapping, parent_key: str = "", delimiter: str = "."):
    """Flatten a nested dict into a single level dict."""

    def _flatten_dict(d, parent_key="", delimiter="."):
        for k, v in d.items():
            key = str(parent_key) + delimiter + str(k) if parent_key else k
            if v and isinstance(v, MutableMapping):
                yield from flatten_dict(v, key, delimiter=delimiter).items()
            else:
                yield key, v

    return dict(_flatten_dict(d, parent_key, delimiter))


def transpose(array, axes=None):
    """
    Framework-agnostic version of transpose operation.
    """
    if is_numpy_array(array):
        return np.transpose(array, axes=axes)
    elif is_torch_tensor(array):
        return array.T if axes is None else array.permute(*axes)
    else:
        raise ValueError(f"Type not supported for transpose: {type(array)}.")


def reshape(array, newshape):
    """
    Framework-agnostic version of reshape operation.
    """
    if is_numpy_array(array):
        return np.reshape(array, newshape)
    elif is_torch_tensor(array):
        return array.reshape(*newshape)
    else:
        raise ValueError(f"Type not supported for reshape: {type(array)}.")


def squeeze(array, axis=None):
    """
    Framework-agnostic version of squeeze operation.
    """
    if is_numpy_array(array):
        return np.squeeze(array, axis=axis)
    elif is_torch_tensor(array):
        return array.squeeze() if axis is None else array.squeeze(dim=axis)
    else:
        raise ValueError(f"Type not supported for squeeze: {type(array)}.")


def expand_dims(array, axis):
    """
    Framework-agnostic version of expand_dims operation.
    """
    if is_numpy_array(array):
        return np.expand_dims(array, axis)
    elif is_torch_tensor(array):
        return array.unsqueeze(dim=axis)
    else:
        raise ValueError(f"Type not supported for expand_dims: {type(array)}.")


def tensor_size(array):
    """
    Framework-agnostic version of size operation.
    """
    if is_numpy_array(array):
        return np.size(array)
    elif is_torch_tensor(array):
        return array.numel()
    else:
        raise ValueError(f"Type not supported for tensor_size: {type(array)}.")


def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    if not _is_torch_available:
        return int(x)

    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


def torch_float(x):
    """
    Casts an input to a torch float32 tensor if we are in a tracing context, otherwise to a Python float.
    """
    if not _is_torch_available:
        return int(x)

    return x.to(torch.float32) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


def filter_out_non_signature_kwargs(extra: list | None = None):
    """
    Decorator to filter out named arguments that are not in the function signature.

    This decorator ensures that only the keyword arguments that match the function's signature, or are specified in the
    `extra` list, are passed to the function. Any additional keyword arguments are filtered out and a warning is issued.

    Parameters:
        extra (`Optional[list]`, *optional*):
            A list of extra keyword argument names that are allowed even if they are not in the function's signature.

    Returns:
        Callable:
            A decorator that wraps the function and filters out invalid keyword arguments.

    Example usage:

        ```python
        @filter_out_non_signature_kwargs(extra=["allowed_extra_arg"])
        def my_function(arg1, arg2, **kwargs):
            print(arg1, arg2, kwargs)

        my_function(arg1=1, arg2=2, allowed_extra_arg=3, invalid_arg=4)
        # This will print: 1 2 {"allowed_extra_arg": 3}
        # And issue a warning: "The following named arguments are not valid for `my_function` and were ignored: 'invalid_arg'"
        ```
    """
    extra = extra or []
    extra_params_to_pass = set(extra)

    def decorator(func):
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        valid_kwargs_to_pass = function_named_args.union(extra_params_to_pass)

        # Required for better warning message
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        # Mark function as decorated
        func._filter_out_non_signature_kwargs = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            valid_kwargs = {}
            invalid_kwargs = {}

            for k, v in kwargs.items():
                if k in valid_kwargs_to_pass:
                    valid_kwargs[k] = v
                else:
                    invalid_kwargs[k] = v

            if invalid_kwargs:
                invalid_kwargs_names = [f"'{k}'" for k in invalid_kwargs]
                invalid_kwargs_names = ", ".join(invalid_kwargs_names)

                # Get the class name for better warning message
                if is_instance_method:
                    cls_prefix = args[0].__class__.__name__ + "."
                elif is_class_method:
                    cls_prefix = args[0].__name__ + "."
                else:
                    cls_prefix = ""

                warnings.warn(
                    f"The following named arguments are not valid for `{cls_prefix}{func.__name__}`"
                    f" and were ignored: {invalid_kwargs_names}",
                    UserWarning,
                    stacklevel=2,
                )

            return func(*args, **valid_kwargs)

        return wrapper

    return decorator


class TransformersKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the forward pass of a `PreTrainedModel`.

    Attributes:
        num_items_in_batch (`Optional[torch.Tensor]`, *optional*):
            Number of items in the batch. It is recommended to pass it when you are doing gradient accumulation.
        output_hidden_states (`Optional[bool]`, *optional*):
            Most of the models support outputting all hidden states computed during the forward pass.
        output_attentions (`Optional[bool]`, *optional*):
            Turn this on to return the intermediary attention scores.
        output_router_logits (`Optional[bool]`, *optional*):
            For MoE models, this allows returning the router logits to compute the loss.
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
        position_ids (`torch.LongTensor`, *optional*)
            Indices of positions of each input sequence tokens.
        is_causal (`bool`, *optional*)
            Can be set to False to enable bi-directional attention, i.e. use decoder Attention modules as encoders.
    """

    num_items_in_batch: torch.Tensor | None
    output_hidden_states: bool | None
    output_attentions: bool | None
    output_router_logits: bool | None
    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_length_q: int | None
    max_length_k: int | None
    position_ids: torch.LongTensor | None
    is_causal: bool | None


def is_timm_config_dict(config_dict: dict[str, Any]) -> bool:
    """Checks whether a config dict is a timm config dict."""
    return "pretrained_cfg" in config_dict


def is_timm_local_checkpoint(pretrained_model_path: str) -> bool:
    """
    Checks whether a checkpoint is a timm model checkpoint.
    """
    if pretrained_model_path is None:
        return False

    # in case it's Path, not str
    pretrained_model_path = str(pretrained_model_path)

    is_file = os.path.isfile(pretrained_model_path)
    is_dir = os.path.isdir(pretrained_model_path)

    # pretrained_model_path is a file
    if is_file and pretrained_model_path.endswith(".json"):
        with open(pretrained_model_path) as f:
            config_dict = json.load(f)
        return is_timm_config_dict(config_dict)

    # pretrained_model_path is a directory with a config.json
    if is_dir and os.path.exists(os.path.join(pretrained_model_path, "config.json")):
        with open(os.path.join(pretrained_model_path, "config.json")) as f:
            config_dict = json.load(f)
        return is_timm_config_dict(config_dict)

    return False


def set_attribute_for_modules(module: nn.Module, key: str, value: Any):
    """
    Set a value to a module and all submodules.
    """
    setattr(module, key, value)
    for submodule in module.children():
        set_attribute_for_modules(submodule, key, value)


def del_attribute_from_modules(module: nn.Module, key: str):
    """
    Delete a value from a module and all submodules.
    """
    # because we might remove it previously in case it's a shared module, e.g. activation function
    if hasattr(module, key):
        delattr(module, key)

    for submodule in module.children():
        del_attribute_from_modules(submodule, key)


def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    use_return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output

    return wrapper


def merge_with_config_defaults(func):
    """
    Decorator using config field (if they exist) as default value for some args and kwargs. Precedence is always
    given to the args/kwargs that are explicitly passed.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        args_with_config_defaults = [
            "use_cache",
            "vision_feature_layer",
            "vision_feature_select_strategy",
            "vision_aspect_ratio",
        ]
        for arg_name in args_with_config_defaults:
            arg_index = None
            if arg_name in func.__code__.co_varnames:
                arg_index = func.__code__.co_varnames.index(arg_name) - 1  # -1 for self

            if arg_index is not None and len(args) > arg_index and args[arg_index] is not None:
                arg_value = args[arg_index]
            elif kwargs.get(arg_name) is not None:
                arg_value = kwargs[arg_name]
            else:
                arg_value = getattr(self.config, arg_name, None)

            if arg_value is not None:
                # Arg-specific handling
                if arg_name == "use_cache":
                    if getattr(self, "gradient_checkpointing", False) and self.training and arg_value:
                        logger.warning_once(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                        )
                        arg_value = False
                elif arg_name == "vision_feature_select_strategy":
                    valid_strategies = ["default", "full"]
                    if arg_value not in valid_strategies:
                        raise ValueError(
                            f"`Unexpected select feature strategy: {arg_value}. Please select from {valid_strategies}."
                        )

                if arg_index is not None and len(args) > arg_index:
                    args = list(args)
                    args[arg_index] = arg_value
                    args = tuple(args)
                else:
                    kwargs[arg_name] = arg_value

        # Maybe temporarily overwrite config value to create the correct mask - kwarg takes precedence
        is_causal = kwargs.get("is_causal", getattr(self.config, "is_causal", None))
        if is_causal is not None:
            is_causal_in_config = hasattr(self.config, "is_causal")
            if is_causal_in_config:
                is_causal_original_value = self.config.is_causal
            # Set it to both config and kwargs (it's needed in both, and can come from only 1 of the sources)
            self.config.is_causal = is_causal
            kwargs["is_causal"] = is_causal

        # Call the original forward with the updated kwargs/config
        try:
            if kwargs.get("debug_io", False):
                with model_addition_debugger_context(
                    self, kwargs.get("debug_io_dir", "model_debug"), kwargs.get("prune_layers")
                ):
                    output = func(self, *args, **kwargs)
            else:
                output = func(self, *args, **kwargs)
        # Restore original config value
        finally:
            if is_causal is not None:
                if is_causal_in_config:
                    self.config.is_causal = is_causal_original_value
                else:
                    del self.config.is_causal

        return output

    return wrapper


class GeneralInterface(MutableMapping):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications though the class mapping, as well as local modifications in a single file with the local mapping.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())
