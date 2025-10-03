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

import inspect
import json
import os
import tempfile
import warnings
from collections import OrderedDict, UserDict, defaultdict
from collections.abc import Iterable, MutableMapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, Callable, Optional, TypedDict

import numpy as np

from ..utils import logging
from .import_utils import (
    is_flax_available,
    is_mlx_available,
    is_tf_available,
    is_torch_available,
    is_torch_fx_proxy,
    requires,
)


_CAN_RECORD_REGISTRY = {}


logger = logging.get_logger(__name__)

if is_torch_available():
    # required for @can_return_tuple decorator to work with torchdynamo
    import torch

    from ..model_debugging_utils import model_addition_debugger_context


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
    elif representation.startswith("<class 'tensorflow."):
        return "tf"
    elif representation.startswith("<class 'jax"):
        return "jax"
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
        "tf": is_tf_tensor,
        "jax": is_jax_tensor,
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
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray`, `np.ndarray` or `mlx.array`
    in the order defined by `infer_framework_from_repr`
    """
    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    # Tracers
    if is_torch_fx_proxy(x):
        return True

    if is_flax_available():
        from jax.core import Tracer

        if isinstance(x, Tracer):
            return True

    return False


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    return _is_numpy(x)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch(x)


def _is_torch_device(x):
    import torch

    return isinstance(x, torch.device)


def is_torch_device(x):
    """
    Tests if `x` is a torch device or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch_device(x)


def _is_torch_dtype(x):
    import torch

    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    return isinstance(x, torch.dtype)


def is_torch_dtype(x):
    """
    Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch_dtype(x)


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def is_tf_tensor(x):
    """
    Tests if `x` is a tensorflow tensor or not. Safe to call even if tensorflow is not installed.
    """
    return False if not is_tf_available() else _is_tensorflow(x)


def _is_tf_symbolic_tensor(x):
    import tensorflow as tf

    # the `is_symbolic_tensor` predicate is only available starting with TF 2.14
    if hasattr(tf, "is_symbolic_tensor"):
        return tf.is_symbolic_tensor(x)
    return isinstance(x, tf.Tensor)


def is_tf_symbolic_tensor(x):
    """
    Tests if `x` is a tensorflow symbolic tensor or not (ie. not eager). Safe to call even if tensorflow is not
    installed.
    """
    return False if not is_tf_available() else _is_tf_symbolic_tensor(x)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    return isinstance(x, jnp.ndarray)


def is_jax_tensor(x):
    """
    Tests if `x` is a Jax tensor or not. Safe to call even if jax is not installed.
    """
    return False if not is_flax_available() else _is_jax(x)


def _is_mlx(x):
    import mlx.core as mx

    return isinstance(x, mx.array)


def is_mlx_array(x):
    """
    Tests if `x` is a mlx array or not. Safe to call even when mlx is not installed.
    """
    return False if not is_mlx_available() else _is_mlx(x)


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        try:
            arr = np.array(obj)
            if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
                return arr.tolist()
        except Exception:
            pass
        return [to_py_obj(o) for o in obj]

    framework_to_py_obj = {
        "pt": lambda obj: obj.tolist(),
        "tf": lambda obj: obj.numpy().tolist(),
        "jax": lambda obj: np.asarray(obj).tolist(),
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
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array.
    """

    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "tf": lambda obj: obj.numpy(),
        "jax": lambda obj: np.asarray(obj),
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
        if is_torch_available():
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
                # reset first field to None
                setattr(self, class_fields[0].name, None)
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
        if name in self.keys() and value is not None:
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


if is_torch_available():
    import torch.utils._pytree as _torch_pytree

    def _model_output_flatten(output: ModelOutput) -> tuple[list[Any], "_torch_pytree.Context"]:
        return list(output.values()), list(output.keys())

    def _model_output_unflatten(
        values: Iterable[Any],
        context: "_torch_pytree.Context",
        output_type=None,
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
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
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
    framework = infer_framework(model_class)
    if framework == "tf":
        signature = inspect.signature(model_class.call)  # TensorFlow models
    elif framework == "pt":
        signature = inspect.signature(model_class.forward)  # PyTorch models
    else:
        signature = inspect.signature(model_class.__call__)  # Flax models

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
    framework = infer_framework(model_class)
    if framework == "tf":
        signature = inspect.signature(model_class.call)  # TensorFlow models
    elif framework == "pt":
        signature = inspect.signature(model_class.forward)  # PyTorch models
    else:
        signature = inspect.signature(model_class.__call__)  # Flax models

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


@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir: bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield working_dir


def transpose(array, axes=None):
    """
    Framework-agnostic version of `numpy.transpose` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.transpose(array, axes=axes)
    elif is_torch_tensor(array):
        return array.T if axes is None else array.permute(*axes)
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.transpose(array, perm=axes)
    elif is_jax_tensor(array):
        import jax.numpy as jnp

        return jnp.transpose(array, axes=axes)
    else:
        raise ValueError(f"Type not supported for transpose: {type(array)}.")


def reshape(array, newshape):
    """
    Framework-agnostic version of `numpy.reshape` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.reshape(array, newshape)
    elif is_torch_tensor(array):
        return array.reshape(*newshape)
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.reshape(array, newshape)
    elif is_jax_tensor(array):
        import jax.numpy as jnp

        return jnp.reshape(array, newshape)
    else:
        raise ValueError(f"Type not supported for reshape: {type(array)}.")


def squeeze(array, axis=None):
    """
    Framework-agnostic version of `numpy.squeeze` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.squeeze(array, axis=axis)
    elif is_torch_tensor(array):
        return array.squeeze() if axis is None else array.squeeze(dim=axis)
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.squeeze(array, axis=axis)
    elif is_jax_tensor(array):
        import jax.numpy as jnp

        return jnp.squeeze(array, axis=axis)
    else:
        raise ValueError(f"Type not supported for squeeze: {type(array)}.")


def expand_dims(array, axis):
    """
    Framework-agnostic version of `numpy.expand_dims` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.expand_dims(array, axis)
    elif is_torch_tensor(array):
        return array.unsqueeze(dim=axis)
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.expand_dims(array, axis=axis)
    elif is_jax_tensor(array):
        import jax.numpy as jnp

        return jnp.expand_dims(array, axis=axis)
    else:
        raise ValueError(f"Type not supported for expand_dims: {type(array)}.")


def tensor_size(array):
    """
    Framework-agnostic version of `numpy.size` that will work on torch/TensorFlow/Jax tensors as well as NumPy arrays.
    """
    if is_numpy_array(array):
        return np.size(array)
    elif is_torch_tensor(array):
        return array.numel()
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.size(array)
    elif is_jax_tensor(array):
        return array.size
    else:
        raise ValueError(f"Type not supported for tensor_size: {type(array)}.")


def infer_framework(model_class):
    """
    Infers the framework of a given model without using isinstance(), because we cannot guarantee that the relevant
    classes are imported or available.
    """
    for base_class in inspect.getmro(model_class):
        module = base_class.__module__
        name = base_class.__name__
        if module.startswith("tensorflow") or module.startswith("keras") or name == "TFPreTrainedModel":
            return "tf"
        elif module.startswith("torch") or name == "PreTrainedModel":
            return "pt"
        elif module.startswith("flax") or module.startswith("jax") or name == "FlaxPreTrainedModel":
            return "flax"
    raise TypeError(f"Could not infer framework from class {model_class}.")


def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    if not is_torch_available():
        return int(x)

    import torch

    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


def torch_float(x):
    """
    Casts an input to a torch float32 tensor if we are in a tracing context, otherwise to a Python float.
    """
    if not is_torch_available():
        return int(x)

    import torch

    return x.to(torch.float32) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


def filter_out_non_signature_kwargs(extra: Optional[list] = None):
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
    """

    num_items_in_batch: Optional["torch.Tensor"]
    output_hidden_states: Optional[bool]
    output_attentions: Optional[bool]
    output_router_logits: Optional[bool]
    cu_seq_lens_q: Optional["torch.LongTensor"]
    cu_seq_lens_k: Optional["torch.LongTensor"]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


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


def set_attribute_for_modules(module: "torch.nn.Module", key: str, value: Any):
    """
    Set a value to a module and all submodules.
    """
    setattr(module, key, value)
    for submodule in module.children():
        set_attribute_for_modules(submodule, key, value)


def del_attribute_from_modules(module: "torch.nn.Module", key: str):
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


# if is_torch_available():
# @torch._dynamo.disable
@dataclass
@requires(backends=("torch",))
class OutputRecorder:
    """
    Configuration for recording outputs from a model via hooks.

    Attributes:
        target_class (Type): The class (e.g., nn.Module) to which the hook will be attached.
        index (Optional[int]): If the output is a tuple/list, optionally record only at a specific index.
        layer_name (Optional[str]): Name of the submodule to target (if needed), e.g., "transformer.layer.3.attn".
        class_name (Optional[str]): Name of the class to which the hook will be attached. Could be the suffix of class name in some cases.
    """

    target_class: "type[torch.nn.Module]"
    index: int = 0
    layer_name: Optional[str] = None
    class_name: Optional[str] = None


def check_model_inputs(func):
    """
    Decorator to intercept specific layer outputs without using hooks.
    Compatible with torch.compile (Dynamo tracing).
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        use_cache = (
            kwargs["use_cache"] if kwargs.get("use_cache") is not None else getattr(self.config, "use_cache", None)
        )
        if use_cache is not None:
            if getattr(self, "gradient_checkpointing", False) and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            kwargs["use_cache"] = use_cache

        return_dict = kwargs.pop("return_dict", None)
        if return_dict is None:
            return_dict = getattr(self.config, "return_dict", True)

        all_args = kwargs.copy()
        if "kwargs" in all_args:
            for k, v in all_args["kwargs"].items():
                all_args[k] = v

        capture_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__), {})  # there is a weak ref for executorch
        recordable_keys = {
            f"output_{k}": all_args.get(
                f"output_{k}",
                getattr(
                    self.config,
                    f"output_{k}",
                    all_args.get("output_attentions", getattr(self.config, "output_attentions", False)),
                ),
            )
            for k in capture_flags
        }

        # We let cross attentions to be saved separately because some models add `cross-attn` layer
        # when certain conditions are met. Let's output cross attention if attentions are requested (for BC)
        if "output_attentions" in recordable_keys:
            recordable_keys["output_cross_attentions"] = recordable_keys["output_attentions"]

        collected_outputs = defaultdict(tuple)
        monkey_patched_layers = []

        # Check attention implementation is properly set for capturing attention outputs
        if recordable_keys.get("output_attentions", False):
            supported_attn = ["eager", "eager_paged", "flex_attention"]
            config_attn = getattr(self.config, "_attn_implementation", None)
            sub_configs = [getattr(self.config, key, None) for key in self.config.sub_configs]
            sub_configs_attn = [
                getattr(config, "_attn_implementation", None) for config in sub_configs if config is not None
            ]
            if config_attn not in supported_attn or any(attn not in supported_attn for attn in sub_configs_attn):
                warnings.warn(
                    f"`output_attentions=True` is not supported with `attn_implementation` other than {supported_attn}. "
                    "Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.",
                    UserWarning,
                )

        def make_capture_wrapper(module, orig_forward, key, index):
            @wraps(orig_forward)
            def wrapped_forward(*args, **kwargs):
                if key == "hidden_states" and len(collected_outputs[key]) == 0:
                    collected_outputs[key] += (args[0],)
                if kwargs.get("debug_io", False):
                    with model_addition_debugger_context(
                        module, kwargs.get("debug_io_dir", "~/model_debug"), kwargs.get("prune_layers")
                    ):
                        output = orig_forward(*args, **kwargs)
                else:
                    output = orig_forward(*args, **kwargs)
                if not isinstance(output, tuple):
                    collected_outputs[key] += (output,)
                elif output[index] is not None:
                    if key not in collected_outputs:
                        collected_outputs[key] = (output[index],)
                    else:
                        collected_outputs[key] += (output[index],)
                return output

            return wrapped_forward

        if any(recordable_keys.values()):
            capture_tasks = []
            for key, layer_specs in capture_flags.items():
                if not recordable_keys.get(f"output_{key}", False):
                    continue
                if not isinstance(layer_specs, list):
                    layer_specs = [layer_specs]
                for specs in layer_specs:
                    if not isinstance(specs, OutputRecorder):
                        index = 0 if "hidden_states" in key else 1
                        class_name = None if not isinstance(specs, str) else specs
                        target_class = specs if not isinstance(specs, str) else None
                        specs = OutputRecorder(target_class=target_class, index=index, class_name=class_name)
                    capture_tasks.append((key, specs))

            for name, module in self.named_modules():
                for key, specs in capture_tasks:
                    # The second check is for multimodals where only backbone layer suffix is available
                    if (specs.target_class is not None and isinstance(module, specs.target_class)) or (
                        specs.class_name is not None and name.endswith(specs.class_name)
                    ):
                        if specs.layer_name is not None and specs.layer_name not in name:
                            continue
                        # Monkey patch forward
                        original_forward = module.forward
                        module.forward = make_capture_wrapper(module, original_forward, key, specs.index)
                        monkey_patched_layers.append((module, original_forward))

        try:
            outputs = func(self, *args, **kwargs)
        except TypeError as original_exception:
            # If we get a TypeError, it's possible that the model is not receiving the recordable kwargs correctly.
            # Get a TypeError even after removing the recordable kwargs -> re-raise the original exception
            # Otherwise -> we're probably missing `**kwargs` in the decorated function
            kwargs_without_recordable = {k: v for k, v in kwargs.items() if k not in recordable_keys}
            try:
                outputs = func(self, *args, **kwargs_without_recordable)
            except TypeError:
                raise original_exception
            raise TypeError(
                "Missing `**kwargs` in the signature of the `@check_model_inputs`-decorated function "
                f"({func.__qualname__})"
            )

        # Restore original forward methods
        for module, original_forward in monkey_patched_layers:
            module.forward = original_forward

        # Inject collected outputs into model output
        for key in collected_outputs:
            if key == "hidden_states":
                if hasattr(outputs, "vision_hidden_states"):
                    collected_outputs[key] = collected_outputs[key][:-1]
                    collected_outputs[key] += (outputs.vision_hidden_states,)
                elif hasattr(outputs, "last_hidden_state"):
                    collected_outputs[key] = collected_outputs[key][:-1]
                    collected_outputs[key] += (outputs.last_hidden_state,)

                outputs[key] = collected_outputs[key]
            elif key == "attentions":
                if isinstance(capture_flags[key], list) and len(capture_flags[key]) == 2:
                    outputs[key] = collected_outputs[key][0::2]
                    outputs["cross_" + key] = collected_outputs[key][1::2]
                else:
                    outputs[key] = collected_outputs[key]
            else:
                outputs[key] = collected_outputs[key]
        if return_dict is False:
            outputs = outputs.to_tuple()
        return outputs

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
