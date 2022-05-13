# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import builtins
import collections
import functools
import inspect
import math
import random
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

import torch
from packaging import version
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer

from .. import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    GPT2DoubleHeadsModel,
    PretrainedConfig,
    PreTrainedModel,
    XLNetForQuestionAnswering,
    logging,
)
from ..models.auto import get_values
from ..utils import TORCH_FX_REQUIRED_VERSION, is_torch_fx_available
from ..utils.versions import importlib_metadata


logger = logging.get_logger(__name__)


def _generate_supported_model_classes(
    model_name: Type[PretrainedConfig],
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[Type[PreTrainedModel]]:

    model_config_class = CONFIG_MAPPING[model_name]
    task_mapping = {
        "default": MODEL_MAPPING,
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING,
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING,
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    }

    if supported_tasks is None:
        supported_tasks = task_mapping.keys()
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]

    model_classes = []
    for task in supported_tasks:
        model_class = task_mapping[task].get(model_config_class, None)
        if model_class:
            model_classes.append(model_class)

    return model_classes


_REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS = [
    "albert",
    "bert",
    "distilbert",
    "mobilebert",
    "electra",
    "megatron-bert",
    "gpt2",
    "gptj",
    "gpt_neo",
    "t5",
    "roberta",
    "vit",
    "swin",
    # TODO: add support for them as it should be quite easy to do so (small blocking issues).
    # "layoutlm",
    # "xlnet",
]

_REGULAR_SUPPORTED_MODELS = []
for item in _REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS:
    if isinstance(item, dict):
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_classes(**item))
    else:
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_classes(item))

_SPECIAL_SUPPORTED_MODELS = [
    GPT2DoubleHeadsModel,
    # TODO: add support for them as it should be quite easy to do so (small blocking issues).
    # XLNetForQuestionAnswering,
]
_SUPPORTED_MODELS = tuple(
    sorted(list(set(_REGULAR_SUPPORTED_MODELS + _SPECIAL_SUPPORTED_MODELS)), key=lambda c: c.__name__)
)


def embedding_override(self, input):
    return torch.empty(*input.shape, self.weight.shape[-1], device="meta")


def torch_nn_layernorm_override(self, input):
    return input


def torch_nn_linear_override(self, input):
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")


def torch_relu_override(x):
    return x


def torch_nn_relu_override(self, x):
    return x


def torch_nn_functional_relu_override(x, inplace=False):
    if not inplace:
        raise ValueError("Don't support in-place functional.relu for MetaTensor analysis")
    return x


def torch_where_override(condition, x, y):
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


def torch_abs_override(input, *, out=None):
    if out is None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    return input


def torch_arange_override(*args, **kwargs):
    n = len(args)
    step = 1
    if n == 1:
        start = 0
        end = args[0]
    elif n == 2:
        start, end = args
    else:
        start, end, step = args
    step = kwargs.get("step", step)
    dtype = kwargs.get("dtype")
    return torch.empty((end - start) // step, dtype=dtype, device="meta")


def torch_cat_override(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + dim
    shapes = [t.shape for t in tensors]
    shape = list(shapes[0])
    concatenated_dim = sum(shape[dim] for shape in shapes)
    final_shape = shape[:dim] + [concatenated_dim] + shape[dim + 1 :]
    return torch.empty(final_shape, device="meta")


def torch_stack_override(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + 1 + dim
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    return torch.empty(shape, device="meta")


def torch_add_override(input, other, *, alpha=1, out=None):
    if not isinstance(input, torch.Tensor):
        return torch.empty_like(other, device="meta")
    if not isinstance(other, torch.Tensor):
        return torch.empty_like(input, device="meta")
    max_length = max(input.dim(), other.dim())
    input_shape = list(input.shape) + [1] * (max_length - input.dim())
    other_shape = list(other.shape) + [1] * (max_length - other.dim())
    shape = []
    for i in range(max_length):
        shape.append(max(input_shape[i], other_shape[i]))
    return torch.empty(shape, device="meta")


def torch_mul_override(input, other, *, out=None):
    return torch_add_override(input, other, out=out)


def torch_tensor_mul_override(self, other):
    return torch_mul_override(self, other)


def torch_matmul_override(input, other, *, out=None):
    d1 = input.dim()
    d2 = other.dim()
    shape = None
    if d1 == 1 and d2 == 1:
        shape = None
    elif d1 == 2 and d2 == 2:
        shape = (input.size(0), other.size(1))
    elif d1 == 1 and d2 == 2:
        shape = (other.size(1),)
    elif d1 == 2 and d1 == 1:
        shape = (input.size(0),)
    else:
        max_length = max(input.dim(), other.dim())
        shape1 = list(input.shape)
        shape2 = list(other.shape)
        if d1 == 1:
            shape1 = [1] + shape1
        if d2 == 1:
            shape2.append(1)
        shape1 = [-1] * (max_length - d1) + list(input.shape)
        shape2 = [-1] * (max_length - d2) + list(other.shape)
        shape = []
        for i in range(max_length):
            shape.append(max(shape1[i], shape2[i]))
        shape[-2] = shape1[-2]
        shape[-1] = shape2[-1]
        if d1 == 1:
            shape.pop(-2)
        if d2 == 1:
            shape.pop(-1)
    if shape is None:
        return torch.tensor(0.0, device="meta")
    return torch.empty(*shape, device="meta")


def torch_tensor_repeat_override(self, *sizes):
    shape = list(self.shape)
    for i, x in enumerate(sizes):
        shape[i] *= x
    return torch.empty(shape, device="meta")


def torch_index_select(input, dim, index, *, out=None):
    shape = list(input.shape)
    shape[dim] = len(index)
    return torch.empty(*shape, device="meta")


def torch_tensor_index_select(self, dim, index):
    return torch_tensor_index_select(self, dim, index)


def torch_roll(input, shifts, dims=None):
    return input


def torch_nn_conv2d(self, input):
    h_in, w_in = input.shape[-2:]
    shape = None
    padding = self.padding
    if padding == "valid":
        padding = (0, 0)
    if padding == "same":
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        h_out = math.floor(
            (h_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        w_out = math.floor(
            (w_in + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        )
        shape[-2:] = [h_out, w_out]
    shape[-3] = self.out_channels
    return torch.empty(shape, device="meta")


def torch_nn_mseloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device="meta")


def torch_nn_crossentropyloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device="meta")


def torch_nn_bcewithlogitsloss(self, input, target):
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    return torch.empty(shape, device="meta")


_MANUAL_META_OVERRIDES: Dict[Callable, Callable] = {
    torch.nn.Embedding: embedding_override,
    torch.nn.LayerNorm: torch_nn_layernorm_override,
    torch.nn.Linear: torch_nn_linear_override,
    torch.relu: torch_relu_override,
    torch.nn.functional.relu: torch_nn_functional_relu_override,
    torch.nn.ReLU: torch_nn_relu_override,
    torch.where: torch_where_override,
    torch.abs: torch_abs_override,
    torch.arange: torch_arange_override,
    torch.cat: torch_cat_override,
    torch.stack: torch_stack_override,
    torch.add: torch_add_override,
    torch.mul: torch_mul_override,
    torch.Tensor.mul: torch_tensor_mul_override,
    torch.matmul: torch_matmul_override,
    torch.Tensor.repeat: torch_tensor_repeat_override,
    torch.roll: torch_roll,
    # TODO: those might not be needed.
    # torch.index_select: torch_index_select,
    # torch.Tensor.index_select: torch_tensor_index_select,
    torch.nn.Conv2d: torch_nn_conv2d,
    torch.nn.MSELoss: torch_nn_mseloss,
    torch.nn.CrossEntropyLoss: torch_nn_crossentropyloss,
    torch.nn.BCEWithLogitsLoss: torch_nn_bcewithlogitsloss,
}


class HFProxy(Proxy):
    """
    Proxy that uses metadata to handle data-dependent control-flow.
    """

    def install_metadata(self, metadata):
        self._metadata = metadata

    @property
    def shape(self):
        return self.tracer.create_proxy("call_method", "size", (self,), {})

    @property
    def dtype(self):
        return self.tracer.root.dtype
        if hasattr(self, "_metadata") and self._metadata is not None:
            return self._metadata.dtype
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "dtype"), {})

    @property
    def device(self):
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    def __len__(self):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return len(self._metadata)
        return super().__len__()

    def __bool__(self):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return self._metadata
        return super().__bool__()

    def __getattr__(self, k):
        if k == "_metadata":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return HFAttribute(self, k)

    def __setitem__(self, indices, values):
        return self.tracer.create_proxy("call_method", "__setitem__", (self, indices, values), {})

    def __contains__(self, key):
        # To handle cases such as :
        # `"some_key" in kwargs`
        if self.node.op == "placeholder":
            return False
        return super().__contains__(key)


class HFAttribute(HFProxy):
    def __init__(self, root, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy("call_function", getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)


class MetaDeviceAttribute(HFAttribute):
    pass


def _proxies_to_metas(v):
    """Returns the underlying metadata for HFProxies, and behaves like the identity for the others."""
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        if not (isinstance(v, HFProxy) and hasattr(v, "_metadata")):
            raise RuntimeError(f"No metadata was found for {v}")
        return v._metadata
    return v


def _gen_constructor_wrapper(target):
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


def _generate_random_int(low: int = 10, high: int = 20, forbidden_values: Optional[List[int]] = None):
    if forbidden_values is None:
        forbidden_values = []
    value = random.randint(low, high)
    while value in forbidden_values:
        value = random.randint(low, high)
    return value


class HFTracer(Tracer):
    """
    Tracer that is able to symbolically trace models from the library. To do that, it uses the HFProxy instead of the
    regular PyTorch torch.fx.Proxy.
    """

    allow_insert_stateless_mods: bool = True
    _TORCH_METHODS_TO_PATCH = ["arange", "zeros", "ones", "full_like", "eye"]

    def __init__(self, autowrap_modules=(math,), autowrap_functions=(), enable_cpatching=False):

        super().__init__(
            autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions, enable_cpatching=enable_cpatching
        )

        if not is_torch_fx_available():
            torch_version = version.parse(importlib_metadata.version("torch"))
            raise ImportError(
                f"Found an incompatible version of torch. Found version {torch_version}, but only version "
                f"{TORCH_FX_REQUIRED_VERSION} is supported."
            )

    def _generate_dummy_input(
        self, model: PreTrainedModel, input_name: str, shape: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Generates dummy input for model inference recording."""
        model_class = model.__class__
        device = model.device
        inputs_dict = {}

        if input_name in ["labels", "start_positions", "end_positions"]:

            batch_size = shape[0]
            if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING),
                XLNetForQuestionAnswering,
            ]:
                inputs_dict["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
                inputs_dict["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING):
                if not hasattr(model.config, "problem_type") or model.config.problem_type is None:
                    raise ValueError(
                        "Could not retrieve the problem type for the sequence classification task, please set "
                        'model.config.problem_type to one of the following values: "regression", '
                        '"single_label_classification", or "multi_label_classification".'
                    )

                if model.config.problem_type == "regression":
                    labels_shape = (batch_size, model.config.num_labels)
                    labels_dtype = torch.float32
                elif model.config.problem_type == "single_label_classification":
                    labels_shape = (batch_size,)
                    labels_dtype = torch.long
                elif model.config.problem_type == "multi_label_classification":
                    labels_shape = (batch_size, model.config.num_labels)
                    labels_dtype = torch.float32
                else:
                    raise ValueError(
                        'Expected model.config.problem_type to be either: "regression", "single_label_classification"'
                        f', or "multi_label_classification", but "{model.config.problem_type}" was provided.'
                    )
                inputs_dict["labels"] = torch.zeros(*labels_shape, dtype=labels_dtype, device=device)

            elif model_class in [
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class in [
                *get_values(MODEL_FOR_PRETRAINING_MAPPING),
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
                GPT2DoubleHeadsModel,
            ]:
                inputs_dict["labels"] = torch.zeros(shape, dtype=torch.long, device=device)
            else:
                raise NotImplementedError(f"{model_class} not supported yet.")
        elif "pixel_values" in input_name:
            batch_size = shape[0]
            image_size = model.config.image_size
            if not isinstance(image_size, collections.abc.Iterable):
                image_size = (image_size, image_size)
            height, width = image_size
            inputs_dict[input_name] = torch.zeros(
                batch_size, model.config.num_channels, height, width, dtype=torch.float32, device=device
            )

        elif "mask" in input_name or "ids" in input_name:
            inputs_dict[input_name] = torch.zeros(shape, dtype=torch.long, device=device)
        else:
            shape_with_hidden_size = shape + [model.config.hidden_size]
            inputs_dict[input_name] = torch.zeros(shape_with_hidden_size, dtype=torch.float, device=device)

        return inputs_dict

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        if kind == "placeholder" and target in self.meta_args:
            rv.install_metadata(self.meta_args[target])
            return rv

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)

            if kind == "call_function":
                meta_target = _MANUAL_META_OVERRIDES.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_method":
                method = getattr(args_metas[0].__class__, target)
                meta_target = _MANUAL_META_OVERRIDES.get(method, method)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AttributeError(f"{self} does not have an attribute called orig_forward")
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in _MANUAL_META_OVERRIDES:
                        meta_out = _MANUAL_META_OVERRIDES[mod_type](mod, *args_metas, **kwargs_metas)
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if isinstance(attr_itr, torch.Tensor):
                        meta_out = attr_itr.to(device="meta")
                    else:
                        meta_out = attr_itr
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            if not isinstance(rv, Proxy):
                raise ValueError("Don't support composite output yet")
            rv.install_metadata(meta_out)
        except Exception as e:
            warnings.warn(f"Could not compute metadata for {kind} target {target}: {e}")

        return rv

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:
            return super()._module_getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def proxy(self, node):
        return HFProxy(node, self)

    def trace(
        self,
        root: PreTrainedModel,
        concrete_args: Optional[Dict[str, Any]] = None,
        method_names: Optional[Iterable[str]] = None,
    ) -> Graph:

        if concrete_args is None:
            concrete_args = {}

        sig = inspect.signature(root.forward)
        input_names = sig.parameters.keys() - concrete_args.keys()

        # Creating a random input shape to generate dummy inputs.
        batch_size = _generate_random_int()
        sequence_length = _generate_random_int()
        shape = [batch_size, sequence_length]

        if root.__class__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            num_choices = _generate_random_int(low=2, high=5)
            shape.insert(1, num_choices)

        inputs = {}
        for input_name in input_names:
            inputs.update(self._generate_dummy_input(root, input_name, shape))

        concrete_metas = {input_name: input_.to("meta") for input_name, input_ in inputs.items()}
        self.meta_args = concrete_metas
        self.patched_torch_methods = {
            target: _gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            self.graph = super().trace(root, concrete_args=concrete_args)
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)

        # TODO: keep this until necessary.
        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        # A PR that solves this was posted: https://github.com/pytorch/pytorch/pull/59569 but it was not merged yet.
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in input_names:
                    node.args = ()
                # It is a concrete arg so it is not used and should be removed.
                else:
                    self.graph.erase_node(node)

            # TODO: solves GraphModule creation.
            # Without this, return type annotation "Tuple" is causing code execution failure.
            if node.op == "output":
                node.type = None

        return self.graph

    def _stateless_mod_instanciation_depends_on_proxies(self, mod: nn.Module) -> bool:
        """
        Whether the module was instantiated with Proxies. If that is the case, such module cannot be a leaf module
        because its attributes are input-dependent.
        """
        return any(isinstance(attr, Proxy) for attr in mod.__dict__.values())

    def _insert_module_as_submodule(self, mod: nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        # If one of the module attributes is a Proxy, it means that its instantiation is input-dependent.
        # It is not possible to insert such modules, those should be traced through.
        if self._stateless_mod_instanciation_depends_on_proxies(mod):
            return ""
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        already_inserted = False
        while hasattr(self.root, path):
            if getattr(self.root, path) is mod:
                already_inserted = True
                break
            path = f"{mod_name}_{idx}"
            idx += 1

        # No need to add multiple instances of the same module.
        if not already_inserted:
            self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: nn.Module) -> str:
        """
        Helper method to find the qualified name of `mod` in the Module hierarchy of `root`. For example, if `root` has
        a submodule named `foo`, which has a submodule named `bar`, passing `bar` into this function will return the
        string "foo.bar".

        Args:
            mod (str): The `Module` to retrieve the qualified name for.
        """
        try:
            return super().path_of_module(mod)
        except NameError as e:
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and len(list(mod.buffers())) == 0:
                path = self._insert_module_as_submodule(mod)
                return path
            raise e

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (not self._stateless_mod_instanciation_depends_on_proxies(m)) and super().is_leaf_module(
            m, module_qualified_name
        )


def symbolic_trace(
    model: PreTrainedModel,
    input_names: Optional[List[str]] = None,
) -> GraphModule:

    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example:

        ```python
        from transformers.utils.fx import symbolic_trace

        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    if not isinstance(model, _SUPPORTED_MODELS):
        supported_model_names = ", ".join((cls.__name__ for cls in _SUPPORTED_MODELS))
        raise NotImplementedError(
            f"Model {model.__class__.__name__} is not supported yet, supported models: {supported_model_names}"
        )

    # Tracing.
    tracer = HFTracer()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    return traced
