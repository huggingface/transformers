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
import functools
import inspect
import math
import operator
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
    # TODO: make this more efficient by not making concrete computation.
    concrete_input = torch.empty_like(input, device="cpu")
    return torch.empty_like(self.forward(concrete_input), device="meta")


def torch_relu_override(x):
    return x


def torch_nn_relu_override(self, x):
    return x


def functional_relu_override(x, inplace=False):
    assert not inplace, "dont support inplace functional.relu for metatensor analysis"
    return x


def torch_where_override(condition, x, y):
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


def torch_abs_override(input, *, out=None):
    assert out is None, "Dont support in-place abs for MetaTensor analysis"
    return input


def torch_arange_with_start_override(
    start, end, *, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False
):
    return torch.empty((end - start) // step, dtype=dtype, layout=layout, device="meta")


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


def torch_matmul_override(input, other, *, out=None):
    # TODO: make this more efficient by not making concrete computation.
    concrete_input = torch.empty_like(input, device="cpu")
    concrete_other = torch.empty_like(other, device="cpu")
    return torch.empty_like(torch.matmul(concrete_input, concrete_other), device="meta")


def torch_tensor_repeat_override(self, *sizes):
    shape = list(self.shape)
    for i, x in enumerate(sizes):
        shape[i] *= x
    return torch.empty(shape, device="meta")


manual_meta_overrides: Dict[Callable, Callable] = {
    torch.nn.Embedding: embedding_override,
    torch.nn.LayerNorm: torch_nn_layernorm_override,
    torch.nn.Linear: torch_nn_linear_override,
    torch.relu: torch_relu_override,
    torch.nn.functional.relu: functional_relu_override,
    torch.nn.ReLU: torch_nn_relu_override,
    torch.where: torch_where_override,
    torch.abs: torch_abs_override,
    torch.ops.aten.arange: torch_arange_with_start_override,
    torch.cat: torch_cat_override,
    torch.stack: torch_stack_override,
    # torch.add: torch_add_override,
    # torch.mul: torch_mul_override,
    torch.matmul: torch_matmul_override,
    torch.Tensor.repeat: torch_tensor_repeat_override,
}


def gen_constructor_wrapper(target):
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


class HFProxy(Proxy):
    """
    Proxy that uses meta data to handle data-dependent control-flow.
    """

    def install_tensor_meta(self, tensor_meta):
        self._tensor_meta = tensor_meta

    def dim(self):
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dim()
        return self.tracer.create_proxy("call_method", "dim", (self,), {})

    @property
    def shape(self):
        return self.tracer.create_proxy("call_method", "size", (self,), {})

    @property
    def dtype(self):
        return self.tracer.root.dtype
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dtype
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "dtype"), {})

    @property
    def device(self):
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    def __getattr__(self, k):
        if k == "_tensor_meta":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return MetaAttribute(self, k)

    def __len__(self):
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return len(self._tensor_meta)
        return super().__len__()

    def __le__(self, other):
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta <= other
        return self.tracer.create_proxy("call_function", operator.le, (self, other), {})

    def __contains__(self, key):
        # To handle kwargs.
        if self.node.op == "placeholder":
            return False
        return super().__contains__(key)


class MetaAttribute(HFProxy):
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
            proxy = self.tracer.create_proxy("call_function", getattr, (self.root, self.attr), {}).node
            # if hasattr(self.root, "_tensor_meta") and self.root._tensor_meta is not None:
            #     proxy.install_tensor_meta(getattr(self.root._tensor_meta, self.attr))
            # self._node = proxy.node
            self._node = proxy

        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)


class MetaDeviceAttribute(MetaAttribute):
    pass


def proxys_to_metas(v):
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        # assert isinstance(v, HFProxy), f'Expected MetaProxy but got {type(v)}'
        # assert hasattr(v, '_tensor_meta'), 'MetaProxy does not have an associated meta'
        if not (isinstance(v, HFProxy) and hasattr(v, "_tensor_meta")):
            raise RuntimeError(f"No tensor meta data was found for {v}")
        return v._tensor_meta
    return v


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

    from transformers import modeling_utils

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
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
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

        elif "mask" in input_name or "ids" in input_name:
            inputs_dict[input_name] = torch.zeros(shape, dtype=torch.long, device=device)
        else:
            shape_with_hidden_size = shape + [model.config.hidden_size]
            inputs_dict[input_name] = torch.zeros(shape_with_hidden_size, dtype=torch.float, device=device)

        return inputs_dict

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        if kind == "placeholder" and target in self.meta_args:
            rv.install_tensor_meta(self.meta_args[target])
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
            args_metas = torch.fx.node.map_aggregate(args, proxys_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, proxys_to_metas)

            if kind == "call_function":
                meta_target = manual_meta_overrides.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_method":
                method = getattr(args_metas[0].__class__, target)
                meta_target = manual_meta_overrides.get(method, method)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                assert hasattr(self, "orig_forward")
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in manual_meta_overrides:
                        meta_out = manual_meta_overrides[mod_type](mod, *args_metas, **kwargs_metas)
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
                    # assert isinstance(attr_itr, torch.Tensor)
                    if isinstance(attr_itr, torch.Tensor):
                        meta_out = attr_itr.to(device="meta")
                    else:
                        meta_out = attr_itr
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            assert isinstance(rv, Proxy), "Dont support composite output yet"
            rv.install_tensor_meta(meta_out)
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
            target: gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
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

        return self.graph

    def _insert_module_as_submodule(self, mod: nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        while hasattr(self.root, path):
            path = f"{mod_name}_{idx}"
            idx += 1

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
                self.prev_module = path
                return path
            raise e

    # def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
    #     is_loss_module = m.__module__.startswith("torch.nn.modules.loss")
    #     return (not is_loss_module) and super().is_leaf_module(m, module_qualified_name)

    # def create_arg(self, a: Any) -> Argument:
    #     if isinstance(a, range):
    #         return super().create_arg(list(a))
    #     return super().create_arg(a)


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
