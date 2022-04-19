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

import functools
import inspect
import math
import random
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

import torch
from packaging import version
from torch import nn
from torch.fx import Graph, GraphModule, Node, Proxy, Tracer
from torch.fx.node import Argument

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


class HFProxy(Proxy):
    """
    Proxy that uses meta data to handle data-dependent control-flow.
    """

    def __init__(self, node, tracer):
        self._meta_tensor = None
        super().__init__(node, tracer)

    def install_meta_tensor(self, meta_tensor):
        assert (
            isinstance(meta_tensor, torch.Tensor)
            and meta_tensor.device == torch.device("meta")
            or isinstance(meta_tensor, (int, torch.Size))
        )
        self._meta_tensor = meta_tensor

    def dim(self):
        if self._meta_tensor is None:
            return super().__getattr__("dim")
        return self._meta_tensor.dim()


def _function_to_leaf(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper that marks func as a leaf function, meaning that it will not be traced through by HFTracer."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _function_leaf_getter(func_name: str, mapping: Dict[str, Callable[..., Any]]) -> Callable[..., Any]:
    @functools.wraps(mapping[func_name])
    def wrapper(*args, **kwargs):
        return mapping[func_name](*args, **kwargs)

    return wrapper


def _create_recorded_proxy_method(proxy: HFProxy, method_name: str, cache_name: str, return_proxy: bool):
    """
    Helper function that sets a recorded torch.Tensor method as a HFProxy method that will use the recorded values
    during symbolic tracing.
    """

    original_method = getattr(torch.Tensor, method_name)

    @functools.wraps(original_method)
    def method(*args, **kwargs):
        cache = getattr(args[0].tracer.root, cache_name)
        res = cache.pop(0)
        if return_proxy:
            proxy = args[0].__torch_function__(
                original_method,
                None,
                args=args,
                kwargs=kwargs,
            )
            proxy.cache = res
            return proxy
        return res

    method.__name__ = method_name
    bound_method = method.__get__(proxy, proxy.__class__)
    setattr(proxy, method_name, bound_method)


def _reset_tensor_methods(original_methods: Dict[str, Callable[..., Any]]):
    """Helper function that resets the monkey patched torch.Tensor methods to their original values."""
    for name, method in original_methods.items():
        setattr(torch.Tensor, name, method)


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

    _DEFAULT_METHODS_TO_RECORD = {"__bool__": False, "size": True, "dim": False}
    from transformers import modeling_utils

    _FUNCTIONS_TO_AUTOWRAP = {
        torch: {"arange", "zeros", "ones", "full_like", "eye"},
        modeling_utils.ModuleUtilsMixin: {"create_extended_attention_mask_for_decoder"},
    }

    def __init__(self, autowrap_modules=(math,), autowrap_functions=(), enable_cpatching=False):

        # Loading the leaf functions register
        self._leaf_functions_register = {}
        for module, names in self._FUNCTIONS_TO_AUTOWRAP.items():
            for name in names:
                self._register_leaf_function(module, name)

        # TODO: adapt the way leaf function are wrapped with the "autowrap function" feature from Tracer.
        # autowrap_functions = autowrap_functions + tuple(
        #     patched for (_, _, patched) in self._leaf_functions_register.values()
        # )

        super().__init__(
            autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions, enable_cpatching=enable_cpatching
        )

        if not is_torch_fx_available():
            torch_version = version.parse(importlib_metadata.version("torch"))
            raise ImportError(
                f"Found an incompatible version of torch. Found version {torch_version}, but only version "
                f"{TORCH_FX_REQUIRED_VERSION} is supported."
            )

        self.prev_module = None
        self.recorded_methods = None

    def _register_leaf_function(self, module: ModuleType, name: str):
        """Registers the function called name in module as a leaf function."""
        orig_func = getattr(module, name)
        patched_func = _function_to_leaf(orig_func)
        patched_func.__module__ = __name__
        self._leaf_functions_register[name] = (module, orig_func, patched_func)

    def _patch_leaf_functions_for_root(self, root: PreTrainedModel, restore: bool = False):
        """Patches leaf functions specifically for root."""
        for name in self._leaf_functions_register:
            module, orig_func, patched_func = self._leaf_functions_register[name]
            if restore:
                root.__class__.forward.__globals__.pop(name)
                setattr(module, name, orig_func)
            else:
                root.__class__.forward.__globals__[name] = patched_func
                leaf_getter = _function_leaf_getter(name, root.__class__.forward.__globals__)
                leaf_getter.__module__ = __name__
                setattr(module, name, leaf_getter)

    def _method_is_called_in_leaf_module(self, module_ids: List[int]) -> bool:
        """
        Finds out if the method (that is being recorded) is called inside a leaf module, this allows to not record
        outputs that will not be encountered by the tracer.
        """

        currentframe = inspect.currentframe()
        while currentframe:
            if currentframe is None:
                return False
            module = currentframe.f_locals.get("self", None)
            if id(module) in module_ids and self.is_leaf_module(module, "Not used anyway"):
                return True
            currentframe = currentframe.f_back
        return False

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
        if proxy_factory_fn is not None:
            raise RuntimeError("Don't support custom proxy factory function for MetaTensorTracer")

        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, lambda n: HFProxy(n, self))

        def extract_meta(a):
            if isinstance(a, HFProxy):
                if getattr(a, "_meta_tensor", None) is not None:
                    return a._meta_tensor
                else:
                    return None
            return a

        try:
            meta_args = torch.fx.node.map_aggregate(args if args else (), extract_meta)
            meta_kwargs = torch.fx.node.map_aggregate(kwargs if kwargs else {}, extract_meta)

            if kind == "call_function":
                meta_target = target
            elif kind == "call_method":
                assert isinstance(args[0], torch.fx.Proxy)
                meta_target = getattr(torch.Tensor, target)
            elif kind == "call_module":
                raise RuntimeError("Not yet implemented")
            elif kind == "placeholder":
                proxy.install_meta_tensor(next(self.concrete_meta_iter))
                return proxy
            else:
                assert False, f"Unknown target {kind}"

            meta_out = meta_target(*meta_args, **meta_kwargs)

            if isinstance(meta_out, (torch.Tensor, int, torch.Size)):
                proxy.install_meta_tensor(meta_out)

        except Exception as e:
            warnings.warn(f"Could not compute shape for value {proxy}: {e}")

        return proxy

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

        # self.record(root, input_names, method_names=method_names)

        # TODO: adapt the way leaf function are wrapped with the "autowrap function" feature from Tracer.
        autowrap_functions = [patched for (_, _, patched) in self._leaf_functions_register.values()]
        self._autowrap_function_ids.update(set([id(f) for f in autowrap_functions]))

        self._patch_leaf_functions_for_root(root)

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

        concrete_metas = [input_.to("meta") for input_ in inputs.values()]

        self.concrete_metas = concrete_metas
        self.concrete_meta_iter = iter(self.concrete_metas)

        self.graph = super().trace(root, concrete_args=concrete_args)

        self._patch_leaf_functions_for_root(root, restore=True)

        # _reset_tensor_methods(self.original_methods)

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
        # Prefer the O(1) algorithm
        if hasattr(self, "submodule_paths") and self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                path = self._insert_module_as_submodule(mod)
            if path is None:
                raise NameError(f"Module named {mod._get_name()} is not installed as a submodule")
            self.prev_module = path
            return path

        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.
        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    self.prev_module = n
                    return n
            path = self._insert_module_as_submodule(mod)
            if path is None:
                raise NameError(f"Module {mod._get_name()} is not installed as a submodule")
            self.prev_module = path
            return path

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        is_loss_module = m.__module__.startswith("torch.nn.modules.loss")
        return (not is_loss_module) and super().is_leaf_module(m, module_qualified_name)

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, range):
            return super().create_arg(list(a))
        return super().create_arg(a)


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
