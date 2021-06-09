import copy
import functools
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
from packaging import version
from torch import nn
from torch.fx import Graph, GraphModule, Node, Proxy, Tracer
from torch.fx.node import Argument

from transformers.file_utils import TORCH_FX_REQUIRED_VERSION, importlib_metadata, is_torch_fx_available

from .. import (
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
    GPT2DoubleHeadsModel,
    PreTrainedModel,
    logging,
)
from ..models.auto import get_values


logger = logging.get_logger(__name__)


class HFProxy(Proxy):
    """
    Proxy that is able to provide the proper ranks, shapes and boolean values during symbolic tracing by implementing
    the dim, size and __bool__ methods. It can be easily extended by either adding new methods or extending the
    existing ones.
    """

    def __init__(self, node: Node, tracer: Optional[Tracer] = None):
        super().__init__(node, tracer=tracer)
        if hasattr(self, "tracer") and self.tracer is not None:
            self.device = self.tracer.root.device
            self.dtype = next(self.tracer.root.parameters()).dtype

    @property
    def shape(self):
        return self.size()

    def __setitem__(self, key, value):
        pass

    def __contains__(self, key):
        return False


def _wrap_method_for_model_recording(model, method_name, cache_name):
    """Helper function that wraps a torch.Tensor method to record its outputs during forward pass."""
    method = getattr(torch.Tensor, method_name)

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        if not hasattr(model, cache_name):
            setattr(model, cache_name, [])
        cache = getattr(model, cache_name)
        res = method(*args, **kwargs)
        cache.append(res)
        return res

    return wrapped


def _create_recorded_proxy_method(proxy, method_name, cache_name):
    """
    Helper function that sets a recorded torch.Tensor method as a HFProxy method that will use the recorded values
    during symbolic tracing.
    """

    def method(self, *args, **kwargs):
        cache = getattr(self.tracer.root, cache_name)
        res = cache.pop(0)
        return res

    method.__name__ = method_name
    bound_method = method.__get__(proxy, proxy.__class__)
    setattr(proxy, method_name, bound_method)


def _wrap_method_for_model_tracing(model, method_name, cache_name):
    """
    Helper function that sets a recorded torch.Tensor method as a torch.Tensor method that will use the recorded values
    during symbolic tracing.
    """

    original_method = getattr(torch.Tensor, method_name)

    @functools.wraps(original_method)
    def method(*args, **kwargs):
        cache = getattr(model, cache_name)
        res = cache.pop(0)
        return res

    setattr(torch.Tensor, method_name, method)

    if method_name == "size":
        setattr(torch.Tensor, "shape", property(getattr(torch.Tensor, method_name)))


def _monkey_patch_tensor_methods_for_model_recording(model, method_names):
    """
    Helper function that patches torch.Tensor methods (specified by the method_names list) to record model inference
    before symbolic tracing.
    """
    cache_names = dict()
    original_methods = dict()
    for method_name in method_names:
        cache_name = f"cache_{method_name}"
        cache_names[method_name] = cache_name
        if not hasattr(torch.Tensor, method_name):
            logger.info(f"torch.Tensor has no method called {method_name}, skipping patching.")
            continue
        original_methods[method_name] = getattr(torch.Tensor, method_name)
        setattr(torch.Tensor, method_name, _wrap_method_for_model_recording(model, method_name, cache_name))

        if method_name == "size":
            original_methods["shape"] = torch.Tensor.shape
            setattr(torch.Tensor, "shape", property(getattr(torch.Tensor, method_name)))

    return cache_names, original_methods


def _reset_tensor_methods(original_methods):
    """Helper function that resets the monkey patched torch.Tensor methods to their original values."""
    for name, method in original_methods.items():
        setattr(torch.Tensor, name, method)


class HFTracer(Tracer):
    """
    Tracer that is able to symbolically trace models from the library. To do that, it uses the HFProxy instead of the
    regular PyTorch torch.fx.Proxy.
    """

    default_methods_to_record = {"__bool__", "size", "dim"}

    def __init__(self, batch_size=1, sequence_length=[128, 128], num_choices=-1):
        super().__init__()

        if not is_torch_fx_available():
            torch_version = version.parse(importlib_metadata.version("torch"))
            raise ImportError(
                f"Found an incompatible version of torch. Found version {torch_version}, but only version "
                f"{TORCH_FX_REQUIRED_VERSION} is supported."
            )

        encoder_sequence_length = sequence_length[0] if isinstance(sequence_length, (list, tuple)) else sequence_length
        decoder_sequence_length = (
            sequence_length[1] if isinstance(sequence_length, (list, tuple)) else encoder_sequence_length
        )
        self.encoder_shape = [batch_size, encoder_sequence_length]
        self.decoder_shape = (
            [batch_size, decoder_sequence_length] if decoder_sequence_length > 0 else list(self.encoder_shape)
        )
        self.num_choices = num_choices
        if self.num_choices > 0:
            self.encoder_shape = [batch_size, self.num_choices, encoder_sequence_length]
            self.decoder_shape = [batch_size, self.num_choices, decoder_sequence_length]

        self.prev_module = None
        self.recorded_methods = None

    def proxy(self, node: Node):
        p = HFProxy(node, self)
        if self.recorded_methods:
            for method_name, cache_name in self.recorded_methods.items():
                _create_recorded_proxy_method(p, method_name, cache_name)
        return p

    def _generate_dummy_input(self, model, input_name):
        """Generates dummy input for model inference recording."""
        model_class = model.__class__
        device = model.device
        inputs_dict = dict()

        if input_name in ["labels", "start_positions", "end_positions"]:
            batch_size = self.encoder_shape[0]
            if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = torch.ones(batch_size, dtype=torch.long, device=device)
            elif model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
                inputs_dict["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
                GPT2DoubleHeadsModel,
            ]:
                inputs_dict["labels"] = torch.zeros(self.decoder_shape, dtype=torch.long, device=device)
            elif model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(self.encoder_shape, dtype=torch.long, device=device)
            else:
                raise NotImplementedError(f"{model_class} not supported yet.")

        elif "mask" in input_name or "ids" in input_name:
            shape = self.encoder_shape if "decoder" not in input_name else self.decoder_shape
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.long, device=device)
        else:
            shape = self.encoder_shape if "decoder" not in input_name else self.decoder_shape
            shape += [model.config.hidden_size]
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.float, device=device)

        return inputs_dict

    def record(self, model, input_names, method_names=None):
        """
        Records torch.Tensor method outputs (specified by the method_names list) that will then be used during symbolic
        tracing.
        """
        if method_names is None:
            method_names = self.default_methods_to_record

        inputs = dict()
        for input_name in input_names:
            inputs.update(self._generate_dummy_input(model, input_name))

        clone = copy.deepcopy(model)
        cache_names, original_methods = _monkey_patch_tensor_methods_for_model_recording(clone, method_names)
        self.original_methods = original_methods

        clone(**inputs)

        # Useful because sometime the config is changed at inference time, for instance for
        # classification tasks where config.problem_type can be set.
        model.config = clone.config

        _reset_tensor_methods(original_methods)

        self.recorded_methods = {
            method_name: cache_name for method_name, cache_name in cache_names.items() if hasattr(clone, cache_name)
        }

        for cache_name in self.recorded_methods.values():
            setattr(model, cache_name, getattr(clone, cache_name))

    def trace(self, root: PreTrainedModel, concrete_args: Optional[Dict[str, Any]] = None, method_names=None) -> Graph:
        sig = inspect.signature(root.forward)
        input_names = sig.parameters.keys() - concrete_args.keys()

        self.record(root, input_names, method_names=method_names)

        for method_name, cache_name in self.recorded_methods.items():
            _wrap_method_for_model_tracing(root, method_name, cache_name)

        graph = super().trace(root, concrete_args=concrete_args)

        _reset_tensor_methods(self.original_methods)

        return graph

    def _insert_module_as_submodule(self, mod):
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
        Helper method to find the qualified name of ``mod`` in the Module hierarchy of ``root``. For example, if
        ``root`` has a submodule named ``foo``, which has a submodule named ``bar``, passing ``bar`` into this function
        will return the string "foo.bar".

        Args:
            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        # Prefer the O(1) algorithm
        if hasattr(self, "submodule_paths") and self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                path = self._insert_module_as_submodule(mod)
            if path is None:
                raise NameError("module is not installed as a submodule")
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
                raise NameError("module is not installed as a submodule")
            self.prev_module = path
            return path

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, range):
            return super().create_arg(list(a))
        return super().create_arg(a)


def symbolic_trace(
    model: PreTrainedModel,
    input_names: Optional[List[str]] = None,
    batch_size: int = 1,
    sequence_length: Union[int, List[int]] = [128, 128],
    num_choices: int = -1,
) -> GraphModule:

    """
    Performs symbolic tracing on the model.

    Args:
        model (:obj:`PretrainedModel`):
            The model to trace.
        input_names (:obj:`List[str]`, `optional`):
            The names of the inputs of the traced model. If unset, model.dummy_inputs().keys() are used instead.
        batch_size (:obj:`int`, `optional`, defaults to 1):
            The batch size of the traced model inputs.
        sequence_length (:obj:`int` or :obj:`List[int]]`):
            The sequence length of the traced model inputs. For sequence-to-sequence models with different sequence
            lengths between the encoder and the decoder inputs, this must be :obj:`[encoder_sequence_length,
            decoder_sequence_length]`.
        num_choices (:obj:`int`, `optional`, defaults to -1):
            The number of possible choices for a multiple choice task.

    Returns:
        :obj:`torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example::

        from transformers.utils.fx import symbolic_trace
        traced_model = symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            batch_size=1,
            sequence_length=128,
        )
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    sig = inspect.signature(model.forward)
    # TODO: how to handle the case of the "return_dict" parameter.
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    tracer = HFTracer(batch_size=batch_size, sequence_length=sequence_length, num_choices=num_choices)

    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    return traced
