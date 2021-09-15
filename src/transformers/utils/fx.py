import copy
import functools
import inspect
import operator
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

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
    MODEL_MAPPING,
    GPT2DoubleHeadsModel,
    PretrainedConfig,
    PreTrainedModel,
    logging,
)
from ..models.albert import AlbertConfig
from ..models.auto import get_values
from ..models.bert import BertConfig
from ..models.distilbert import DistilBertConfig
from ..models.electra import ElectraConfig
from ..models.gpt2 import GPT2Config
from ..models.gpt_neo import GPTNeoConfig
from ..models.gptj import GPTJConfig
from ..models.megatron_bert import MegatronBertConfig
from ..models.mobilebert import MobileBertConfig
from ..models.t5 import T5Config


logger = logging.get_logger(__name__)


T = TypeVar("T")
TypeOrListOfType = Union[T, List[T]]


def _generate_supported_model_classes(
    model_config_class: Type[PretrainedConfig],
    supported_tasks: Optional[TypeOrListOfType[str]] = None,
    other_model_classes: Optional[TypeOrListOfType[Type[PretrainedConfig]]] = None,
):
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

    if other_model_classes:
        if not isinstance(other_model_classes, (tuple, list)):
            other_model_classes = [other_model_classes]
        model_classes.extend(other_model_classes)

    return tuple(model_classes)


_SUPPORTED_MODELS = (
    *_generate_supported_model_classes(AlbertConfig),
    *_generate_supported_model_classes(BertConfig),
    *_generate_supported_model_classes(DistilBertConfig),
    *_generate_supported_model_classes(MobileBertConfig),
    *_generate_supported_model_classes(ElectraConfig),
    *_generate_supported_model_classes(MegatronBertConfig),
    *_generate_supported_model_classes(GPT2Config, other_model_classes=GPT2DoubleHeadsModel),
    *_generate_supported_model_classes(GPTJConfig),
    *_generate_supported_model_classes(GPTNeoConfig),
    *_generate_supported_model_classes(T5Config),
)

_SUPPORTED_MODELS_FOR_DYNAMIC_AXES = (
    *_generate_supported_model_classes(AlbertConfig),
    *_generate_supported_model_classes(BertConfig),
    *_generate_supported_model_classes(DistilBertConfig),
    *_generate_supported_model_classes(MobileBertConfig),
    *_generate_supported_model_classes(ElectraConfig),
    *_generate_supported_model_classes(MegatronBertConfig),
)


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
                raise NameError("Module is not installed as a submodule")
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
                raise NameError("Module is not installed as a submodule")
            self.prev_module = path
            return path

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, range):
            return super().create_arg(list(a))
        return super().create_arg(a)


def _insert_batch_size_node(gm: GraphModule) -> Tuple[GraphModule, Node]:
    """Helper function that inserts a node that retrieves the batch size dynamically from the input of the model."""
    graph = gm.graph
    input_names = gm.dummy_inputs.keys()
    batch_size_node = None
    for node in graph.nodes:
        if node.op == "placeholder" and node.name in input_names:
            with graph.inserting_after(node):
                batch_size_node = graph.call_method("size", args=(node, 0))

    if batch_size_node is None:
        raise ValueError("Could not insert the node that computes the batch size")

    graph.lint()
    gm.recompile()

    return gm, batch_size_node


def _insert_encoder_sequence_length_node(gm: GraphModule) -> Tuple[GraphModule, Node]:
    """
    Helper function that inserts a node that retrieves the encoder sequence length dynamically from the input of the
    model.
    """
    graph = gm.graph
    input_names = gm.dummy_inputs.keys()
    encoder_sequence_length_node = None
    for node in graph.nodes:
        if (node.op == "placeholder") and ("decoder" not in node.name) and (node.name in input_names):
            with graph.inserting_after(node):
                encoder_sequence_length_node = graph.call_method("size", args=(node, 1 if gm.num_choices < 0 else 2))

    if encoder_sequence_length_node is None:
        raise ValueError("Could not insert the node that computes the encoder sequence length")

    graph.lint()
    gm.recompile()

    return gm, encoder_sequence_length_node


def _change_view_methods(gm: GraphModule, mapping: Union[Dict[Node, int], Dict[int, Node]]) -> GraphModule:
    """
    Changes arguments of view ops that refer to static batch size / sequence lengths to make them refer to the
    batch_size / sequence_length nodes.
    """
    graph = gm.graph
    for node in graph.nodes:
        if node.op == "call_method" and node.target == "view":
            if isinstance(node.args[1], tuple):
                node.args = (node.args[0], *node.args[1])
            node.args = tuple((mapping.get(arg, arg) for arg in node.args))
    graph.lint()
    gm.recompile()
    return gm


def _patch_getitem(gm: GraphModule, mapping: Union[Dict[Node, int], Dict[int, Node]]) -> GraphModule:
    # TODO: combine this with the patch_argument function which seems to do almost the same thing.
    graph = gm.graph
    for node in graph.nodes:
        if node.op == "call_function" and node.target == operator.getitem:
            indices = node.args[1]
            if isinstance(indices, tuple):
                new_indices = []
                for idx in indices:
                    if isinstance(idx, slice):
                        new_indices.append(
                            slice(
                                mapping.get(idx.start, idx.start),
                                mapping.get(idx.stop, idx.stop),
                                mapping.get(idx.step, idx.step),
                            )
                        )
                    elif isinstance(idx, int):
                        new_indices.append(mapping.get(idx, idx))
                    else:
                        new_indices.append(idx)

                node.args = (node.args[0], tuple(new_indices))
            else:
                node.args = (node.args[0], mapping.get(node.args[1], node.args[1]))

        graph.lint()
        gm.recompile()

        return gm


def _register_position_ids_and_replace(gm: GraphModule, sequence_length_node: Node) -> GraphModule:
    """
    Redefines position_ids (as tracing with static shapes can introduce optimizations that fix position_ids to a value
    not suitable for dynamic input shapes), and replaces old position_ids usage by the redefined version.
    """

    graph = gm.graph

    any_buffer = next(gm.buffers())
    position_ids = torch.arange(gm.config.max_position_embeddings).expand(1, -1).to(any_buffer.device)
    partial_position_ids = position_ids[:, : gm.static_sequence_length[0]]
    position_ids_buffer_name = None
    for name, buffer in gm.named_buffers():
        if (
            isinstance(buffer, torch.Tensor)
            and partial_position_ids.size() == buffer.size()
            and torch.all(partial_position_ids == buffer)
        ):
            position_ids_buffer_name = name
            gm.register_buffer(name, position_ids)

    inserted = False
    position_ids_node = None
    for node in graph.nodes:
        arg_names = [arg.name if isinstance(arg, Node) else "" for arg in node.args]
        if position_ids_buffer_name in arg_names:
            if not inserted:
                with graph.inserting_before(node):
                    get_position_ids = graph.get_attr(position_ids_buffer_name)
                with graph.inserting_after(get_position_ids):
                    position_ids_args = [
                        get_position_ids,
                        (slice(None, None, None), slice(None, sequence_length_node, None)),
                    ]
                    position_ids_node = graph.call_function(operator.getitem, args=tuple(position_ids_args))
                inserted = True

            index = arg_names.index(position_ids_buffer_name)
            old_position_ids_node = node.args[index]
            old_position_ids_node.replace_all_uses_with(position_ids_node)

    graph.lint()
    gm.recompile()

    return gm


def transform_to_dynamic_input(gm: GraphModule, is_retracing: bool = False) -> GraphModule:
    """Transformation that enables traced models to perform inference on dynamic input shapes."""
    graph = gm.graph
    input_names = gm.dummy_inputs.keys()
    static2dynamic = {}

    # Inserting the nodes that will fetch the batch size and sequence lengths dynamically.
    if gm.use_dynamic_batch_size:
        _, batch_size_node = _insert_batch_size_node(gm)
        static2dynamic[gm.static_batch_size] = batch_size_node
        if gm.num_choices > 0:
            with graph.inserting_after(batch_size_node):
                static2dynamic[gm.static_batch_size * gm.num_choices] = graph.call_function(
                    operator.mul, args=(batch_size_node, gm.num_choices)
                )
            graph.lint()
            gm.recompile()

    if gm.use_dynamic_sequence_length:
        _, encoder_sequence_length_node = _insert_encoder_sequence_length_node(gm)
        static2dynamic[gm.static_sequence_length[0]] = encoder_sequence_length_node

        # TODO: do the same for the decoder.
        pass

    gm = _change_view_methods(gm, static2dynamic)
    gm = _patch_getitem(gm, static2dynamic)

    if (
        gm.use_dynamic_sequence_length
        and "position_ids" not in input_names
        and hasattr(gm.config, "max_position_embeddings")
    ):
        gm = _register_position_ids_and_replace(gm, encoder_sequence_length_node)

    graph.lint()
    gm.recompile()

    gm.static2dynamic = static2dynamic
    gm.dynamic2static = {v: k for (k, v) in static2dynamic.items()}

    return gm


def patch_arguments(gm: GraphModule, mapping: Union[Dict[Node, int], Dict[int, Node]]) -> GraphModule:
    """Helper function that patches node arguments (supports regular types, tuples and slices) using the mapping."""

    def _patch_slice(s, mapping):
        return slice(mapping.get(s.start, s.start), mapping.get(s.stop, s.stop), mapping.get(s.step, s.step))

    graph = gm.graph
    supported_types = (Node, str, int, float)
    for node in graph.nodes:
        new_args = []
        for arg in node.args:
            if isinstance(arg, tuple):
                new_arg = []
                for a in arg:
                    if isinstance(a, slice):
                        print("node", node)
                        new_arg.append(_patch_slice(a, mapping))
                    else:
                        new_arg.append(mapping.get(a, a))
                new_args.append(tuple(new_arg))
            elif isinstance(arg, slice):
                new_args.append(_patch_slice(arg, mapping))
            elif isinstance(arg, supported_types):
                new_args.append(mapping.get(arg, arg))
            else:
                new_args.append(arg)
        node.args = tuple(new_args)

    graph.lint()
    gm.recompile()

    return gm


def prepare_for_retracing(gm: GraphModule) -> Tuple[GraphModule, Dict[str, Any]]:
    """
    Prepares a GraphModule produced by symbolic_trace for retracing by:

        - Caching all the attributes specific to the way the model was initially traced
        - Patching back the model to a "static input shapes" version if it was traced to accept dynamic input shapes
    For instance, the need to retrace a GraphModule can happen when applying quantization.
    """
    attributes_to_keep = [
        "config",
        "num_choices",
        "dummy_inputs",
        "use_dynamic_batch_size",
        "use_dynamic_sequence_length",
        "static_batch_size",
        "static_sequence_length",
        "static2dynamic",
        "dynamic2static",
    ]
    attributes = {k: getattr(gm, k, None) for k in attributes_to_keep}
    gm = patch_arguments(gm, gm.dynamic2static)

    return gm, attributes


def restore_after_retracing(gm: GraphModule, attributes: Dict[str, Any]) -> GraphModule:
    """Restores a GraphModule that was retraced to its initial state in terms of static / dynamic input shapes."""
    for name, attr in attributes.items():
        setattr(gm, name, attr)

    gm = transform_to_dynamic_input(gm, is_retracing=True)
    gm = patch_arguments(gm, gm.static2dynamic)

    return gm


def retrace_graph_with(
    gm: GraphModule, tracer: Tracer = None, func: Callable[[GraphModule], GraphModule] = None
) -> GraphModule:
    """
    Retraces a GraphModule by either using a tracer or a function using a tracer (for instance
    torch.quantization.quantize_fx.prepare_fx). It takes care of preparing the model for retracing, retracing it and
    restoring anything necessary after the retrace.
    """
    if tracer is None and func is None:
        raise ValueError("Either a tracer or a function using a tracer must be provided.")
    elif tracer is not None and func is not None:
        raise ValueError("Either provide a tracer or a function using a tracer, but not both.")
    else:
        gm, attributes = prepare_for_retracing(gm)
        tracing_func = tracer.trace if tracer else func
        traced = tracing_func(gm)
        traced = restore_after_retracing(traced, attributes)
        return traced


def _generate_random_int(low: int = 3, high: int = 100, forbidden_values: Optional[List[int]] = None):
    if forbidden_values is None:
        forbidden_values = []
    value = random.randint(low, high)
    while value in forbidden_values:
        value = random.randint(low, high)
    return value


def symbolic_trace(
    model: PreTrainedModel,
    input_names: Optional[List[str]] = None,
    batch_size: int = 1,
    sequence_length: Union[int, List[int], Tuple[int]] = (128, 128),
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

    # Preparing HFTracer batch_size and sequence_lenght values for potential dynamic axes.
    use_dynamic_batch_size = batch_size <= 0
    if isinstance(sequence_length, (list, tuple)):
        use_dynamic_sequence_length = sequence_length[0] <= 0 or sequence_length[1] <= 0
    else:
        use_dynamic_sequence_length = sequence_length <= 0

    if use_dynamic_batch_size or use_dynamic_sequence_length:
        forbidden_values = [
            model.config.num_attention_heads,
            model.config.hidden_size,
            model.config.hidden_size // model.config.num_attention_heads,
        ]
        if use_dynamic_batch_size:
            batch_size = _generate_random_int(forbidden_values=forbidden_values)
        forbidden_values.append(batch_size)
        if use_dynamic_sequence_length:
            encoder_sequence_length = _generate_random_int(forbidden_values=forbidden_values)
            forbidden_values.append(encoder_sequence_length)
            decoder_sequence_length = _generate_random_int(forbidden_values=forbidden_values)
            sequence_length = [encoder_sequence_length, decoder_sequence_length]

    if not isinstance(model, _SUPPORTED_MODELS):
        supported_model_names = ", ".join((cls.__name__ for cls in _SUPPORTED_MODELS))
        raise NotImplementedError(
            f"Model {model.__class__.__name__} is not supported yet, supported models: {supported_model_names}"
        )
    if (use_dynamic_batch_size or use_dynamic_sequence_length) and not isinstance(
        model, _SUPPORTED_MODELS_FOR_DYNAMIC_AXES
    ):
        supported_model_names = ", ".join((cls.__name__ for cls in _SUPPORTED_MODELS_FOR_DYNAMIC_AXES))
        raise NotImplementedError(
            f"Dynamic axes are not supported for {model.__class__.__name__} yet, supported models: {supported_model_names}"
        )

    # Tracing.
    tracer = HFTracer(batch_size=batch_size, sequence_length=sequence_length, num_choices=num_choices)

    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    traced.config = copy.deepcopy(model.config)
    traced.num_choices = num_choices
    traced.dummy_inputs = {}

    for name in input_names:
        traced.dummy_inputs.update(tracer._generate_dummy_input(model, name))

    traced.use_dynamic_batch_size = use_dynamic_batch_size
    traced.use_dynamic_sequence_length = use_dynamic_sequence_length
    traced.static_batch_size = batch_size
    traced.static_sequence_length = sequence_length

    traced = transform_to_dynamic_input(traced)

    return traced
