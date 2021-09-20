import copy
import functools
import operator
from typing import Any, Dict, Union

import torch
from torch.fx import GraphModule, Node


# Torch FX transformation convention:
#   - transformations that are supposed to act on a copy of the original GraphModule are decorated with @transformation
#   - transformations that are inplace have a name ending with "_"


def _cache_attributes(gm: GraphModule) -> Dict[str, Any]:
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
    return attributes


def _restore_attributes_(gm: GraphModule, attributes: Dict[str, Any]):
    for name, attr in attributes.items():
        if attr:
            setattr(gm, name, attr)


def deepcopy_graph(gm: GraphModule) -> GraphModule:
    """
    Performs a deepcopy of the GraphModule while also copying the relevant attributes to know whether the model was
    traced with dynamic axes, and what were the values if that is the case.
    """
    attributes = _cache_attributes(gm)
    clone = copy.deepcopy(gm)
    _restore_attributes_(clone, attributes)
    return clone


def transformation(func):
    """
    Decorator that wraps a torch.fx transformation by feeding it a copy of the GraphModule to transform instead of the
    original.
    """

    def map_fn(arg):
        if isinstance(arg, GraphModule):
            return deepcopy_graph(arg)
        return arg

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = tuple(map_fn(arg) for arg in args)
        new_kwargs = {k: map_fn(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapper


def _remove_unused_nodes_(gm: GraphModule, lint_and_recompile: bool = True):
    """Removes all the unused nodes in a GraphModule."""
    graph = gm.graph
    for node in graph.nodes:
        if not node.users and node.op != "output":
            print(node.name)
            graph.erase_node(node)

    if lint_and_recompile:
        graph.lint()
        gm.recompile()


def _insert_batch_size_node_(gm: GraphModule, lint_and_recompile: bool = True) -> Node:
    """Inserts a node that retrieves the batch size dynamically from the input of the model."""
    graph = gm.graph
    input_names = gm.dummy_inputs.keys()
    batch_size_node = None
    for node in graph.nodes:
        if node.op == "placeholder" and node.name in input_names:
            with graph.inserting_after(node):
                batch_size_node = graph.call_method("size", args=(node, 0))

    if batch_size_node is None:
        raise ValueError("Could not insert the node that computes the batch size")

    if lint_and_recompile:
        graph.lint()
        gm.recompile()

    # Useful when retracing for quantization.
    if hasattr(gm, "_qconfig_map"):
        gm._qconfig_map[batch_size_node.name] = None

    return batch_size_node


def _insert_encoder_sequence_length_node_(gm: GraphModule, lint_and_recompile: bool = True) -> Node:
    """Inserts a node that retrieves the encoder sequence length dynamically from the input of the model."""
    graph = gm.graph
    input_names = gm.dummy_inputs.keys()
    encoder_sequence_length_node = None
    for node in graph.nodes:
        if (node.op == "placeholder") and ("decoder" not in node.name) and (node.name in input_names):
            with graph.inserting_after(node):
                encoder_sequence_length_node = graph.call_method("size", args=(node, 1 if gm.num_choices < 0 else 2))

    if encoder_sequence_length_node is None:
        raise ValueError("Could not insert the node that computes the encoder sequence length")

    if lint_and_recompile:
        graph.lint()
        gm.recompile()

    # Useful when retracing for quantization.
    if hasattr(gm, "_qconfig_map"):
        gm._qconfig_map[encoder_sequence_length_node.name] = None

    return encoder_sequence_length_node


def _change_view_methods_(
    gm: GraphModule, mapping: Union[Dict[Node, int], Dict[int, Node]], lint_and_recompile: bool = True
):
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

    if lint_and_recompile:
        graph.lint()
        gm.recompile()


def _patch_getitem_(
    gm: GraphModule, mapping: Union[Dict[Node, int], Dict[int, Node]], lint_and_recompile: bool = True
):
    """Patches getitem nodes by replacing current arguments to their corresponding values in mapping."""
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

        if lint_and_recompile:
            graph.lint()
            gm.recompile()


def _register_position_ids_and_replace_(gm: GraphModule, sequence_length_node: Node, lint_and_recompile: bool = True):
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
                    get_position_ids_node = graph.get_attr(position_ids_buffer_name)
                with graph.inserting_after(get_position_ids_node):
                    position_ids_args = [
                        get_position_ids_node,
                        (slice(None, None, None), slice(None, sequence_length_node, None)),
                    ]
                    position_ids_node = graph.call_function(operator.getitem, args=tuple(position_ids_args))
                inserted = True

            index = arg_names.index(position_ids_buffer_name)
            old_position_ids_node = node.args[index]
            old_position_ids_node.replace_all_uses_with(position_ids_node)

    if lint_and_recompile:
        graph.lint()
        gm.recompile()

    # Useful when retracing for quantization.
    if hasattr(gm, "_qconfig_map"):
        gm._qconfig_map[get_position_ids_node.name] = None
        gm._qconfig_map[position_ids_node.name] = None


def _patch_arguments_(
    gm: GraphModule, mapping: Union[Dict[Node, int], Dict[int, Node]], lint_and_recompile: bool = True
):
    """
    Patches node by replacing their argument to their corresponding values in mapping (supports regular types, tuples
    and slices).
    """

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

    if lint_and_recompile:
        graph.lint()
        gm.recompile()


def transform_to_dynamic_input_(gm: GraphModule, is_retracing: bool = False):
    """Transformation that enables traced models to perform inference on dynamic input shapes."""
    graph = gm.graph
    input_names = gm.dummy_inputs.keys()
    static2dynamic = {}

    # Inserting the nodes that will fetch the batch size and sequence lengths dynamically.
    if gm.use_dynamic_batch_size:
        batch_size_node = _insert_batch_size_node_(gm, lint_and_recompile=False)
        static2dynamic[gm.static_batch_size] = batch_size_node
        if gm.num_choices > 0:
            with graph.inserting_after(batch_size_node):
                static2dynamic[gm.static_batch_size * gm.num_choices] = graph.call_function(
                    operator.mul, args=(batch_size_node, gm.num_choices)
                )
            # graph.lint()
            # gm.recompile()

            # Useful when retracing for quantization.
            if hasattr(gm, "_qconfig_map"):
                gm._qconfig_map[static2dynamic[gm.static_batch_size * gm.num_choices]] = None

    if gm.use_dynamic_sequence_length:
        encoder_sequence_length_node = _insert_encoder_sequence_length_node_(gm, lint_and_recompile=False)
        static2dynamic[gm.static_sequence_length[0]] = encoder_sequence_length_node

        # TODO: do the same for the decoder.
        pass

    _change_view_methods_(gm, static2dynamic, lint_and_recompile=False)
    _patch_getitem_(gm, static2dynamic, lint_and_recompile=False)

    if (
        gm.use_dynamic_sequence_length
        and "position_ids" not in input_names
        and hasattr(gm.config, "max_position_embeddings")
    ):
        _register_position_ids_and_replace_(gm, encoder_sequence_length_node, lint_and_recompile=False)

    _remove_unused_nodes_(gm, lint_and_recompile=False)

    graph.lint()
    gm.recompile()

    gm.static2dynamic = static2dynamic
    gm.dynamic2static = {v: k for (k, v) in static2dynamic.items()}
