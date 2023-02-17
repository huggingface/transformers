"""
Code to remove duplicate initializers to reduce ONNX model size.
"""

import os

import numpy
import onnx


def _is_equal_tensor_proto(a, b):
    name_a = a.name
    name_b = b.name

    a.name = ""
    b.name = ""

    res = a == b

    a.name = name_a
    b.name = name_b

    return res


def _node_replace_input_with(node_proto, name, new_name):
    for i, input_name in enumerate(node_proto.input):
        if input_name == name:
            node_proto.input.insert(i, new_name)
            node_proto.input.pop(i + 1)

    if node_proto.op_type == "If":
        _graph_replace_input_with(node_proto.attribute[0].g, name, new_name)
        _graph_replace_input_with(node_proto.attribute[1].g, name, new_name)
    if node_proto.op_type == "Loop":
        _graph_replace_input_with(node_proto.attribute[0].g, name, new_name)


def _graph_replace_input_with(graph_proto, name, new_name):
    for n in graph_proto.node:
        _node_replace_input_with(n, name, new_name)


def _remove_dup_initializers_from_model(model, model_without_ext, ind_to_replace):
    inits_with_data = [i for i in model.graph.initializer]
    inits = [i for i in model_without_ext.graph.initializer]
    for i, ref_i in ind_to_replace:
        assert inits_with_data[i].name == inits[i].name
        assert inits_with_data[ref_i].name == inits[ref_i].name
        assert i > ref_i

        name_i = inits[i].name
        name_ref = inits[ref_i].name

        model_without_ext.graph.initializer.remove(inits[i])

        # for n in model.graph.node:
        _graph_replace_input_with(model_without_ext.graph, name_i, name_ref)


def remove_dup_initializers(onnx_file_path):
    """
    Removes duplicate initializers from the model to reduce its size.
    Writes a new file in the same directory as onnx_file_path and returns the path to that file.
    """

    model_file_folder = os.path.dirname(onnx_file_path)
    model_file_name = os.path.basename(onnx_file_path)

    model = onnx.load(os.path.join(model_file_folder, model_file_name))

    inits = [i for i in model.graph.initializer]

    dup_set = set()
    dup_map = {}
    ind_to_replace = []

    total_reduced_size = 0

    for i in range(len(inits)):
        if i in dup_set:
            continue

        for j in range(i + 1, len(inits)):
            if j in dup_set:
                continue
            if _is_equal_tensor_proto(inits[i], inits[j]):
                dup_set.add(i)
                dup_set.add(j)

                dtype = inits[j].data_type
                mem_size = numpy.prod(inits[j].dims)
                if dtype == 1:
                    mem_size *= 4
                elif dtype == 6:
                    mem_size *= 4
                elif dtype == 7 or dtype == 11:
                    mem_size *= 8
                else:
                    print("unexpected data type: ", dtype)
                total_reduced_size += mem_size

                name_i = inits[i].name
                name_j = inits[j].name

                if name_i in dup_map:
                    dup_map[name_i].append(name_j)
                else:
                    dup_map[name_i] = [name_j]
                ind_to_replace.append((j, i))

    print("total reduced size: ", total_reduced_size / 1024 / 1024 / 1024, "GB")

    ind_to_replace = sorted(ind_to_replace)
    _remove_dup_initializers_from_model(model, model, ind_to_replace)

    optimized_model_file_name = "optimized_" + model_file_name
    new_model = os.path.join(model_file_folder, optimized_model_file_name)
    onnx.save(model, new_model)

    return new_model
