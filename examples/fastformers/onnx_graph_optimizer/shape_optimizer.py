#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# This tool is not used directly in bert optimization. It could assist developing the optimization script on the following senarios:
# (1) It could simplify graph by removing many sub-graphs related to reshape.
# (2) It could reduce extra inputs and outputs to fit other tools. The script compare_bert_results.py or bert_perf_test.py requires 3 inputs.

import sys
import argparse
import numpy as np
from collections import deque
from typing import List
import onnx
import re
import tempfile
import os
import logging
from datetime import datetime
from pathlib import Path
from onnx import ModelProto, TensorProto, numpy_helper
import onnxruntime
from .onnx_model import OnnxModel

logger = logging.getLogger(__name__)

CONSTANT_SHAPE_NAME_PREFIX = 'constant_shape_opt__'
RESHAPE_INPUT_SHAPE_PREFIX = 'reshape_input_shape__'


class BertOnnxModelShapeOptimizer(OnnxModel):
    """
    This optimizer will replace Shape output or the shape input of Reshape node by initializer. Currently, it requires
    model inputs to have static shape.
    """
    def __init__(self, onnx_model):
        super().__init__(onnx_model.model)

    def add_shape_initializer(self, shape):
        """
        Add an initializer for constant shape.
        """
        shape_value = np.asarray(shape, dtype=np.int64)
        constant_shape_name = self.create_node_name('Constant', CONSTANT_SHAPE_NAME_PREFIX)
        tensor = onnx.helper.make_tensor(name=constant_shape_name,
                                         data_type=TensorProto.INT64,
                                         dims=shape_value.shape,
                                         vals=shape_value)
        self.add_initializer(tensor)
        return tensor

    def get_shape_outputs(self):
        """
        Returns a list of output names of all Shape nodes.
        """
        input_name_to_nodes = self.input_name_to_nodes()

        outputs = []
        for node in self.model.graph.node:
            if node.op_type == 'Shape':
                if node.output[0] in input_name_to_nodes:
                    outputs.append(node.output[0])

        return outputs

    def get_reshape_shape_inputs(self):
        """
        Returns a list of shape input names of Reshape nodes.
        """
        output_name_to_node = self.output_name_to_node()

        shape_inputs = []
        for node in self.model.graph.node:
            if node.op_type == 'Reshape':
                shape_inputs.append(node.input[1])

        return shape_inputs

    def add_shape_for_reshape_input(self):
        """
        For each Reshape node, create a Shape node for its first input.
        Returns the output names of these Shape nodes.
        """
        output_names = []
        nodes_to_add = []
        for node in self.model.graph.node:
            if node.op_type == 'Reshape':
                input = node.input[0]
                output_name = self.create_node_name('Reshape_Input', RESHAPE_INPUT_SHAPE_PREFIX)
                shape_node = onnx.helper.make_node('Shape', inputs=[input], outputs=[output_name])
                nodes_to_add.append(shape_node)
                output_names.append(output_name)

        self.add_nodes(nodes_to_add)
        return output_names

    def add_extra_graph_output(self, extra_outputs):
        """
        Add a list of output names to graph output.
        """
        names_to_evaluate = []
        output_names = [output.name for output in self.model.graph.output]
        for name in extra_outputs:

            if self.get_initializer(name) is not None:  # already a constant
                continue
            names_to_evaluate.append(name)

            if name not in output_names:
                output_info = onnx.helper.ValueInfoProto()
                output_info.name = name
                self.model.graph.output.extend([output_info])
                output_names.append(name)

        return names_to_evaluate

    # Update input and output shape to be static
    def use_static_input(self, inputs, batch_size=1, max_seq_len=128):
        """
        Update the model to use static axes instead of dynamic axes for graph inputs.
        """
        for input in self.model.graph.input:
            if input.name in inputs:
                dim_proto = input.type.tensor_type.shape.dim[0]
                dim_proto.dim_value = batch_size
                dim_proto = input.type.tensor_type.shape.dim[1]
                if dim_proto.HasField('dim_param'):
                    dim_proto.dim_value = max_seq_len
                elif dim_proto.HasField('dim_value') and dim_proto.dim_value != max_seq_len:
                    raise ValueError(
                        'Unable to set dimension value to {} for axis {} of {}. Contradicts existing dimension value {}.'
                        .format(max_seq_len, 1, input.name, dim_proto.dim_value))

    def create_dummy_inputs(self,
                            input_ids,
                            segment_ids,
                            input_mask,
                            batch_size,
                            sequence_length,
                            elem_type,
                            dictionary_size=8):
        """
        Create dummy data for model inputs. If the model has more than 3 inputs, please update this function accordingly before running the tool.
        """
        assert elem_type in [1, 6, 7]  # only int32, int64 and float32 are supported.

        # Create dummy inputs
        input_1 = np.random.randint(dictionary_size, size=(batch_size, sequence_length), dtype=np.int32)
        input_2 = np.ones((batch_size, sequence_length), dtype=np.int32)
        input_3 = np.zeros((batch_size, sequence_length), dtype=np.int32)

        # Here we assume that 3 inputs have same data type
        if elem_type == 1:  #float32
            input_1 = np.float32(input_1)
            input_2 = np.float32(input_2)
            input_3 = np.float32(input_3)
        elif elem_type == 7:  #int64
            input_1 = np.int64(input_1)
            input_2 = np.int64(input_2)
            input_3 = np.int64(input_3)

        inputs = {input_ids: input_1, input_mask: input_2, segment_ids: input_3}
        return inputs

    def shape_optimization(self, temp_model_path, input_ids, segment_ids, input_mask, output_names, batch_size,
                           sequence_length, enable_shape_opt, enable_reshape_opt, verbose):
        self.bert_inputs = [input_ids, segment_ids, input_mask]

        extra_outputs = []
        if enable_shape_opt:
            extra_outputs.extend(self.get_shape_outputs())

        if enable_reshape_opt:
            reshape_shape_inputs = self.get_reshape_shape_inputs()
            reshape_input_shapes = self.add_shape_for_reshape_input()
            extra_outputs.extend(reshape_shape_inputs)
            extra_outputs.extend(reshape_input_shapes)

        if len(extra_outputs) == 0:
            return

        names_to_evaluate = self.add_extra_graph_output(extra_outputs)

        # This tool does not support dynamic axes right now.
        self.use_static_input(self.bert_inputs, batch_size, sequence_length)

        with open(temp_model_path, "wb") as out:
            out.write(self.model.SerializeToString())
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxruntime.InferenceSession(temp_model_path,
                                               sess_options,
                                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        elem_type = 7
        for input in self.model.graph.input:
            if input.name == input_ids:
                elem_type = input.type.tensor_type.elem_type
        inputs = self.create_dummy_inputs(input_ids, segment_ids, input_mask, batch_size, sequence_length, elem_type)

        outputs = session.run(names_to_evaluate, inputs)
        shapes = {}
        for i, name in enumerate(names_to_evaluate):
            shapes[name] = outputs[i]

        logger.debug(f"shapes={shapes}")

        if enable_reshape_opt:
            for i, shape_input in enumerate(reshape_shape_inputs):
                input_shape = reshape_input_shapes[i]
                self.update_target_shape(shapes, shape_input, input_shape, verbose)

        for name, shape in shapes.items():
            tensor = self.add_shape_initializer(shape)
            self.replace_input_of_all_nodes(name, tensor.name)

        # Remove extra outputs, and prune all nodes not linked to output.
        self.prune_graph(output_names)

    def update_target_shape(self, shapes, shape_input, input_shape, verbose):
        """
        Update the target shape to use 0 to represent that dimension value does not change.
        For example, shape of source data is (2, 5, 8) and target shape is (2, 5, 4, 2), the target shape will be updated to (0, 0, 4, 2).
        """
        if shape_input in shapes:
            target_shape = shapes[shape_input]
        else:
            initializer = self.get_initializer(shape_input)
            assert initializer is not None
            target_shape = numpy_helper.to_array(initializer)

        if input_shape in shapes:
            source_shape = shapes[input_shape]
        else:
            initializer = self.get_initializer(input_shape)
            assert initializer is not None
            source_shape = numpy_helper.to_array(initializer)

        new_target_shape = []
        for i, dim_value in enumerate(target_shape):
            if i < len(source_shape) and source_shape[i] == dim_value:
                new_target_shape.append(0)
            else:
                new_target_shape.append(dim_value)
        shapes[shape_input] = new_target_shape

        logger.debug(f"source_shape={source_shape}, target_shape={target_shape}, new_target_shape={new_target_shape}")

    def validate_input(self, input: str):
        if not self.find_graph_input(input):
            valid_names = [input.name for input in self.model.graph.input]
            raise Exception("Input {} does not exist in the graph inputs: {}".format(input, valid_names))

    def validate_outputs(self, output_names: List[str]):
        valid_names = [output.name for output in self.model.graph.output]
        for name in output_names:
            if name not in valid_names:
                raise Exception("Output {} does not exist in the graph outputs: {}".format(name, valid_names))

    def optimize(self,
                 output_path: str,
                 input_ids: str,
                 segment_ids: str,
                 input_mask: str,
                 enable_shape_opt: bool,
                 enable_reshape_opt: bool,
                 output_names: List[str] = None,
                 batch_size=1,
                 sequence_length=128,
                 verbose=False):
        # Skip if shape optimization has been done before.
        for tensor in self.model.graph.initializer:
            if tensor.name.startswith(CONSTANT_SHAPE_NAME_PREFIX):
                logger.info('Skip shape optimization since it has been done before')
                return

        self.validate_input(input_ids)
        self.validate_input(segment_ids)
        self.validate_input(input_mask)

        if output_names is not None:
            self.validate_outputs(output_names)
            self.prune_graph(output_names)

        remaining_outputs = [output.name for output in self.model.graph.output]

        if enable_shape_opt or enable_reshape_opt:
            if len(self.get_graph_inputs_excluding_initializers()) != 3:
                logger.info('Skip shape optimization since graph input number is not 3')
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_name = 'temp_{}.onnx'.format(datetime.now().strftime("%m_%d-%H_%M_%S"))
                dir = "." if verbose else temp_dir
                temp_file = os.path.join(dir, temp_file_name)
                self.shape_optimization(temp_file, input_ids, segment_ids, input_mask, remaining_outputs, batch_size,
                                        sequence_length, enable_shape_opt, enable_reshape_opt, verbose)
            logger.debug(f"Temp model with additional outputs: {temp_file}")
            logger.warning(
                f'Shape optimization is done. The optimized model might only work for input with batch_size={batch_size} sequence_length={sequence_length}'
            )

        if output_path is not None:
            with open(output_path, "wb") as out:
                out.write(self.model.SerializeToString())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--input_ids', required=True, type=str)
    parser.add_argument('--segment_ids', required=True, type=str)
    parser.add_argument('--input_mask', required=True, type=str)
    parser.add_argument('--output_names', required=False, type=str, default=None)
    parser.add_argument('--batch_size', required=False, type=int, default=1)
    parser.add_argument('--sequence_length', required=False, type=int, default=128)
    parser.add_argument('--enable_shape_opt', required=False, action='store_true')
    parser.set_defaults(enable_shape_opt=False)
    parser.add_argument('--enable_reshape_opt', required=False, action='store_true')
    parser.set_defaults(enable_reshape_opt=False)
    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    return args


def setup_logging(verbose):
    log_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        log_handler.setFormatter(logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s'))
        logging_level = logging.DEBUG
    else:
        log_handler.setFormatter(logging.Formatter('%(filename)20s: %(message)s'))
        logging_level = logging.INFO
    log_handler.setLevel(logging_level)
    logger.addHandler(log_handler)
    logger.setLevel(logging_level)


def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    output_names = None if args.output_names is None else args.output_names.split(';')

    model = ModelProto()
    with open(args.input, "rb") as input_file:
        model.ParseFromString(input_file.read())
    onnx_model = OnnxModel(model)

    optimizer = BertOnnxModelShapeOptimizer(onnx_model)

    optimizer.optimize(args.output, args.input_ids, args.segment_ids, args.input_mask, args.enable_shape_opt,
                       args.enable_reshape_opt, output_names, args.batch_size, args.sequence_length, args.verbose)


if __name__ == "__main__":
    main()
