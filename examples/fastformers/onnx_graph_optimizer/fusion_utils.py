#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
from logging import getLogger
from .onnx_model import OnnxModel
from typing import Tuple
from onnx import helper, TensorProto

logger = getLogger(__name__)


class FusionUtils:
    def __init__(self, model: OnnxModel):
        self.model: OnnxModel = model

    def cast_graph_input_to_int32(self, input_name: str) -> Tuple[bool, str]:
        graph_input = self.model.find_graph_input(input_name)
        if graph_input is not None and graph_input.type.tensor_type.elem_type != TensorProto.INT32:
            cast_output, cast_node = self.cast_input_to_int32(input_name)
            logger.debug(f"Casted graph input {input_name} to int32")
            return True, cast_output

        logger.debug(f"Did not cast graph input {input_name} to int32: found {graph_input is not None}")
        return False, input_name

    def cast_input_to_int32(self, input_name: str):
        cast_output = input_name + '_int32'

        # Avoid consequent Cast nodes.
        inputs = [input_name]
        output_name_to_node = self.model.output_name_to_node()
        if input_name in output_name_to_node:
            parent_node = output_name_to_node[input_name]
            if parent_node and parent_node.op_type == 'Cast':
                inputs = [parent_node.input[0]]

        cast_node = helper.make_node('Cast', inputs=inputs, outputs=[cast_output])
        cast_node.attribute.extend([helper.make_attribute("to", int(TensorProto.INT32))])
        self.model.add_node(cast_node)

        return cast_output, cast_node

    def remove_cast_int32(self, input_name: str):
        input_name_to_nodes = self.model.input_name_to_nodes()
        nodes = input_name_to_nodes[input_name]
        for node in nodes:
            if node.op_type == "Cast":
                is_int32 = False
                for att in node.attribute:
                    if att.name == 'to' and att.i == int(TensorProto.INT32):
                        is_int32 = True
                        break
                if is_int32:
                    output_name = node.output[0]
                    self.model.remove_node(node)
                    self.model.replace_input_of_all_nodes(output_name, input_name)
