#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from logging import getLogger
from onnx import helper, numpy_helper
from .onnx_model import OnnxModel
from .fusion_base import Fusion

logger = getLogger(__name__)


class FusionBiasGelu(Fusion):
    def __init__(self, model: OnnxModel, is_fastgelu):
        if is_fastgelu:
            super().__init__(model, 'FastGelu', 'FastGelu', 'add bias')
        else:
            super().__init__(model, 'BiasGelu', 'Gelu')

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        gelu_op_type = node.op_type
        fuse_op_type = 'BiasGelu' if gelu_op_type == 'Gelu' else 'FastGelu'

        if len(node.input) != 1:
            return

        nodes = self.model.match_parent_path(node, ['Add', 'MatMul'], [0, None])
        if nodes is None:
            return
        (add, matmul) = nodes

        # bias should be one dimension
        bias_index = -1
        for i, input in enumerate(add.input):
            initializer = self.model.get_initializer(input)
            if initializer is None:
                continue
            bias_index = i
            bias_weight = numpy_helper.to_array(initializer)
            break
        if bias_weight is None:
            return
        if len(bias_weight.shape) != 1:
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [node.output[0]], input_name_to_nodes,
                                                output_name_to_node):
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        fused_node = helper.make_node(fuse_op_type,
                                      inputs=[matmul.output[0], add.input[bias_index]],
                                      outputs=node.output,
                                      name=self.model.create_node_name(fuse_op_type, gelu_op_type + "_AddBias_"))
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
