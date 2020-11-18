#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
from logging import getLogger
from onnx import helper, numpy_helper
from .onnx_model import OnnxModel
from .fusion_base import Fusion

logger = getLogger(__name__)


class FusionQuantizeMatMulFbgemm(Fusion):
    def __init__(self, model: OnnxModel, fuse_relu, fuse_bias):
        if fuse_relu:
          super().__init__(model, 'QuantizeMatMulFbgemm', 'Relu', 'quantize matmul with fbgemm')
        elif fuse_bias:
          super().__init__(model, 'QuantizeMatMulFbgemm', 'Add', 'quantize matmul with fbgemm')
        else:
          super().__init__(model, 'QuantizeMatMulFbgemm', ['SkipLayerNormalization', 'BiasGelu'], 'quantize matmul with fbgemm')

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        fuse_op_type = 'QuantizeMatMulFbgemm'

        # Fused matmul + bias + relu
        if node.op_type == 'Relu':
          nodes = self.model.match_parent_path(node, ['Add', 'MatMul'], [0, None])
          if nodes is not None:
            nodes.append(node)
            self.nodes_to_remove.extend(nodes)
            fused_inputs = nodes[1].input
            fused_inputs.append(nodes[0].input[1])
            fused_node = helper.make_node(fuse_op_type,
                                          inputs=fused_inputs,
                                          outputs=nodes[-1].output,
                                          name=self.model.create_node_name(fuse_op_type, "_MatmulQuantFuseBiasRelu_"))
            fused_node.domain = "com.microsoft"
            fused_node.attribute.extend([helper.make_attribute("fuse_bias", 1)])
            fused_node.attribute.extend([helper.make_attribute("fuse_relu", 1)])
            self.nodes_to_add.append(fused_node)
          return

        # Fused matmul + bias
        if node.op_type == 'Add':
          nodes = self.model.match_parent_path(node, ['MatMul'], [None])
          if nodes is not None:
            nodes.append(node)
            self.nodes_to_remove.extend(nodes)
            fused_inputs = nodes[0].input
            fused_inputs.append(node.input[1])
            fused_node = helper.make_node(fuse_op_type,
                                          inputs=fused_inputs,
                                          outputs=nodes[-1].output,
                                          name=self.model.create_node_name(fuse_op_type, "_MatmulQuantFuseBiasRelu_"))
            fused_node.domain = "com.microsoft"
            fused_node.attribute.extend([helper.make_attribute("fuse_bias", 1)])
            fused_node.attribute.extend([helper.make_attribute("fuse_relu", 0)])
            self.nodes_to_add.append(fused_node)
          return

        # Just replace matmul
        nodes = self.model.match_parent_path(node, ['MatMul'], [None])
        if nodes is None:
            return
        # (matmul) = nodes
        if len(nodes) != 1:
            return

        self.nodes_to_remove.extend(nodes)

        fused_node = helper.make_node(fuse_op_type,
                                      inputs=nodes[0].input,
                                      outputs=nodes[0].output,
                                      name=self.model.create_node_name(fuse_op_type, "_ReplacedMatmulQuant_"))
        fused_node.domain = "com.microsoft"
        fused_node.attribute.extend([helper.make_attribute("fuse_bias", 0)])
        fused_node.attribute.extend([helper.make_attribute("fuse_relu", 0)])
        self.nodes_to_add.append(fused_node)
