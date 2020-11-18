#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from logging import getLogger
from onnx import helper, numpy_helper
from .onnx_model import OnnxModel
from .fusion_base import Fusion

logger = getLogger(__name__)


class FusionSkipLayerNormalization(Fusion):
    """
    Fuse Add + LayerNormalization into one node: SkipLayerNormalization
    Note: This fusion does not check the input shape of Add and LayerNormalization.
    """
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipLayerNormalization", "LayerNormalization")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        add = self.model.get_parent(node, 0, output_name_to_node)
        if add is not None and add.op_type == 'Add' and self.model.is_safe_to_fuse_nodes(
            [add, node], node.output, input_name_to_nodes, output_name_to_node):
            self.nodes_to_remove.extend([add, node])

            inputs = [add.input[0], add.input[1], node.input[1], node.input[2]]
            normalize_node = helper.make_node("SkipLayerNormalization",
                                              inputs=inputs,
                                              outputs=[node.output[0]],
                                              name=self.model.create_node_name("SkipLayerNormalization",
                                                                               name_prefix="SkipLayerNorm"))
            normalize_node.domain = "com.microsoft"

            # Pass attribute "epsilon" from layernorm node to SkipLayerNormalization
            for att in node.attribute:
                if att.name == 'epsilon':
                    normalize_node.attribute.extend([att])

            # Set default epsilon if no epsilon exists from layernorm
            if len(normalize_node.attribute) == 0:
                normalize_node.attribute.extend([helper.make_attribute("epsilon", 1.0E-12)])

            self.nodes_to_add.append(normalize_node)


class FusionBiasSkipLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipLayerNormalization", "SkipLayerNormalization", "add bias")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 4:
            return

        return_indice = []
        nodes = self.model.match_parent_path(node, ['Add', 'MatMul'], [None, None], None, return_indice)
        if nodes is None:
            return
        assert len(return_indice) == 2
        add_input_index = return_indice[0]
        if add_input_index >= 2:
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
            logger.debug(f"Bias weight not found")
            return
        if len(bias_weight.shape) != 1:
            logger.debug(f"Bias weight is not 1D")
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [node.output[0]], input_name_to_nodes,
                                                output_name_to_node):
            logger.debug(f"Skip fusing SkipLayerNormalization with Bias since it is not safe")
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        inputs = [
            node.input[1 - add_input_index], matmul.output[0], node.input[2], node.input[3], add.input[bias_index]
        ]
        new_node = helper.make_node("SkipLayerNormalization",
                                    inputs=inputs,
                                    outputs=node.output,
                                    name=self.model.create_node_name("SkipLayerNormalization",
                                                                     "SkipLayerNorm_AddBias_"))
        new_node.domain = "com.microsoft"
        self.nodes_to_add.append(new_node)
