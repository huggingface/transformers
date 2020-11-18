#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
from typing import Dict
from logging import getLogger
from onnx import helper
from .onnx_model import OnnxModel
from .fusion_base import Fusion

logger = getLogger(__name__)


class FusionLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "LayerNormalization", "ReduceMean")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse Layer Normalization subgraph into one node LayerNormalization:
              +----------------------+
              |                      |
              |                      v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                     (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0)    ^
                                     |                                               |
                                     +-----------------------------------------------+

         It also handles cases of duplicated sub nodes exported from older version of PyTorch:
              +----------------------+
              |                      v
              |           +-------> Sub-----------------------------------------------+
              |           |                                                           |
              |           |                                                           v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul --> Add
              |                      ^
              |                      |
              +----------------------+
        """
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) == 0 or len(children) > 2:
            return

        parent = self.model.get_parent(node, 0, output_name_to_node)
        if parent is None:
            return

        if children[0].op_type != 'Sub' or self.model.get_parent(children[0], 0, output_name_to_node) != parent:
            return

        if len(children) == 2:
            if children[1].op_type != 'Sub' or self.model.get_parent(children[1], 0, output_name_to_node) != parent:
                return

        div_node = None
        for child in children:
            div_node = self.model.find_first_child_by_type(child, 'Div', input_name_to_nodes, recursive=False)
            if div_node is not None:
                break
        if div_node is None:
            return

        path_id, parent_nodes, _ = self.model.match_parent_paths(
            div_node, [(['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Sub'], [1, 0, 0, 0, 0]),
                       (['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Cast', 'Sub'], [1, 0, 0, 0, 0, 0])], output_name_to_node)
        if path_id < 0:
            return

        sub_node = parent_nodes[-1]
        if sub_node not in children:
            return

        second_add_node = parent_nodes[1]
        i, add_weight = self.model.get_constant_input(second_add_node)
        if add_weight is None or add_weight <= 0 or add_weight > 1.0E-4:
            logger.warning(f"epsilon value is not expeced: {add_weight}")
            return

        pow_node = parent_nodes[3]
        if not self.model.find_constant_input(pow_node, 2.0) == 1:
            return

        mul_node = input_name_to_nodes[div_node.output[0]][0]
        if mul_node.op_type != 'Mul':
            return

        last_add_node = input_name_to_nodes[mul_node.output[0]][0]
        if last_add_node.op_type != 'Add':
            return

        subgraph_nodes = [node]
        subgraph_nodes.extend(children)
        subgraph_nodes.extend(parent_nodes[:-1])

        subgraph_nodes.extend([last_add_node, mul_node, div_node])
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, last_add_node.output, input_name_to_nodes,
                                                output_name_to_node):
            logger.debug(f"It is not safe to fuse LayerNormalization node. Skip")
            return

        weight_input = mul_node.input[1 - self.model.input_index(div_node.output[0], mul_node)]
        if not self.model.is_constant_with_specified_dimension(weight_input, 1, "layernorm weight"):
            return

        bias_input = last_add_node.input[1 - self.model.input_index(mul_node.output[0], last_add_node)]
        if not self.model.is_constant_with_specified_dimension(bias_input, 1, "layernorm bias"):
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        normalize_node = helper.make_node('LayerNormalization',
                                          inputs=[node.input[0], weight_input, bias_input],
                                          outputs=[last_add_node.output[0]])
        normalize_node.attribute.extend([helper.make_attribute("epsilon", float(add_weight))])
        self.nodes_to_add.append(normalize_node)


class FusionLayerNormalizationTF(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "LayerNormalization", "Add", "TF")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Layer Norm from Keras in Tensorflow:
         +----------------------+
         |                      |
         |                      v                               (B)                             (B)             (A)
        Add --> ReduceMean -->  Sub  --> Mul --> ReduceMean --> Add --> Sqrt --> Reciprocol --> Mul --> Mul --> Sub --> Add
         |          |                                                                            |       ^              ^
         |          |                                                                            |       |              |
         |          +----------------------------------------------------------------------------|-------+              |
         |                                                                                       v                      |
         +-------------------------------------------------------------------------------------> Mul--------------------+
        """
        return_indice = []
        parent_nodes = self.model.match_parent_path(
            node,
            ['Sub', 'Mul', 'Mul', 'Reciprocal', 'Sqrt', 'Add', 'ReduceMean', 'Mul', 'Sub', 'ReduceMean'],
            [   1,     1,   None,            0,      0,     0,         None,     0,    0,          None],
            output_name_to_node,
            return_indice=return_indice) # yapf: disable

        if parent_nodes is None:
            return

        assert len(return_indice) == 3
        if not (return_indice[0] in [0, 1] and return_indice[1] in [0, 1] and return_indice[2] in [0, 1]):
            logger.debug("return indice is exepected in [0, 1], but got {return_indice}")
            return

        sub_node_0, mul_node_0, mul_node_1, reciprocol_node, sqrt_node, add_node_0, reduce_mean_node_0, mul_node_2, sub_node_1, reduce_mean_node_1 = parent_nodes

        mul_node_3 = self.model.match_parent(node, 'Mul', 0, output_name_to_node)
        if mul_node_3 is None:
            logger.debug("mul_node_3 not found")
            return

        root_node = self.model.get_parent(reduce_mean_node_1, 0, output_name_to_node)
        if root_node is None:
            logger.debug("root node is none")
            return

        i, epsilon = self.model.get_constant_input(add_node_0)
        if epsilon is None or epsilon <= 0 or epsilon > 1.0E-5:
            logger.debug("epsilon is not matched")
            return

        if reduce_mean_node_1.input[0] not in mul_node_3.input or reduce_mean_node_1.input[0] not in sub_node_1.input:
            logger.debug("reduce_mean_node_1 and mul_node_3 shall link from root node")
            return

        if mul_node_2.input[0] != mul_node_2.input[1]:
            logger.debug("mul_node_2 shall have two same inputs")
            return

        subgraph_nodes = [
            node, sub_node_0, mul_node_0, mul_node_1, reciprocol_node, sqrt_node, add_node_0, reduce_mean_node_0,
            mul_node_2, sub_node_1, reduce_mean_node_1, mul_node_3
        ]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, node.output, self.model.input_name_to_nodes(),
                                                self.model.output_name_to_node()):
            logger.debug("not safe to fuse layer normalization")
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        weight_input = mul_node_1.input[1]
        bias_input = sub_node_0.input[0]

        #TODO: add epsilon attribute
        fused_node = helper.make_node('LayerNormalization',
                                      inputs=[reduce_mean_node_1.input[0], weight_input, bias_input],
                                      outputs=[node.output[0]])
        fused_node.attribute.extend([helper.make_attribute("epsilon", float(epsilon))])
        self.nodes_to_add.append(fused_node)
