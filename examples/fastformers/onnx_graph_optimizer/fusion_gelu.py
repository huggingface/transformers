#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
from typing import Dict, Optional
from logging import getLogger
from onnx import helper
from .onnx_model import OnnxModel
from .fusion_base import Fusion

logger = getLogger(__name__)


class FusionGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Gelu", "Erf")

    def fuse(self, erf_node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        if self.fuse_1(erf_node, input_name_to_nodes, output_name_to_node):
            return
        if self.fuse_2(erf_node, input_name_to_nodes, output_name_to_node):
            return
        self.fuse_3(erf_node, input_name_to_nodes, output_name_to_node)

    def fuse_1(self, erf_node, input_name_to_nodes: Dict, output_name_to_node: Dict) -> Optional[bool]:
        """
        This pattern is from PyTorch model
        Fuse Gelu with Erf into one node:
        Pattern 1:
                       +-------Mul(0.5)---------------------+
                       |                                    |
                       |                                    v
                    [root] --> Div -----> Erf  --> Add --> Mul -->
                              (B=1.4142...)       (1)

        Pattern 2:
                       +------------------------------------+
                       |                                    |
                       |                                    v
                    [root] --> Div -----> Erf  --> Add --> Mul -->Mul -->
                              (B=1.4142...)       (1)            (0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if erf_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'Add':
            return
        add_after_erf = children[0]

        if not self.model.has_constant_input(add_after_erf, 1):
            return

        if add_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        mul_after_erf = children[0]

        div = self.model.match_parent(erf_node, 'Div', 0, output_name_to_node)
        if div is None:
            return

        if self.model.find_constant_input(div, 1.4142, delta=0.001) != 1:
            return

        subgraph_input = div.input[0]

        another = 1 if mul_after_erf.input[0] == add_after_erf.output[0] else 0
        if subgraph_input == mul_after_erf.input[another]:  # pattern 2
            children = input_name_to_nodes[mul_after_erf.output[0]]
            if len(children) != 1 or children[0].op_type != 'Mul':
                return
            mul_half = children[0]
            if not self.model.has_constant_input(mul_half, 0.5):
                return
            subgraph_output = mul_half.output[0]
        else:  # pattern 1
            mul_half = self.model.match_parent(mul_after_erf, 'Mul', another, output_name_to_node)
            if mul_half is None:
                return

            if not self.model.has_constant_input(mul_half, 0.5):
                return

            if subgraph_input not in mul_half.input:
                return

            subgraph_output = mul_after_erf.output[0]

        subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul_half]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [subgraph_output], input_name_to_nodes,
                                                output_name_to_node):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node('Gelu', inputs=[subgraph_input], outputs=[subgraph_output])
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        return True

    def fuse_2(self, erf_node, input_name_to_nodes: Dict, output_name_to_node: Dict) -> Optional[bool]:
        """
        This pattern is from Keras model
        Fuse Gelu with Erf into one node:
                       +------------------------------------------+
                       |                                          |
                       |                                          v
                    [root] --> Div -----> Erf  --> Add --> Mul -->Mul
                              (B=1.4142...)       (A=1)   (A=0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if erf_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'Add':
            return
        add_after_erf = children[0]

        if not self.model.has_constant_input(add_after_erf, 1):
            return

        if add_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        mul_after_erf = children[0]

        if not self.model.has_constant_input(mul_after_erf, 0.5):
            return

        if mul_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        mul = children[0]

        div = self.model.match_parent(erf_node, 'Div', 0, output_name_to_node)
        if div is None:
            return

        sqrt_node = None
        if self.model.find_constant_input(div, 1.4142, delta=0.001) != 1:
            sqrt_node = self.model.match_parent(div, 'Sqrt', 1, output_name_to_node)
            if sqrt_node is None:
                return
            if not self.model.has_constant_input(sqrt_node, 2.0):
                return

        root_node = self.model.get_parent(div, 0, output_name_to_node)
        if root_node is None:
            return

        if root_node.output[0] not in mul.input:
            return

        subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul]
        if sqrt_node:
            subgraph_nodes.append(sqrt_node)

        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [mul.output[0]], input_name_to_nodes,
                                                output_name_to_node):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node('Gelu', inputs=[root_node.output[0]], outputs=[mul.output[0]])
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        return True

    def fuse_3(self, erf_node, input_name_to_nodes: Dict, output_name_to_node: Dict) -> Optional[bool]:
        """
        This pattern is from TensorFlow model
        Fuse Gelu with Erf into one node:
                       +----------------------------------------------+
                       |                                              |
                       |                                              v
                    [root] --> Mul -----> Erf    -->   Add --> Mul -->Mul
                               (A=0.7071067690849304)  (B=1)  (B=0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """

        if erf_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'Add':
            return
        add_after_erf = children[0]

        if not self.model.has_constant_input(add_after_erf, 1):
            return

        if add_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        mul_half = children[0]

        if not self.model.has_constant_input(mul_half, 0.5):
            return

        first_mul = self.model.match_parent(erf_node, 'Mul', 0, output_name_to_node)
        if first_mul is None:
            return

        i = self.model.find_constant_input(first_mul, 0.7071067690849304, delta=0.001)
        if i < 0:
            return

        root_node = self.model.get_parent(first_mul, 0 if i == 1 else 1, output_name_to_node)
        if root_node is None:
            return

        if mul_half.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_half.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        last_mul = children[0]

        if not (last_mul.input[0] == root_node.output[0] or last_mul.input[1] == root_node.output[0]):
            return

        subgraph_nodes = [first_mul, erf_node, add_after_erf, mul_half, last_mul]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [last_mul.output[0]], input_name_to_nodes,
                                                output_name_to_node):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node('Gelu', inputs=[root_node.output[0]], outputs=[last_mul.output[0]])
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        return True
