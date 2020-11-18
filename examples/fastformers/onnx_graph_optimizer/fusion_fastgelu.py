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


class FusionFastGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "FastGelu", "Tanh")

    def fuse(self, erf_node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        if self.fuse_1(erf_node, input_name_to_nodes, output_name_to_node):
            return
        self.fuse_2(erf_node, input_name_to_nodes, output_name_to_node)

    def fuse_1(self, tanh_node, input_name_to_nodes, output_name_to_node) -> Optional[bool]:
        """
        Fuse Gelu with tanh into one node:
              +---------------------------+
              |                           |
              |                           v
            [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul
              |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)     ^
              |                                                              |
              +------> Mul(B=0.5)--------------------------------------------+
        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'Add':
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        mul_after_tanh = children[0]

        mul_half = self.model.match_parent(mul_after_tanh, 'Mul', None, output_name_to_node)
        if mul_half is None:
            return

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        root_node = self.model.get_parent(mul_half, 0 if i == 1 else 1, output_name_to_node)
        if root_node is None:
            return

        mul_before_tanh = self.model.match_parent(tanh_node, 'Mul', 0, output_name_to_node)
        if mul_before_tanh is None:
            return

        i = self.model.find_constant_input(mul_before_tanh, 0.7978, delta=0.0001)
        if i < 0:
            return

        add_before_tanh = self.model.match_parent(mul_before_tanh, 'Add', 0 if i == 1 else 1, output_name_to_node)
        if add_before_tanh is None:
            return

        mul_after_pow = self.model.match_parent(add_before_tanh, 'Mul', None, output_name_to_node, exclude=[root_node])
        if mul_after_pow is None:
            return

        i = self.model.find_constant_input(mul_after_pow, 0.0447, delta=0.0001)
        if i < 0:
            return

        pow = self.model.match_parent(mul_after_pow, 'Pow', 0 if i == 1 else 1, output_name_to_node)
        if pow is None:
            return

        if not self.model.has_constant_input(pow, 3.0):
            return

        if pow.input[0] != root_node.output[0]:
            return

        subgraph_nodes = [
            mul_after_tanh, mul_half, add_after_tanh, tanh_node, mul_before_tanh, add_before_tanh, mul_after_pow, pow
        ]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [mul_after_tanh.output[0]], input_name_to_nodes,
                                                output_name_to_node):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node('FastGelu',
                                      inputs=[root_node.output[0]],
                                      outputs=mul_after_tanh.output,
                                      name=self.model.create_node_name('FastGelu'))
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)

    def fuse_2(self, tanh_node, input_name_to_nodes: Dict, output_name_to_node: Dict) -> Optional[bool]:
        """
        This pattern is from Tensorflow mode.
        Fuse Gelu with tanh into one node:
              +---------------------------+
              |                           |
              |                           v
            [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul(B=0.5)-->Mul-->
              |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)                  ^
              |                                                                           |
              +---------------------------------------------------------------------------+
        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != 'Add':
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        mul_half = children[0]

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        if mul_half.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_half.output[0]]
        if len(children) != 1 or children[0].op_type != 'Mul':
            return
        mul_after_mul_half = children[0]

        root_node = self.model.get_parent(mul_after_mul_half,
                                          0 if mul_after_mul_half.input[1] == mul_half.output[0] else 1,
                                          output_name_to_node)
        if root_node is None:
            return

        mul_before_tanh = self.model.match_parent(tanh_node, 'Mul', 0, output_name_to_node)
        if mul_before_tanh is None:
            return

        i = self.model.find_constant_input(mul_before_tanh, 0.7978, delta=0.0001)
        if i < 0:
            return

        add_before_tanh = self.model.match_parent(mul_before_tanh, 'Add', 0 if i == 1 else 1, output_name_to_node)
        if add_before_tanh is None:
            return

        mul_after_pow = self.model.match_parent(add_before_tanh, 'Mul', None, output_name_to_node, exclude=[root_node])
        if mul_after_pow is None:
            return

        i = self.model.find_constant_input(mul_after_pow, 0.0447, delta=0.0001)
        if i < 0:
            return

        pow = self.model.match_parent(mul_after_pow, 'Pow', 0 if i == 1 else 1, output_name_to_node)
        if pow is None:
            return

        if not self.model.has_constant_input(pow, 3.0):
            return

        if pow.input[0] != root_node.output[0]:
            return

        subgraph_nodes = [
            mul_after_mul_half, mul_half, add_after_tanh, tanh_node, mul_before_tanh, add_before_tanh, mul_after_pow,
            pow
        ]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, [mul_after_mul_half.output[0]], input_name_to_nodes,
                                                output_name_to_node):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node('FastGelu',
                                      inputs=[root_node.output[0]],
                                      outputs=mul_after_mul_half.output,
                                      name=self.model.create_node_name('FastGelu'))
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        return True
