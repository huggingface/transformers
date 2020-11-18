#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import numpy as np
from logging import getLogger
from onnx import helper, numpy_helper, TensorProto
from .onnx_model import OnnxModel
from .fusion_base import Fusion
from .fusion_utils import FusionUtils
from .fusion_attention import AttentionMask

logger = getLogger(__name__)


class FusionAttentionFbgemm(Fusion):
    """
    Fuse Attention subgraph into one Attention node.
    """
    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int, head_size: int, attention_mask: AttentionMask):
        super().__init__(model, "QAttentionFbgemm", ["SkipLayerNormalization", "LayerNormalization"], 'fuse attention and quantize with fbgemm')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.attention_mask = attention_mask

    def create_attention_node(self, mask_index, q_matmul, k_matmul, v_matmul, q_add, k_add, v_add, input, output):
        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        q_bias = self.model.get_initializer(q_add.input[1])
        k_bias = self.model.get_initializer(k_add.input[1])
        v_bias = self.model.get_initializer(v_add.input[1])

        if q_weight is None:
            print(f"{q_matmul.input[1]} is not initializer. Please set do_constant_folding=True in torch.onnx.export")
            return None
        if not (k_weight and v_weight and q_bias and k_bias):
            return None

        qw = numpy_helper.to_array(q_weight)
        assert qw.shape == (self.hidden_size, self.num_heads * self.head_size)

        kw = numpy_helper.to_array(k_weight)
        assert kw.shape == (self.hidden_size, self.num_heads * self.head_size)

        vw = numpy_helper.to_array(v_weight)
        assert vw.shape == (self.hidden_size, self.num_heads * self.head_size)

        qkv_weight = np.stack((qw, kw, vw), axis=-2)

        qb = numpy_helper.to_array(q_bias)
        assert qb.shape == (self.num_heads * self.head_size, )

        kb = numpy_helper.to_array(k_bias)
        assert kb.shape == (self.num_heads * self.head_size, )

        vb = numpy_helper.to_array(v_bias)
        assert vb.shape == (self.num_heads * self.head_size, )

        qkv_bias = np.stack((qb, kb, vb), axis=-2)

        attention_node_name = self.model.create_node_name('QAttentionFbgemm')

        weight = helper.make_tensor(name=attention_node_name + '_qkv_weight',
                                    data_type=TensorProto.FLOAT,
                                    dims=[self.hidden_size, int(3 * self.num_heads * self.head_size)],
                                    vals=qkv_weight.flatten().tolist())
        self.model.add_initializer(weight)

        bias = helper.make_tensor(name=attention_node_name + '_qkv_bias',
                                  data_type=TensorProto.FLOAT,
                                  dims=[int(3 * self.num_heads * self.head_size)],
                                  vals=qkv_bias.flatten().tolist())
        self.model.add_initializer(bias)

        attnetion_inputs = [input, attention_node_name + '_qkv_weight', attention_node_name + '_qkv_bias', mask_index]
        attention_node = helper.make_node('QAttentionFbgemm',
                                          inputs=attnetion_inputs,
                                          outputs=[output],
                                          name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", self.num_heads)])
        attention_node.attribute.extend([helper.make_attribute("head_size", int(self.head_size))])

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if normalize_node.op_type == "SkipLayerNormalization":
            # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
            qkv_nodes = self.model.match_parent_path(normalize_node, ['Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul'],
                                                    [None, 0, 0, 0, 0])
        elif normalize_node.op_type == "LayerNormalization":
            # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
            qkv_nodes = self.model.match_parent_path(normalize_node, ['Add', 'Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul'],
                                                    [None, None, 0, 0, 0, 0])
        else:
            qkv_nodes = None

        if qkv_nodes is None:
            return

        other_inputs = []
        node_to_check = normalize_node if normalize_node.op_type == "SkipLayerNormalization" else qkv_nodes[0]
        other_input_node = qkv_nodes[0] if normalize_node.op_type == "SkipLayerNormalization" else qkv_nodes[1]
        for i, input in enumerate(node_to_check.input):
            if input not in output_name_to_node:
                continue

            if input == other_input_node.output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]
        # in case of pre-layernorm, LayerNormalization is the root node.
        if normalize_node.op_type == "LayerNormalization":
            children = input_name_to_nodes[root_input]
            for child in children:
                if child.op_type == "LayerNormalization":
                    root_input = child.output[0]
        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count('MatMul') != 3:
            return

        if normalize_node.op_type == "SkipLayerNormalization":
            (_, matmul_qkv, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        elif normalize_node.op_type == "LayerNormalization":
            (_, _, matmul_qkv, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(matmul_qkv, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Div', 'MatMul'], [0, 0, 0, 0])
        if qk_nodes is None:
            qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Mul', 'MatMul'], [0, 0, 0, 0])
            if qk_nodes is None:
                logger.debug("fuse_attention: failed to match qk path")
                return
        (_, add_qk, _, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [0, 0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (_, _, add_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
        if k_nodes is None:
            k_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Transpose', 'Reshape', 'Add', 'MatMul'],
                                                   [1, 0, 0, 0, 0])
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        _, mask_nodes, _ = self.model.match_parent_paths(
            add_qk, [(['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Unsqueeze'], [1, 0, 1, 0, 0]),
                     (['Mul', 'Sub', 'Unsqueeze', 'Unsqueeze'], [1, 0, 1, 0])], output_name_to_node)
        if mask_nodes is None:
            logger.debug("fuse_attention: failed to match mask path")
            return

        if (matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_v.input[0] == root_input) or normalize_node.op_type == "Add":
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

            new_node = self.create_attention_node(mask_index, matmul_q, matmul_k, matmul_v, add_q, add_k, add_v,
                                                  root_input, reshape_qkv.output[0])
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)

            self.nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            #self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True
