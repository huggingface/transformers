#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from typing import Dict
from logging import getLogger
from onnx import helper
from .onnx_model import OnnxModel
from .fusion_base import Fusion
from .fusion_utils import FusionUtils

logger = getLogger(__name__)


class FusionEmbedLayerNoMask(Fusion):
    """
     Embed Layer Normalization will fuse embeddings and mask processing into one node.
     The embeddings before conversion:

     (input_ids) -------->  Gather ----------+       (segment_ids)
        |                                    |            |
        |                                    v            v
        +--> Shape --> Expand -> Gather---->Add         Gather
        |                ^                   |            |
        |                |                   v            v
        +---(optional graph)               SkipLayerNormalization

      Optional graph is used to generate position list (0, 1, ...) per batch. It can be a constant in some model.

      (input_ids) --> Gather -----+           Slice
                                  |            |
                                  v            v
     (segment_ids)--> Gather --->Add        Reshape
                                  |            |
                                  v            v
                              SkipLayerNormalization
    """
    def __init__(self, model: OnnxModel, description='no mask'):
        super().__init__(model, "EmbedLayerNormalization", "SkipLayerNormalization", description)
        self.utils = FusionUtils(model)
        self.attention = None

    def match_segment_path(self, normalize_node, input_name_to_nodes, output_name_to_node, input_ids_cast_node):
        segment_ids = None
        segment_embedding_gather = None

        segment_embedding_path = self.model.match_parent_path(normalize_node, ['Gather'], [1])

        if segment_embedding_path is None:
            segment_embedding_path = self.model.match_parent_path(normalize_node, ['Add', 'Gather'], [0, 1])
            if segment_embedding_path is None:
                logger.info("Segment embedding is not found. Embed layer cannot be fused.")
                return
            _, segment_embedding_gather = segment_embedding_path
        else:
            segment_embedding_gather = segment_embedding_path[0]

        segment_ids = segment_embedding_gather.input[1]

        self.nodes_to_remove.extend(segment_embedding_path)

        if self.model.find_graph_input(segment_ids):
            casted, segment_ids = self.utils.cast_graph_input_to_int32(segment_ids)
        else:
            segment_ids, segment_ids_cast_node = self.utils.cast_input_to_int32(segment_ids)

            # Cast might be removed by OnnxRuntime.
            _, segment_id_path, _ = self.model.match_parent_paths(
                segment_ids_cast_node,
                [(['ConstantOfShape', 'Concat', 'Unsqueeze', 'Gather', 'Shape', 'Cast'], [0, 0, 1, 0, 0, 0]),
                 (['ConstantOfShape', 'Concat', 'Unsqueeze', 'Gather', 'Shape'], [0, 0, 1, 0, 0])], output_name_to_node)

            if segment_id_path and input_ids_cast_node and input_ids_cast_node.input[0] == segment_id_path[-1].input[0]:
                logger.debug("Simplify semgent id path...")
                self.model.add_node(
                    helper.make_node('Shape', inputs=[input_ids_cast_node.input[0]], outputs=["input_shape"]))
                self.model.add_node(
                    helper.make_node('ConstantOfShape',
                                     inputs=["input_shape"],
                                     outputs=["zeros_for_input_shape"],
                                     value=helper.make_tensor("value", onnx.TensorProto.INT32, [1], [1])))
                segment_ids = "zeros_for_input_shape"

        return segment_ids, segment_embedding_gather

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        is_distill = False

        if self.model.match_parent_path(node, ['Add', 'Gather'], [0, 0]) is None and self.model.match_parent_path(
                node, ['Gather'], [0]) is None:
            logger.debug(
                "Failed to match path SkipLayerNormalization[0] <-- Add <-- Gather or SkipLayerNormalization[0] <-- Gather"
            )
            return

        self.attention = self.model.find_first_child_by_type(node, 'Attention', input_name_to_nodes, recursive=False)
        if self.attention is None:
            # In case user disables attention fusion, check whether subgraph looks like Attention.
            if node.output[0] not in input_name_to_nodes:
                return
            children = input_name_to_nodes[node.output[0]]
            children_types = sorted([child.op_type for child in children])
            if children_types != ['MatMul', 'MatMul', 'MatMul', 'SkipLayerNormalization'] and children_types != [
                    'MatMul', 'MatMul', 'MatMul', 'Shape', 'Shape', 'SkipLayerNormalization'
            ]:
                logger.debug("No Attention like subgraph in children of SkipLayerNormalization")
                return

        # Assume the order of embeddings are word_embedding + position_embedding + segment_embedding
        normalize_node = node
        add_node = None
        word_embedding_path = self.model.match_parent_path(normalize_node, ['Add', 'Gather'], [0, 0])
        if word_embedding_path is not None:
            add_node, word_embedding_gather = word_embedding_path
        else:
            word_embedding_path = self.model.match_parent_path(normalize_node, ['Gather'], [0])
            if word_embedding_path is not None:
                word_embedding_gather = word_embedding_path[0]
                is_distill = True
                from packaging.version import Version
                import onnxruntime
                if Version(onnxruntime.__version__) <= Version("1.4.0"):
                    logger.warning(
                        'Please install onnxruntime with version > 1.4.0 for embedlayer fusion support for distilbert')
                    return
            else:
                logger.info("Word embedding path is not found. Embed layer cannot be fused.")
                return

        input_ids = word_embedding_gather.input[1]

        position_embedding_expand = None
        position_embedding_shape = None

        position_embedding_path = self.model.match_parent_path(normalize_node, ['Gather', 'Expand'],
                                                               [1, 1])  # for distill-bert
        if position_embedding_path is not None:
            position_embedding_weight_node, position_embedding_expand = position_embedding_path
        else:
            position_embedding_path = self.model.match_parent_path(normalize_node, ['Reshape', 'Slice'], [1, 0])
            if position_embedding_path is not None:
                _, position_embedding_weight_node = position_embedding_path
            else:
                position_embedding_path = self.model.match_parent_path(add_node, ['Gather', 'Expand', 'Shape'],
                                                                       [1, 1, 1])
                if position_embedding_path is not None:
                    position_embedding_weight_node, position_embedding_expand, position_embedding_shape = position_embedding_path
                else:
                    position_embedding_path = self.model.match_parent_path(
                        add_node, ['Gather', 'Expand', 'Concat', 'Unsqueeze', 'Gather', 'Shape'], [1, 1, 1, 1, 0, 0])
                    if position_embedding_path is not None:
                        position_embedding_weight_node, position_embedding_expand, _, _, _, position_embedding_shape = position_embedding_path
                    else:
                        # Here we will not try to get exact match. Instead, we only try identify position embedding weights.
                        position_embedding_path = self.model.match_parent_path(add_node, ['Gather', 'Expand'], [1, 1])
                        if position_embedding_path is not None:
                            position_embedding_weight_node, position_embedding_expand = position_embedding_path
                        else:
                            logger.info("Position embedding path is not found. Embed layer cannot be fused.")
                            return

                if position_embedding_shape is not None and position_embedding_shape.input[0] != input_ids:
                    logger.info("position and word embedding is expected to be applied on same input")
                    return

        if position_embedding_expand and position_embedding_shape:
            input_parent = self.model.get_parent(position_embedding_shape, 0, output_name_to_node)
            subgraph_nodes = self.model.get_parent_subgraph_nodes(position_embedding_expand,
                                                                  [input_parent] if input_parent else [],
                                                                  output_name_to_node)
            self.nodes_to_remove.extend(subgraph_nodes)

        self.nodes_to_remove.extend(word_embedding_path)
        self.nodes_to_remove.extend(position_embedding_path)

        self.nodes_to_remove.extend([normalize_node])

        # Cast input_ids and segment_ids to int32.
        input_ids_cast_node = None
        if self.model.find_graph_input(input_ids):
            casted, input_ids = self.utils.cast_graph_input_to_int32(input_ids)
        else:
            input_ids, input_ids_cast_node = self.utils.cast_input_to_int32(input_ids)

        node_name = self.model.create_node_name('EmbedLayerNormalization')
        output_name = node_name + "_output"

        embed_node_inputs = None
        if is_distill == False:
            segment_path = self.match_segment_path(normalize_node, input_name_to_nodes, output_name_to_node,
                                                   input_ids_cast_node)
            if segment_path is None:
                return
            else:
                segment_ids, segment_embedding_gather = segment_path

                embed_node_inputs = [
                    input_ids,
                    segment_ids,
                    word_embedding_gather.input[0],
                    position_embedding_weight_node.input[0],
                    segment_embedding_gather.input[0],
                    normalize_node.input[2],
                    normalize_node.input[3]  # gamma and beta
                ]
        else:
            embed_node_inputs = [
                input_ids,
                '',
                word_embedding_gather.input[0],
                position_embedding_weight_node.input[0],
                '',
                normalize_node.input[2],
                normalize_node.input[3]  # gamma and beta
            ]

        embed_node = helper.make_node('EmbedLayerNormalization',
                                      embed_node_inputs,
                                      outputs=[node_name + "_output", node_name + "_dummy_mask_index"],
                                      name=node_name)

        embed_node.domain = "com.microsoft"

        # Pass attribute "epsilon" from normalize node to EmbedLayerNormalization.
        for att in normalize_node.attribute:
            if att.name == 'epsilon':
                embed_node.attribute.extend([att])
        # Set default value to 1e-12 if no attribute is found.
        # OnnxRuntime 1.2.0 or older has no epsilon attribute. The optimized model can only work for 1.3.0 or later.
        if len(embed_node.attribute) == 0:
            embed_node.attribute.extend([helper.make_attribute("epsilon", 1.0E-12)])

        self.model.replace_input_of_all_nodes(normalize_node.output[0], output_name)
        self.nodes_to_add.append(embed_node)


class FusionEmbedLayerNormalization(FusionEmbedLayerNoMask):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "with mask")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        old_count = len(self.nodes_to_add)

        super().fuse(node, input_name_to_nodes, output_name_to_node)
        if len(self.nodes_to_add) == old_count:
            return

        if self.attention is not None:
            mask_index = self.attention.input[3]
            if mask_index in output_name_to_node:
                node = output_name_to_node[mask_index]
                if node.op_type == "ReduceSum":
                    embed_node = self.nodes_to_add.pop()
                    mask_input_name = node.input[0]
                    self.nodes_to_remove.extend([node])
                    embed_node.input.append(mask_input_name)
                    embed_node.output[1] = mask_index
                    self.nodes_to_add.append(embed_node)

        self.prune_graph = True
