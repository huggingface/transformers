#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
from logging import getLogger
from .onnx_model import OnnxModel
from typing import Union, List

logger = getLogger(__name__)


class Fusion:
    def __init__(self,
                 model: OnnxModel,
                 fused_op_type: str,
                 search_op_types: Union[str, List[str]],
                 description: str = None):
        self.search_op_types: List[str] = [search_op_types] if isinstance(search_op_types, str) else search_op_types
        self.fused_op_type: str = fused_op_type
        self.description: str = f"{fused_op_type}({description})" if description else fused_op_type
        self.model: OnnxModel = model
        self.nodes_to_remove: List = []
        self.nodes_to_add: List = []
        self.prune_graph: bool = False

    def apply(self):
        logger.debug(f"start {self.description} fusion...")
        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        # This assumes that two search ops will not be fused at same time!
        for search_op_type in self.search_op_types:
            for node in self.model.get_nodes_by_op_type(search_op_type):
                self.fuse(node, input_name_to_nodes, output_name_to_node)

        op_list = [node.op_type for node in self.nodes_to_add]
        count = op_list.count(self.fused_op_type)
        if count > 0:
            logger.info(f"Fused {self.description} count: {count}")

        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add)

        if self.prune_graph:
            self.model.prune_graph()
        elif self.nodes_to_remove or self.nodes_to_add:
            self.model.update_graph()
