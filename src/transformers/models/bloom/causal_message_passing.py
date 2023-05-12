""" A set of functions to perform message passing on a serialized graph in an LLM """

import enum
from collections import defaultdict
import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric

from .desequence_graph_ids import SequenceElement


class GNNLayerFactory(enum.Enum):
    gcn = torch_geometric.nn.GCNConv
    sage = torch_geometric.nn.SAGEConv
    gat = torch_geometric.nn.GATConv


def build_message_passing_matrices(
    token_ids: torch.Tensor,
    edge_sequences: List[List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]]
) -> List[Dict[str, torch.Tensor]]:
    """ Returns the adjacency matrices required to perform causal message passing in between
        language model blocks of an autoregressive language model
    """
    message_passing_dicts = []
    for edge_sequence in edge_sequences:
        message_passing_dict = defaultdict(list)
        node2edge_idxs = defaultdict(list)
        prev_node_idx = defaultdict(lambda: -1)

        def add_element(end_idx: int, element_type: str):
            """ Adds an element to the edge or node graphs used for message passing """
            assert element_type in ['nodes', 'edges']
            message_passing_dict[f"tokens2{element_type}"].append(end_idx - 1)
            message_passing_dict[f"{element_type}2tokens"].append(end_idx)

        for edge_idx, sequenced_edge in enumerate(edge_sequence):
            pred_node, edge, succ_node = sequenced_edge
            if edge_idx == len(edge_sequence) - 1:
                if (
                    not isinstance(succ_node, SequenceElement)
                    and not isinstance(edge, SequenceElement)
                ):
                    continue
                else:
                    add_element(pred_node.end_idx, 'nodes')
                    num_nodes = len(message_passing_dict["tokens2nodes"])
                    if prev_node_idx[pred_node.ids] != -1:
                        message_passing_dict['edge_index_nodes'].append(
                            [prev_node_idx[pred_node.ids], num_nodes - 1]
                        )
            else:
                add_element(pred_node.end_idx, 'nodes')
                add_element(succ_node.end_idx, 'edges')
                add_element(succ_node.end_idx, 'nodes')
                node2edge_idxs[pred_node.ids].append(edge_idx)
                node2edge_idxs[succ_node.ids].append(edge_idx)
                num_nodes = len(message_passing_dict["tokens2nodes"])
                message_passing_dict['edge_index_nodes'].append([num_nodes - 2, num_nodes - 1])
                if prev_node_idx[pred_node.ids] != -1:
                    message_passing_dict['edge_index_nodes'].append(
                        [prev_node_idx[pred_node.ids], num_nodes - 2]
                    )
                if prev_node_idx[succ_node.ids] != -1:
                    message_passing_dict['edge_index_nodes'].append(
                        [prev_node_idx[succ_node.ids], num_nodes - 1]
                    )
                prev_node_idx[pred_node.ids] = num_nodes - 2
                prev_node_idx[succ_node.ids] = num_nodes - 1

        for edge_idxs in node2edge_idxs.values():
            if len(edge_idxs) < 2:
                continue
            for (idx0, idx1) in itertools.combinations(list(set(edge_idxs)), 2):
                message_passing_dict['edge_index_edges'].append(sorted([idx0, idx1]))

        def to_torch(array: Union[List[int], List[List[int]]]) -> torch.Tensor:
            """ Converts an array to a torch Tensor and returns it"""
            if len(array) == 0 or isinstance(array[0], int):
                return torch.from_numpy(np.array(array)).long().to(token_ids.device)
            else:
                return torch.from_numpy(np.array(array).transpose(1, 0)).long().to(token_ids.device)

        message_passing_dict['tokens2edges'] = to_torch(message_passing_dict['tokens2edges'])
        message_passing_dict['edges2tokens'] = to_torch(message_passing_dict['edges2tokens'])
        message_passing_dict['tokens2nodes'] = to_torch(message_passing_dict['tokens2nodes'])
        message_passing_dict['nodes2tokens'] = to_torch(message_passing_dict['nodes2tokens'])
        message_passing_dict['edge_index_nodes'] = to_torch(message_passing_dict['edge_index_nodes'])
        message_passing_dict['edge_index_edges'] = to_torch(message_passing_dict['edge_index_edges'])
        message_passing_dicts.append(dict(message_passing_dict))
    return message_passing_dicts


class CausalMessagePassingLayer(torch.nn.Module):
    """ A torch.nn.Module for performing causal message passing within an autoregressive
        language model
    """
    def __init__(self, gnn_type: str, embedding_size: int):
        super().__init__()
        self.nodes_layer = GNNLayerFactory[gnn_type].value(embedding_size, embedding_size)
        self.edges_layer = GNNLayerFactory[gnn_type].value(embedding_size, embedding_size)
        self.gating_parameter_a = torch.nn.Parameter(torch.zeros(1))
        self.gating_parameter_b = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        token_embeddings: torch.Tensor,
        message_passing_dicts: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        new_token_embeddings = []
        for t_embeddings, message_passing_dict in zip(token_embeddings, message_passing_dicts):
            token_edges_embeddings = torch.zeros_like(t_embeddings)
            token_nodes_embeddings = torch.zeros_like(t_embeddings)
            if message_passing_dict['tokens2edges'].numel() > 0:
                edges_embeddings = t_embeddings[message_passing_dict['tokens2edges']]
                if message_passing_dict['edge_index_edges'].numel() > 0:
                    edges_embeddings = self.edges_layer(
                        edges_embeddings,
                        message_passing_dict['edge_index_edges']
                    )
                token_edges_embeddings[message_passing_dict['edges2tokens']] = edges_embeddings
            if message_passing_dict['tokens2nodes'].numel() > 0:
                nodes_embeddings = t_embeddings[message_passing_dict['tokens2nodes']]
                if message_passing_dict['edge_index_nodes'].numel() > 0:
                    nodes_embeddings = self.nodes_layer(
                        nodes_embeddings,
                        message_passing_dict['edge_index_nodes']
                    )
                token_nodes_embeddings[message_passing_dict['nodes2tokens']] = nodes_embeddings
            new_t_embeddings = (
                t_embeddings
                + torch.tanh(self.gating_parameter_a) * token_edges_embeddings
                + torch.tanh(self.gating_parameter_b) * token_nodes_embeddings
            )
            new_token_embeddings.append(new_t_embeddings.unsqueeze(0))
        return torch.cat(new_token_embeddings, dim=0)
