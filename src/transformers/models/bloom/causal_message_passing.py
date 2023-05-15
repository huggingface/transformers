""" A module for learning to pass information between elements on a serialized graph in an LLM
    without violating the causality constraint of autoregressive generation (passing information
    backwards in the sequence)
"""

import enum
from functools import partial
from collections import defaultdict
import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_scatter import scatter, scatter_softmax
import torch_geometric

from .desequence_graph_ids import SequenceElement


class GNNLayerFactory(enum.Enum):
    gcn = torch_geometric.nn.GCNConv
    sage = torch_geometric.nn.SAGEConv
    gat = torch_geometric.nn.GATConv


class GatedGraphCrossAttentionLayer(torch.nn.Module):
    """ A module for performing gated cross attention between elements in a graph that
        have been serialized in a sequence of tokens and the token sequence

        a key element of this layer is that it enforces that information about elements in the graph
        can only be passed to tokens describing later elements in the sequence

        This layer contains methods to pass information either between nodes or edges within
        the serialized graph

        This layer is heavily inspired by Flamingo a paper on incorporating image information
        into LLM inference - https://arxiv.org/pdf/2204.14198
    """
    def __init__(self, gnn_type: str, embedding_size: int):
        super().__init__()
        self.gnn_layer = GNNLayerFactory[gnn_type].value(embedding_size, embedding_size)
        self.gating_message_passing = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        token_embeddings: torch.Tensor,
        message_passing_dicts: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        new_token_embeddings = []
        for t_embeddings, message_passing_dict in zip(token_embeddings, message_passing_dicts):
            new_t_embeddings = torch.zeros_like(t_embeddings)
            if message_passing_dict['tokens2elements'].numel() > 0:
                element_embeddings = t_embeddings[message_passing_dict['tokens2elements']]
                if message_passing_dict['edge_index'].numel() > 0:
                    element_embeddings = self.gnn_layer(
                        element_embeddings,
                        message_passing_dict['edge_index']
                    )
                new_t_embeddings[message_passing_dict['elements2tokens']] = element_embeddings
            new_t_embeddings = t_embeddings + torch.tanh(self.gating_message_passing) * new_t_embeddings
            new_token_embeddings.append(new_t_embeddings.unsqueeze(0))
        return torch.cat(new_token_embeddings, dim=0)

    @classmethod
    def build_node_information_passing(
        cls,
        edge_sequences: List[List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]],
        device: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        """ Returns the indice mappings required to perform pass node information in between
            language model blocks of an autoregressive language model for nodes in a serialized
            graph
        """
        message_passing_dicts = []
        for edge_sequence in edge_sequences:
            message_passing_dict = {'tokens2elements': [], 'elements2tokens': [], 'edge_index': []}
            add_node = partial(
                cls.add_node,
                end_idx=cls.get_sequence_end(edge_sequence),
                last_occurence_idx=defaultdict(lambda: -1),
                message_passing_dict=message_passing_dict
            )
            for edge_idx, sequenced_edge in enumerate(edge_sequence):
                pred_node, edge, succ_node = sequenced_edge
                if edge_idx == len(edge_sequence) - 1:
                    if (
                        not isinstance(succ_node, SequenceElement)
                        and not isinstance(edge, SequenceElement)
                    ):
                        continue
                    else:
                        add_node(pred_node)
                else:
                    add_node(pred_node)
                    add_node(succ_node)
            message_passing_dicts.append(cls.to_torch(dict(message_passing_dict), device))
        return message_passing_dicts

    @classmethod
    def build_edge_information_passing(
        cls,
        edge_sequences: List[List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]],
        device: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        """ Returns the indice mappings required to perform pass edge information in between
            language model blocks of an autoregressive language model for nodes in a serialized
            graph
        """
        message_passing_dicts = []
        for edge_sequence in edge_sequences:
            message_passing_dict = {'tokens2elements': [], 'elements2tokens': [], 'edge_index': []}
            node2edge_idxs = defaultdict(list)
            add_edge = partial(
                cls.add_edge,
                end_idx=cls.get_sequence_end(edge_sequence),
                node2edge_idxs=node2edge_idxs,
                message_passing_dict=message_passing_dict
            )
            for sequenced_edge in edge_sequence[:-1]:
                add_edge(sequenced_edge)
            # calculating adjacency matrix between edges (edges in this adjacency matrix always
            # point from edges earlier in the serialized version of the graph to edges later in
            # the graph)
            for edge_idxs in node2edge_idxs.values():
                if len(edge_idxs) < 2:
                    continue
                for (idx0, idx1) in itertools.combinations(list(set(edge_idxs)), 2):
                    message_passing_dict['edge_index'].append(sorted([idx0, idx1]))
            message_passing_dicts.append(cls.to_torch(dict(message_passing_dict), device))
        return message_passing_dicts

    @staticmethod
    def get_sequence_end(
        edge_sequence: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]],
    ) -> int:
        """ Returns last index + 1 of elements in the serialized graph sequence """
        pred_node, edge, succ_node = edge_sequence[-1]
        if isinstance(succ_node, SequenceElement):
            end_idx = succ_node.end_idx
        elif isinstance(edge, SequenceElement):
            end_idx = edge.end_idx
        else:
            end_idx = pred_node.end_idx
        return end_idx

    @classmethod
    def add_node(
        cls,
        current_occurence: SequenceElement,
        end_idx: int,
        last_occurence_idx: Dict[Tuple[int], int],
        message_passing_dict: Dict[str, Union[List[int], List[List[int]]]]
    ):
        """ Each time a node is listed in a serialized version of its corresponding graph, it is
            added as a node in a new artificial graph. This means in the new artificial graph, a
            node in the original graph may appear more than once. For every node added to the
            artificial graph, this function adds an edge which maps between occurences of
            the same node in the original graph if the node has been printed previously in the
            serialized graph. The edge points from the previous occurence to the current occurence.
            i.e. H_1 - O, O - H_2, would create an edge from O -> O since it occurs more than
            once in the graph
        """
        prev_length = len(message_passing_dict[f"tokens2elements"])
        cls.add_element_for_information_passing(
            start_idx=current_occurence.end_idx,
            end_idx=end_idx,
            message_passing_dict=message_passing_dict
        )
        curr_length = len(message_passing_dict[f"tokens2elements"])
        if last_occurence_idx[current_occurence.ids] != -1 and curr_length > prev_length:
            current_idx = len(message_passing_dict["tokens2elements"]) - 1
            message_passing_dict['edge_index'].append(
                [last_occurence_idx[current_occurence.ids], current_idx]
            )
            last_occurence_idx[current_occurence.ids] = current_idx

    @classmethod
    def add_edge(
        cls,
        sequenced_edge: Tuple[SequenceElement, SequenceElement, SequenceElement],
        end_idx: int,
        node2edge_idxs: Dict[Tuple[int], List[int]],
        message_passing_dict: Dict[str, Union[List[int], List[List[int]]]]
    ):
        """ Adds an edge as element to pass information between in a serialized graph """
        pred_node, _, succ_node = sequenced_edge
        prev_length = len(message_passing_dict[f"tokens2elements"])
        cls.add_element_for_information_passing(
            start_idx=succ_node.end_idx,
            end_idx=end_idx,
            message_passing_dict=message_passing_dict
        )
        curr_length = len(message_passing_dict[f"tokens2elements"])
        if curr_length > prev_length:
            current_idx = len(message_passing_dict["tokens2elements"]) - 1
            node2edge_idxs[pred_node.ids].append(current_idx)
            node2edge_idxs[succ_node.ids].append(current_idx)

    @staticmethod
    def add_element_for_information_passing(
        start_idx: int,
        end_idx: int,
        message_passing_dict: Dict[str, Union[List[int], List[List[int]]]]
    ):
        """ Adds an element to the message passing dictionary, the element is either a node
            or an edge. Adding the element means adding the necessary indices to the mapping
            tokens2elements and elements2tokens, so that it is possible to map to elements
            and back
        """
        if start_idx != end_idx:
            message_passing_dict["tokens2elements"].append(start_idx - 1)
            message_passing_dict["elements2tokens"].append(start_idx)

    @staticmethod
    def to_torch(
        array_dict: Dict[str, Union[List[int], List[List[int]]]],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """ Converts a dictionary of lists of integers to a dictionary of torch Tensor and returns it
        """
        for key, array in array_dict.items():
            if len(array) == 0 or isinstance(array[0], int):
                array_dict[key] = torch.from_numpy(np.array(array)).long().to(device)
            else:
                array_dict[key] = torch.from_numpy(np.array(array).transpose(1, 0)).long().to(device)
        return array_dict
