""" A set of functions to perform message passing on a serialized graph in an LLM """

from collections import defaultdict
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_scatter import scatter

from desequence_graph_ids import SequenceElement


def build_message_passing_matrices(
    token_ids: torch.Tensor,
    edge_sequences: List[List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]]
) -> List[Dict[str, torch.Tensor]]:
    """ Returns the adjacency matrices required to perform causal message passing in between
        language model blocks of an autoregressive language model
    """
    message_passing_dicts = []
    for t_ids, edge_sequence in zip(token_ids, edge_sequences):
        message_passing_dict = {'tokens2edges': [], 'edges2tokens': [], 'inverse_edge_index': []}
        node2edge_idxs = defaultdict(list)
        for edge_idx, sequenced_edge in enumerate(edge_sequence):
            pred_node, edge, succ_node = sequenced_edge
            node2edge_idxs[pred_node.ids].append(edge_idx)
            if isinstance(succ_node, SequenceElement):
                end_idx = succ_node.end_idx
                node2edge_idxs[succ_node.ids].append(edge_idx)
            elif isinstance(edge, SequenceElement):
                end_idx = edge.end_idx
            else:
                end_idx = pred_node.end_idx
            for token_idx in range(pred_node.start_idx, end_idx):
                message_passing_dict['tokens2edges'].append([token_idx, edge_idx])
                message_passing_dict['edges2tokens'].append([edge_idx, token_idx])
        if len(message_passing_dict['edges2tokens']) > 0:
            message_passing_dict['edges2tokens'] = add_missing_idxs(
                message_passing_dict['edges2tokens'],
                num_incoming_nodes=len(edge_sequence),
                num_outgoing_nodes=len(t_ids)
            )
        message_passing_dict['inverse_edge_index'] = []
        for edge_idxs in node2edge_idxs.values():
            if len(edge_idxs) < 2:
                continue
            for (idx0, idx1) in itertools.combinations(list(set(edge_idxs)), 2):
                message_passing_dict['inverse_edge_index'].append(
                    [idx0, idx1] if idx0 < idx1 else [idx1, idx0]
                )
        if len(message_passing_dict['inverse_edge_index']) > 0:
            message_passing_dict['inverse_edge_index'] = add_missing_idxs(
                message_passing_dict['inverse_edge_index'],
                num_incoming_nodes=len(edge_sequence),
                num_outgoing_nodes=len(edge_sequence)
            )
        message_passing_dicts.append({
            key: torch.from_numpy(np.array(value).transpose(1, 0)).long().to(token_ids.device)
            if len(value) > 0 else torch.from_numpy(np.array(value)).long().to(token_ids.device)
            for key, value in message_passing_dict.items()
        })
    return message_passing_dicts


def add_missing_idxs(
    edge_index: List[List[int]],
    *,
    num_incoming_nodes: int,
    num_outgoing_nodes: int
) -> List[List[int]]:
    """ Adds edges from a dummy node to all outgoing nodes which do not have an edge pointing to
        them. This is to facilitate causal message passing where a node should have access to its
        own embedding using torch.scatter to perform message passing.
    """
    existing_idxs = set([node_idxs[-1] for node_idxs in edge_index])
    missing_idxs = set(range(num_outgoing_nodes)) - existing_idxs
    for missing_idx in missing_idxs:
        edge_index.append([num_incoming_nodes, missing_idx])
    return edge_index


def perform_causal_message_passing(
    token_embeddings: torch.Tensor,
    message_passing_dicts: List[Dict[str, torch.Tensor]],
    reduce: str = 'mean'
) -> torch.Tensor:
    """ Returns token embeddings in a sequence where causal message passing has been performed on
        the token ids  based on the serialized graph described in the sequence
    """
    new_token_embeddings = []
    for t_embeddings, message_passing_dict in zip(token_embeddings, message_passing_dicts):
        if message_passing_dict['inverse_edge_index'].numel() == 0:
            new_t_embeddings = t_embeddings
        else:
            edge_embeddings = scatter(
                src=t_embeddings[message_passing_dict['tokens2edges'][0]],
                dim=0,
                index=message_passing_dict['tokens2edges'][1],
                reduce=reduce
            )
            # adding dummy tensor to make sure that the output tensor of message passing is the
            # correct size because causal message passing does not allow self loops
            edge_embeddings = torch.cat([
                edge_embeddings,
                torch.zeros_like(edge_embeddings[0].unsqueeze(0))
            ], dim=0)
            # adding dummy tensor to make sure that the output tensor of message passing is the
            # correct size because causal message passing does not allow self loops
            edge_embeddings = scatter(
                src=edge_embeddings[message_passing_dict['inverse_edge_index'][0]],
                dim=0,
                index=message_passing_dict['inverse_edge_index'][1],
                reduce=reduce
            )
            edge_embeddings = torch.cat([
                edge_embeddings,
                torch.zeros_like(edge_embeddings[0].unsqueeze(0))
            ], dim=0)
            new_t_embeddings = scatter(
                src=edge_embeddings[message_passing_dict['edges2tokens'][0]],
                dim=0,
                index=message_passing_dict['edges2tokens'][1],
                reduce=reduce
            )
        assert new_t_embeddings.shape == t_embeddings.shape
        new_token_embeddings.append(new_t_embeddings.unsqueeze(0))
    return torch.cat(new_token_embeddings, dim=0) + token_embeddings
