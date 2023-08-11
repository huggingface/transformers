""" A set of functions to identify a serialized graph within a list of token ids """

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SequenceElement:
    """ A data class for representing an element in a sequence which is an element of a serialized
        graph
    """
    token: int
    start_idx: int
    end_idx: int
    ids: Tuple[int]
    length: int


def extract_edge_sequence(
    token_ids: List[int],
    graph_tokens: Dict[str, List[int]]
) -> List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]:
    """ Returns a list of edges of the graph sequence identified in a sequence of generated token ids. """
    sequence = _extract_graph_elements(token_ids, graph_tokens)
    edges = []
    if len(sequence) > 2:
        for elem0, elem1, elem2 in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
            if (
                elem0.token == graph_tokens['pred_node']
                and elem1.token == graph_tokens['edge']
                and elem2.token == graph_tokens['succ_node']
            ): # edge syntax
                edges.append((elem0, elem1, elem2))
    if (
        len(sequence) > 1
        and sequence[-2].token == graph_tokens['pred_node']
        and sequence[-1].token == graph_tokens['edge']
    ):
        edges.append((sequence[-2], sequence[-1], None))
    elif len(sequence) > 0 and sequence[-1].token == graph_tokens['pred_node']:
        edges.append((sequence[-1], None, None))
    return edges


def _extract_graph_elements(
    token_ids: List[int],
    graph_tokens: Dict[str, List[int]]
) -> List[SequenceElement]:
    """ Returns a parsable representation of the serialized graph in a sequence of token ids,
        if none is found, returns an empty list
    """
    sequence = []
    prev_token_id, prev_idx, final_idx = None, -1, len(token_ids)
    for token_idx, token_id in enumerate(token_ids):
        if token_id == graph_tokens['pred_node'] and prev_token_id is None:
            prev_token_id, prev_idx = token_id, token_idx
        elif (
            token_id in [graph_tokens['pred_node'], graph_tokens['edge'], graph_tokens['succ_node']]
            and prev_token_id is not None
        ):
            sequence.append(SequenceElement(
                token=prev_token_id,
                start_idx=prev_idx,
                end_idx=token_idx,
                ids=tuple(token_ids[prev_idx:token_idx])[1:],
                length=token_idx - prev_idx
            ))
            prev_token_id, prev_idx = token_id, token_idx
        elif token_id in [graph_tokens['eos'], graph_tokens['pad']] and prev_token_id is not None:
            final_idx = token_idx
            break
    if prev_token_id is not None:
        sequence.append(SequenceElement(
            token=prev_token_id,
            start_idx=prev_idx,
            end_idx=final_idx,
            ids=tuple(token_ids[prev_idx:final_idx])[1:],
            length=final_idx - prev_idx
        ))
    return sequence
