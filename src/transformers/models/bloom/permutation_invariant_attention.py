""" Implementing permutation invariant masking """

from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import BloomTokenizerFast

from data.base_dataset import TextGraph
from models.serialize import GEN_NODE_TOKEN, NODE_TOKEN, EDGE_TOKEN, SerializedGraphGenerator, S_TOKENS


TEST_TOKENIZER = BloomTokenizerFast.from_pretrained('bigscience/bloom-3b')
TEST_TOKENIZER.padding_side = 'left'
TEST_TOKENIZER.add_tokens(S_TOKENS)

MOLECULE_GRAPH = TextGraph(
    text="a made up molecule that is probably impossible",
    nodes=['C', 'O', 'N', 'C', 'C', 'N'],
    # contains cycle
    edge_index=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [2, 4], [3, 5]],
    edges=['SINGLE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'AROMATIC', 'SINGLE', 'SINGLE', 'SINGLE']
)

ABSTRACT_SYNTAX_TREE = TextGraph(
    text="a made up function which is probably impossible",
    nodes=['FUNCTION_DEF', 'FOR', 'WHILE', 'VARIABLE', 'ADDITION', 'VARIABLE', 'TYPE'],
    # tree structure
    edge_index=[[0, 1], [0, 2], [1, 3], [2, 4], [1, 5], [4, 6]],
    edges=['A', 'B', 'C', 'D', 'E', 'F']
)

KNOWLEDGE_GRAPH = TextGraph(
    text="a made up knowledge graph which probably contains incorrect syntax",
    nodes=['sun', 'earth', 'plant', 'water'],
    # contains self loop
    edge_index=[[0, 0], [0, 1], [1, 2], [2, 0], [3, 0]],
    edges=['iscalled', 'orbitsthe', 'containedin', 'needs', 'needs'],
)

NESTED_CYCLES = TextGraph(
    text='this graph has a cycle nested in another cycle',
    nodes=['0', '1', '2', '3', '4', '5', '6'],
    edge_index=[[0, 1],[1, 2], [1, 3], [2, 4], [3, 4], [4, 5], [5, 6], [6, 0]],
    edges=['0', '1', '2', '3', '4', '5', '6', '7'],
)

CHAIN_WITH_CYCLES = TextGraph(
    text='this graph has a chain with a cycle in it, so not really a chain',
    nodes=['0', '1', '2', '3', '4', '5', '6'],
    edge_index=[[0, 1],[1, 2], [1, 3], [2, 4], [3, 4], [4, 5], [5, 6]],
    edges=['0', '1', '2', '3', '4', '5', '6'],
)

MOLECULE_GRAPH_TWO = TextGraph(
    text='this molecule has only carbon atoms',
    nodes=['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
    edge_index=[[0, 3], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8]],
    edges=['SINGLE', 'SINGLE', 'SINGLE', 'SINGLE', 'SINGLE', 'SINGLE', 'SINGLE', 'SINGLE'],
)

GRAPH_BATCH = [
    MOLECULE_GRAPH,
    KNOWLEDGE_GRAPH,
    ABSTRACT_SYNTAX_TREE,
    MOLECULE_GRAPH_TWO,
    NESTED_CYCLES,
    CHAIN_WITH_CYCLES
]

GRAPH_TOKENS = {
    'graph': TEST_TOKENIZER.convert_tokens_to_ids([GEN_NODE_TOKEN, NODE_TOKEN, EDGE_TOKEN]),
    'nodes': TEST_TOKENIZER.convert_tokens_to_ids([GEN_NODE_TOKEN, NODE_TOKEN]),
    'gen_node': TEST_TOKENIZER.convert_tokens_to_ids([GEN_NODE_TOKEN]),
    'node': TEST_TOKENIZER.convert_tokens_to_ids([NODE_TOKEN]),
    'edge': TEST_TOKENIZER.convert_tokens_to_ids([EDGE_TOKEN]),
    'eos': [TEST_TOKENIZER.pad_token_id, TEST_TOKENIZER.eos_token_id]
}


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


def build_alibi_tensor(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_heads: int,
    dtype: torch.dtype,
    graph_tokens: Dict[str, List[int]],
    position_type: str = 'vanilla'
) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.
    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    token_positions = []
    for (t_ids, mask, positions) in enumerate(zip(token_ids, attention_mask, arange_tensor)):
        token_positions.append(
            _get_graph_positions(t_ids, mask, positions, graph_tokens, position_type)
        )
    alibi = slopes[..., None] * torch.cat(token_positions, dim=0)
    import pdb;pdb.set_trace()
    return alibi.reshape(
        batch_size * num_heads,
        seq_length if position_type == 'all_edges_previous' else 1,
        seq_length
    ).to(dtype)


def _get_graph_positions(
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    graph_tokens: Dict[str, List[int]],
    position_type: str
) -> torch.Tensor:
    """ Returns a revised position tensor (arange_tensor) where the position of tokens in the
        sequence reflect the position of nodes and edges in a graph
    """
    if position_type == 'vanilla':
        return positions
    assert len(token_ids.shape) == 1, f"Incorrectly sized tensor - {token_ids.shape}"
    edges = _extract_graph(token_ids.int().tolist(), graph_tokens)
    if len(edges) > 0:
        assert len(positions.shape) == 2 and positions.size(0) == 1
        assert token_ids.shape[0] == positions.shape[1], f"Tensors must be the same shape {positions.shape}   {token_ids.shape}"
        if position_type == 'action_invariant':
            new_positions = _get_action_invariant_positions(
                positions=positions[0],
                sequenced_edges=edges,
                graph_tokens=graph_tokens,
            ) * mask
        elif position_type == 'no_positions':
            new_positions = _remove_positions_from_graph_tokens(
                positions=positions[0],
                sequenced_edges=edges,
            ) * mask
        elif position_type == 'all_edges_previous':
            new_positions = _get_all_edges_previous_positions(
                positions=positions[0],
                sequenced_edges=edges,
            ) * mask
        return new_positions.unsqueeze(0)
    else:
        return positions


def _get_all_edges_previous_positions(
    positions: torch.Tensor,
    sequenced_edges: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]
) -> torch.Tensor:
    start_idx = sequenced_edges[0][0].start_idx
    init_position = positions[start_idx].item()
    new_positions_list = []
    for s_idx, sequenced_edge in enumerate(sequenced_edges):
        new_positions = positions.clone()
        max_length = 0
        for prev_edge in sequenced_edges[:s_idx]:
            pred_node, _, succ_node = prev_edge
            new_positions[pred_node.start_idx:succ_node.end_idx] = torch.arange(start_idx - end_idx, 0, -1).to(positions.device)
            max_length = max(end_idx - start_idx, max_length)
        pred_node, edge, succ_node = sequenced_edge
        if succ_node is None and edge is None:
            end_idx = pred_node.end_idx
        elif succ_node is None:
            end_idx = edge.end_idx
        else:
            end_idx = succ_node.end_idx
        new_positions[pred_node.start_idx:end_idx] = torch.arange(end_idx - start_idx).to(positions.device)
        new_positions[start_idx:end_idx] += init_position + max_length
        for idx in range(pred_node.start_idx, end_idx):
            token_positions = new_positions.clone()
            if idx < token_positions.numel():
                token_positions[(idx + 1):] = -1e6
            new_positions_list.append(token_positions.unsqueeze(0))
    return torch.cat([new_positions_list], dim=0)


def _remove_positions_from_graph_tokens(
    positions: torch.Tensor,
    sequenced_edges: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]
) -> torch.Tensor:
    start_idx = sequenced_edges[0][0].start_idx
    if sequenced_edges[-1][2] is None and sequenced_edges[-1][1] is None:
        end_idx = sequenced_edges[-1][0].end_idx
    elif sequenced_edges[-1][2] is None:
        end_idx = sequenced_edges[-1][1].end_idx
    else:
        end_idx = sequenced_edges[-1][2].end_idx
    init_position = positions[start_idx].item()
    new_positions = positions.clone()
    new_positions[start_idx:end_idx] = init_position # setting all isolated graph tokens to have a low bias
    return new_positions


def _get_action_invariant_positions(
    positions: torch.Tensor,
    sequenced_edges: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]],
    graph_tokens: Dict[str, List[int]]
) -> torch.Tensor:
    """ Returns a revised position tensor where the position of tokens in a graph sequence reflect
        their position in the graph
    """
    device = positions.device
    init_position = positions[sequenced_edges[0][0].start_idx].item()
    new_positions = positions.clone()
    element_position = defaultdict(lambda: -1)
    for sequenced_edge in sequenced_edges:
        pred_node, edge, succ_node = sequenced_edge
        pred_bias = (
            init_position if element_position[pred_node.ids] < 0
            else element_position[pred_node.ids]
        )
        new_positions[pred_node.start_idx:pred_node.end_idx] = (
            pred_bias + torch.arange(pred_node.length).to(device)
        )
        if isinstance(edge, SequenceElement):
            new_positions[edge.start_idx:edge.end_idx] = (
                new_positions[pred_node.end_idx - 1] + 1 + torch.arange(edge.length).to(device)
            )
        if isinstance(edge, SequenceElement) and isinstance(succ_node, SequenceElement):
            new_positions[succ_node.start_idx:succ_node.end_idx] = (
                new_positions[edge.end_idx - 1] + 1 + torch.arange(succ_node.length).to(device)
            )
        after_edge_pos = new_positions[succ_node.end_idx - 1] + 1
        if pred_node.token in graph_tokens['gen_node']:
            element_position[pred_node.ids] = after_edge_pos.item()
        if isinstance(succ_node, SequenceElement) and succ_node.token in graph_tokens['gen_node']:
            element_position[succ_node.ids] = after_edge_pos.item()
    return new_positions


def _extract_graph(
    token_ids: List[int],
    graph_tokens: Dict[str, List[int]]
) -> List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]:
    """ Returns a list of edges of the graph sequence identified in a sequence of token ids """
    sequence = _extract_graph_sequence(token_ids, graph_tokens)
    edges = []
    if len(sequence) > 2:
        node_explored = defaultdict(lambda: False)
        for elem0, elem1, elem2 in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
            if (
                elem0.token in graph_tokens['nodes']
                and elem1.token in graph_tokens['edge']
                and elem2.token in graph_tokens['nodes']
            ): # edge syntax
                if (
                # test to see if there is an ungenerated node contained within the identified edge
                # essentially another syntax error that can occur during graph generation
                    (elem0.token in graph_tokens['node'] and not node_explored[elem0.ids])
                    or (
                        elem2.token in graph_tokens['node']
                        and not node_explored[elem2.ids]
                        and elem2.ids != elem0.ids # to account for self loops
                    )
                ):
                    continue
                if elem0.token in graph_tokens['gen_node'] and not node_explored[elem0.ids]:
                    node_explored[elem0.ids] = True
                if elem2.token in graph_tokens['gen_node'] and not node_explored[elem2.ids]:
                    node_explored[elem2.ids] = True
                edges.append((elem0, elem1, elem2))
    if (
        (
            len(edges) > 0
            and edges[-1].pred_node != sequence[-3]
            and edges[-1].edge != sequence[-2]
            and edges[-1].succ_node != sequence[-1]
        )
        or len(edges) == 0
    ):
        if (
            len(sequence) > 1
            and sequence[-2].token in graph_tokens['nodes']
            and sequence[-1].token in graph_tokens['edge']
            and not (
                sequence[-2].token in graph_tokens['node']
                and not node_explored[sequence[-2].ids]
            )
        ):

            edges.append((sequence[-2], sequence[-1], None))
        elif (
            len(sequence) > 0 and sequence[-1].token in graph_tokens['nodes']
            and not (
                sequence[-1].token in graph_tokens['node']
                and not node_explored[sequence[-1].ids]
            )
        ):
            edges.append((sequence[-1], None, None))
    return edges


def _extract_graph_sequence(
    token_ids: List[int],
    graph_tokens: Dict[str, List[int]]
) -> List[SequenceElement]:
    """ Returns a parsable representation of the serialized graph in a sequence of token ids,
        if none is found, returns an empty list
    """
    sequence = []
    prev_token_id, prev_idx, final_idx = None, -1, len(token_ids)
    for token_idx, token_id in enumerate(token_ids):
        if token_id in graph_tokens['gen_node'] and prev_token_id is None:
            prev_token_id, prev_idx = token_id, token_idx
        elif token_id in graph_tokens['graph'] and prev_token_id is not None:
            sequence.append(SequenceElement(
                token=prev_token_id,
                start_idx=prev_idx,
                end_idx=token_idx,
                ids=tuple(token_ids[prev_idx:token_idx])[1:],
                length=token_idx - prev_idx
            ))
            prev_token_id, prev_idx = token_id, token_idx
        elif token_id in graph_tokens['eos'] and prev_token_id is not None:
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


if __name__ == "__main__":
    graphs_batch = SerializedGraphGenerator.text_graph2inputs_batch(
        text_graph_pairs=GRAPH_BATCH,
        tokenizer=TEST_TOKENIZER,
        add_directions=False,
        randomize=False
    )
    alibi = build_alibi_tensor(
        token_ids=graphs_batch['input_sequence'],
        attention_mask=graphs_batch['input_attn_mask'],
        num_heads=8,
        dtype=graphs_batch['input_sequence'].dtype,
        graph_tokens=GRAPH_TOKENS
    )
