""" A set of functions to modify the position embeddings of BLOOM to edit position embeddings in
    a sequence to describe the order generation position invariance of a serialized graph
"""

from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .desequence_graph_ids import SequenceElement


def build_alibi_tensor(
    token_ids: torch.Tensor,
    edge_sequences: List[List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]],
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
    for (t_ids, edge_sequence, mask, positions) in zip(
        token_ids,
        edge_sequences,
        attention_mask,
        arange_tensor
    ):
        token_positions.append(
            _get_graph_positions(t_ids, edge_sequence, mask, positions, position_type)
        )
    alibi = (
        slopes.tile([token_ids.shape[0]]).unsqueeze(1).unsqueeze(2)
        * torch.cat(token_positions, dim=0).tile([num_heads, 1, 1])
    )
    return alibi.reshape(
        batch_size * num_heads,
        seq_length if position_type in ['all_edges_previous', 'action_invariant']  else 1,
        seq_length
    ).to(dtype)


def _get_graph_positions(
    token_ids: torch.Tensor,
    edge_sequence: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]],
    mask: torch.Tensor,
    positions: torch.Tensor,
    position_type: str
) -> torch.Tensor:
    """ Returns a revised position tenso where the position of tokens in the
        sequence reflect the position of nodes and edges in a graph
    """
    if position_type == 'normal':
        return positions.unsqueeze(0)
    assert len(token_ids.shape) == 1, f"Incorrectly sized tensor - {token_ids.shape}"
    if len(edge_sequence) > 0:
        assert len(positions.shape) == 2 and positions.size(0) == 1
        assert (
            token_ids.shape[0] == positions.shape[1]
        ), f"Tensors must be the same shape {positions.shape}   {token_ids.shape}"
        if position_type == 'action_invariant':
            new_positions = _get_action_invariant_positions(
                positions=positions[0],
                edge_sequence=edge_sequence
            ) * mask
        elif position_type == 'all_edges_previous':
            new_positions = _get_all_edge_previous_positions(
                positions=positions[0],
                edge_sequence=edge_sequence
            ) * mask
        elif position_type == 'no_positions':
            new_positions = _remove_positions_from_graph_tokens(
                positions=positions[0],
                edge_sequence=edge_sequence,
            ) * mask
        else:
            raise ValueError(f"Unknown position embedding type {position_type}")
        return new_positions.unsqueeze(0)
    else:
        return positions.unsqueeze(0)


def _remove_positions_from_graph_tokens(
    positions: torch.Tensor,
    edge_sequence: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]
) -> torch.Tensor:
    """ Returns a revised position tenso where all token ids in a serialized graph are given the
        same position
    """
    start_idx = edge_sequence[0][0].start_idx
    if edge_sequence[-1][2] is None and edge_sequence[-1][1] is None:
        end_idx = edge_sequence[-1][0].end_idx
    elif edge_sequence[-1][2] is None:
        end_idx = edge_sequence[-1][1].end_idx
    else:
        end_idx = edge_sequence[-1][2].end_idx
    init_position = positions[start_idx].item()
    new_positions = positions.clone()
    new_positions[start_idx:end_idx] = init_position # setting all isolated graph tokens to have a low bias
    return new_positions.unsqueeze(0)


def _get_action_invariant_positions(
    positions: torch.Tensor,
    edge_sequence: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]
) -> torch.Tensor:
    """ Returns a revised position tensor where the position of tokens in a graph sequence reflect
        the order in which they can be generated, i.e. some edges can be generated at the same time
        so they will have similar positions
    """
    position_vector = positions[:edge_sequence[0][0].start_idx].tolist()
    element_position = defaultdict(lambda: -1)
    new_positions = [position_vector[:] for _ in range(edge_sequence[0][0].start_idx)]
    for sequenced_edge in edge_sequence:
        pred_node, edge, succ_node = sequenced_edge
        position_vector += list(range(
            max(position_vector) + 1,
            max(position_vector) + 1 + pred_node.length
        ))
        new_positions += [position_vector[:] for _ in range(pred_node.length)]
        if not isinstance(edge, SequenceElement):
            continue
        if element_position[pred_node.ids] > 0:
            position_vector = position_vector[:pred_node.start_idx]
            position_vector += list(range(
                element_position[pred_node.ids],
                pred_node.length + element_position[pred_node.ids]
            ))
        position_vector += list(range(
            position_vector[-1] + 1,
            position_vector[-1] + 1 + edge.length
        ))
        new_positions += [position_vector[:] for _ in range(edge.length)]
        if not isinstance(succ_node, SequenceElement):
            continue
        position_vector += list(range(
            position_vector[-1] + 1,
            position_vector[-1] + 1 + succ_node.length
        ))
        new_positions += [position_vector[:] for _ in range(succ_node.length)]
        if element_position[pred_node.ids] < -1:
            element_position[pred_node.ids] = position_vector[-1] + 1
        if element_position[succ_node.ids] < -1:
            element_position[succ_node.ids] = position_vector[-1] + 1
    length = positions.numel()
    new_positions += [[0] * length for _ in range(length - len(new_positions[-1]))]
    new_positions = [n_pos + (length - len(n_pos)) * [0] for n_pos in new_positions]
    new_positions = torch.from_numpy(np.array(new_positions)).long().to(positions.device)
    return new_positions


def _get_all_edge_previous_positions(
    positions: torch.Tensor,
    edge_sequence: List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]
) -> torch.Tensor:
    """ Returns a revised position tensor where the position of tokens in a graph sequence reflect
        the order in which they can be generated, i.e. some edges can be generated at the same time
        so they will have similar positions
    """
    text_positions = positions[:edge_sequence[0][0].start_idx].tolist()
    new_positions = [text_positions[:] for _ in range(edge_sequence[0][0].start_idx)]
    previous_lengths = []
    for sequenced_edge in edge_sequence:
        pred_node, edge, succ_node = sequenced_edge
        start_idx = pred_node.start_idx
        if isinstance(succ_node, SequenceElement):
            end_idx = succ_node.end_idx
        elif isinstance(edge, SequenceElement):
            end_idx = edge.end_idx
        else:
            end_idx = pred_node.end_idx
        current_length = end_idx - start_idx
        position_vector = text_positions[:]
        max_length = max(previous_lengths) if len(previous_lengths) > 0 else 0
        for previous_length in previous_lengths:
            position_vector += list(range(
                text_positions[-1] + 1 + max_length - previous_length,
                text_positions[-1] + 1 + max_length
            ))
        position_vector += list(range(
            text_positions[-1] + 1 + max_length,
            text_positions[-1] + 1 + max_length + current_length
        ))
        previous_lengths.append(current_length)
        new_positions += [position_vector[:] for _ in range(current_length)]
    length = positions.numel()
    new_positions += [[0] * length for _ in range(length - len(new_positions[-1]))]
    new_positions = [n_pos + (length - len(n_pos)) * [0] for n_pos in new_positions]
    new_positions = torch.from_numpy(np.array(new_positions)).long().to(positions.device)
    return new_positions
