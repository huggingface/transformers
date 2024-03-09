"""copied from BLOOM's code with some minor changes"""

import math

import torch
import torch.nn as nn


class Alibi(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads

        closest_power_of_2 = 2 ** math.floor(math.log2(self.num_heads))
        base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != self.num_heads:
            extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, self.num_heads - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        self.register_buffer("slopes", slopes, persistent=False)

    def forward(
        self, attention_mask: torch.Tensor, batch_size: int, key_length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
        relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
        `softmax(l+a) = softmax(l)`. Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

        Args:
            attention_mask (torch.Tensor): attention_mask tensor of shape (`batch_size`, `key_length`)
            num_heads (int): `num_heads` for the model
            batch_size (int): `batch_size`
            key_length (int): `key_length`
            device (torch.device): device for the tensors
            dtype (torch.dtype): dtype to use for the tensors

        Returns:
            torch.Tensor: alibi tensor of shape (`batch_size`, `num_heads`, `key_length`)
        """

        # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
        # => the query_length dimension will then be broadcasted correctly
        # This is more or less identical to T5's relative position bias:
        # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
        if attention_mask is None:
            arange_tensor = (
                torch.arange(key_length, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            )
        else:
            arange_tensor = (attention_mask.cumsum(dim=-1) - 1).masked_fill_(attention_mask == 0, 0).unsqueeze(1)

        alibi = self.slopes.unsqueeze(1) * arange_tensor
        return alibi.to(dtype)
